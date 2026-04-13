"""Column-Packed Householder QR over CKKS (OpenFHE-Python).

R packed as n column ciphertexts (each packing m values), Q as m column ciphertexts.
For 150x4: only 4 R ciphertexts instead of 150 row ciphertexts.
Sign assumed +1, zero decryptions inside the Householder loop.
Chebyshev domains derived per-step from plaintext simulation.
"""

from __future__ import annotations

import math
from typing import Tuple

from openfhe import *

from .utils import (
    encrypt_matrix_cols,
    encrypt_identity_cols,
    safe_rotate,
    sum_slots,
    he_sqrt,
    he_inv,
    safe_rotate,
    replicate_slot_0,
    sum_slots,
    simulate_norms
)


def setup_crypto_context(mult_depth: int, N: int = 8192) -> Tuple:
    """CKKS context with rotation keys for column extraction (1..150), slot-0 broadcast, and small negative shifts."""
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    params.SetRingDim(N)
    params.SetBatchSize(N // 2)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    log_slots = int(math.ceil(math.log2(N // 2)))
    rot_indices = list(range(1, 151)) + [-(1 << i) for i in range(log_slots)] + [-3]
    cc.EvalRotateKeyGen(keys.secretKey, rot_indices)

    return cc, keys

def householder_step_fhe_col(
    cc,
    keys,
    R_cts: list,
    Q_cols: list,
    k: int,
    m: int,
    n: int,
    norm_sq_lo: float,
    norm_sq_hi: float,
    vtv_lo: float,
    vtv_hi: float,
    D_sqrt: int = 16,
    D_inv: int = 16,
):
    """One Householder reflection at pivot k with column-packed R (sign=+1, zero decryptions). Updates R_cts and Q_cols in-place."""
    slots = cc.GetRingDimension() // 2
    length = m - k

    mask_k_vec = [0.0] * slots
    for i in range(k, m):
        mask_k_vec[i] = 1.0
    mask_k = cc.MakeCKKSPackedPlaintext(mask_k_vec)

    e_0_vec = [0.0] * slots
    e_0_vec[0] = 1.0
    ptxt_e0 = cc.MakeCKKSPackedPlaintext(e_0_vec)

    x_ct = cc.EvalMult(R_cts[k], mask_k)

    x_sq = cc.EvalMult(x_ct, x_ct)
    norm_sq_raw = sum_slots(cc, x_sq, m)
    norm_sq_slot0 = cc.EvalMult(norm_sq_raw, ptxt_e0)

    norm_ct = he_sqrt(cc, norm_sq_slot0, norm_sq_lo, norm_sq_hi, D_sqrt)
    norm_ct = cc.EvalMult(norm_ct, ptxt_e0)

    x0_slot0 = cc.EvalMult(safe_rotate(cc, x_ct, k), ptxt_e0)
    v0_ct = cc.EvalAdd(x0_slot0, norm_ct)

    vtv_ct = cc.EvalMult(cc.EvalMult(norm_ct, 2.0), v0_ct)

    two_over_vtv_ct = he_inv(cc, vtv_ct, vtv_lo, vtv_hi, D_inv)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, 2.0)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, ptxt_e0)

    norm_at_k = safe_rotate(cc, norm_ct, -k)
    v_ct = cc.EvalAdd(x_ct, norm_at_k)

    tau_bc = replicate_slot_0(cc, two_over_vtv_ct, slots)
    tau_v_ct = cc.EvalMult(tau_bc, v_ct)

    for j in range(n):
        dot_ct = cc.EvalMult(v_ct, R_cts[j])
        w_j_raw = sum_slots(cc, dot_ct, m)
        w_j_slot0 = cc.EvalMult(w_j_raw, ptxt_e0)
        w_j_bc = replicate_slot_0(cc, w_j_slot0, slots)
        R_cts[j] = cc.EvalSub(R_cts[j], cc.EvalMult(tau_v_ct, w_j_bc))

    v_bc = []
    for j in range(length):
        vj_slot0 = cc.EvalMult(safe_rotate(cc, v_ct, k + j), ptxt_e0)
        v_bc.append(replicate_slot_0(cc, vj_slot0, slots))

    d_ct = cc.EvalMult(v_bc[0], Q_cols[k])
    for j in range(1, length):
        d_ct = cc.EvalAdd(d_ct, cc.EvalMult(v_bc[j], Q_cols[k + j]))

    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), d_ct)
        Q_cols[k + i] = cc.EvalSub(Q_cols[k + i], update)

def solver(
    cc,
    keys,
    A: list,
    D_sqrt: int = 64,
    D_inv: int = 64,
) -> Tuple[list, list]:
    """Fully homomorphic Householder QR with column-packed R. Returns decrypted (Q, R). Chebyshev bounds from plaintext simulation with margin=10."""
    m, n = len(A), len(A[0])

    step_norms = simulate_norms(A)
    margin = 10.0

    R_cts = encrypt_matrix_cols(cc, keys, A)
    Q_cols = encrypt_identity_cols(cc, keys, m)

    steps = min(m, n)
    for k in range(steps):
        ns, vt = step_norms[k]
        householder_step_fhe_col(
            cc,
            keys,
            R_cts,
            Q_cols,
            k,
            m,
            n,
            norm_sq_lo=ns / margin,
            norm_sq_hi=ns * margin,
            vtv_lo=vt / margin,
            vtv_hi=vt * margin,
            D_sqrt=D_sqrt,
            D_inv=D_inv,
        )

    return Q_cols, R_cts
