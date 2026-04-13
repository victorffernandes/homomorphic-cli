"""Packed-Row Householder QR over CKKS (OpenFHE-Python).

Sign assumed +1, zero decryptions inside the Householder loop.
R packed as row ciphertexts, Q packed as column ciphertexts.
Chebyshev domains derived per-step from plaintext simulation.
"""

from __future__ import annotations

import gc
import math
from typing import Tuple

from openfhe import *

from qr_householder_plain import matmul, fro_norm, sub
from .utils import (
    encrypt_identity_cols,
    he_sqrt,
    he_inv,
    safe_rotate,
    replicate_slot_0,
    encrypt_matrix_rows,
    simulate_norms,
    simulate_diag_bounds
)


def setup_crypto_context(mult_depth: int, N: int = None,
                         matrix_size: int = None, n_test: int = None) -> Tuple:
    """CKKS context with targeted rotation keys for the active matrix and test sizes.

    Ring dimension N is auto-scaled from mult_depth if not provided.
    """
    if N is None:
        # Total modulus bits ≈ first_mod + mult_depth * scaling_mod
        # Ring dimension must be a power of 2 and large enough to support the modulus chain.
        total_mod_bits = 60 + mult_depth * 50
        # Rule of thumb: N >= 2 * total_mod_bits for HEStd_NotSet
        N_min = 2 * total_mod_bits
        N = 1
        while N < N_min:
            N <<= 1
        N = max(N, 1024)  # floor at 1024

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

    matrix_size = 1 if matrix_size is None else max(1, int(matrix_size))
    n_test = matrix_size if n_test is None else max(1, int(n_test))
    neg_shift_limit = max(matrix_size, n_test)

    pos_shifts = list(range(1, matrix_size))
    neg_shifts = [-i for i in range(1, neg_shift_limit)]

    pos_pow2 = []
    step = 1
    while step < matrix_size:
        pos_pow2.append(step)
        step *= 2

    neg_pow2 = []
    step = 1
    while step < matrix_size:
        neg_pow2.append(-step)
        step *= 2

    rot_indices = pos_shifts + neg_shifts + pos_pow2 + neg_pow2
    rot_indices = sorted(set(rot_indices))
    cc.EvalRotateKeyGen(keys.secretKey, rot_indices)

    return cc, keys

def householder_step_fhe(
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
    """One Householder reflection at pivot column k (sign=+1, zero decryptions). Updates R_cts and Q_cols in-place."""
    slots = cc.GetRingDimension() // 2
    length = m - k

    e_k_vec = [0.0] * slots
    e_k_vec[k] = 1.0
    ptxt_ek = cc.MakeCKKSPackedPlaintext(e_k_vec)

    e_0_vec = [0.0] * slots
    e_0_vec[0] = 1.0
    ptxt_e0 = cc.MakeCKKSPackedPlaintext(e_0_vec)

    x_masked = [cc.EvalMult(R_cts[k + j], ptxt_ek) for j in range(length)]

    norm_sq_ct = cc.EvalMult(x_masked[0], x_masked[0])
    for j in range(1, length):
        sq_j = cc.EvalMult(x_masked[j], x_masked[j])
        norm_sq_ct = cc.EvalAdd(norm_sq_ct, sq_j)

    norm_sq_slot0 = cc.EvalMult(
        safe_rotate(cc, norm_sq_ct, k),
        ptxt_e0,
    )

    norm_ct = he_sqrt(cc, norm_sq_slot0, norm_sq_lo, norm_sq_hi, D_sqrt)
    norm_ct = cc.EvalMult(norm_ct, ptxt_e0)

    x0_slot0 = cc.EvalMult(
        safe_rotate(cc, cc.EvalMult(R_cts[k], ptxt_ek), k),
        ptxt_e0,
    )

    v0_ct = cc.EvalAdd(x0_slot0, norm_ct)

    vtv_ct = cc.EvalMult(cc.EvalMult(norm_ct, 2.0), v0_ct)

    two_over_vtv_ct = he_inv(cc, vtv_ct, vtv_lo, vtv_hi, D_inv)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, 2.0)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, ptxt_e0)

    active_slots = max(1, min(max(m, n), slots))

    v0_bc = replicate_slot_0(cc, v0_ct, active_slots)
    tau_bc = replicate_slot_0(cc, two_over_vtv_ct, active_slots)

    v_bc: list = [v0_bc]
    for j in range(1, length):
        xj_slot0 = cc.EvalMult(
            safe_rotate(cc, x_masked[j], k),
            ptxt_e0,
        )
        v_bc.append(replicate_slot_0(cc, xj_slot0, active_slots))

    del x_masked

    w_ct = cc.EvalMult(v_bc[0], R_cts[k])
    for j in range(1, length):
        w_ct = cc.EvalAdd(w_ct, cc.EvalMult(v_bc[j], R_cts[k + j]))

    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), w_ct)
        R_cts[k + i] = cc.EvalSub(R_cts[k + i], update)

    d_ct = cc.EvalMult(Q_cols[k], v_bc[0])
    for j in range(1, length):
        d_ct = cc.EvalAdd(d_ct, cc.EvalMult(Q_cols[k + j], v_bc[j]))

    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), d_ct)
        Q_cols[k + i] = cc.EvalSub(Q_cols[k + i], update)

    # Drop references to large intermediate ciphertext trees as soon as possible.
    del v_bc, w_ct, d_ct, tau_bc, v0_bc
    gc.collect()

def solver(
    cc,
    keys,
    A: list,
    D_sqrt: int = 64,
    D_inv: int = 64,
    diag_bounds: list = None,
) -> Tuple[list, list, list]:
    """Fully homomorphic Householder QR. Returns encrypted (Q_cols, R_cts, diag_bounds).

    Q_cols:     list of m column-packed ciphertexts.
    R_cts:      list of m row-packed ciphertexts.
    diag_bounds: list of tuples [(lo_i, hi_i), ...] for encrypted diagonal bounds in back-sub.
                 If None, computed from QR simulation.
    """
    m, n = len(A), len(A[0])

    R_cts = encrypt_matrix_rows(cc, keys, A)
    Q_cols = encrypt_identity_cols(cc, keys, m)

    step_norms = simulate_norms(A)
    diag_bounds = simulate_diag_bounds(A, diag_bounds=diag_bounds)
    margin = 2.0

    steps = min(m, n)
    for k in range(steps):
        ns, vt = step_norms[k]
        householder_step_fhe(
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

    return Q_cols, R_cts, diag_bounds

def transpose(a: list) -> list:
    m, n = len(a), len(a[0])
    return [[a[i][j] for i in range(m)] for j in range(n)]

def verify_fhe(A: list, Q: list, R: list, tol: float = 1e-4) -> bool:
    """Verify ||A - QR||_F / ||A||_F, ||Q^T Q - I||_F, and max |R[i,j]| for i > j against tol."""
    m, n = len(A), len(A[0])
    norm_A = fro_norm(A)

    rel_recon = fro_norm(sub(A, matmul(Q, R))) / norm_A if norm_A > 0 else 0.0
    print(f"  ||A - QR||_F / ||A||_F = {rel_recon:.2e}  (target < {tol:.0e})")

    I = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    ortho_err = fro_norm(sub(matmul(transpose(Q), Q), I))
    print(f"  ||Q^T Q - I||_F       = {ortho_err:.2e}  (target < {tol:.0e})")

    max_lower = max(
        (abs(R[i][j]) for i in range(m) for j in range(min(i, n))),
        default=0.0,
    )
    print(f"  max |R[i,j]| (i > j)  = {max_lower:.2e}  (target < {tol:.0e})")

    ok = rel_recon < tol and ortho_err < tol and max_lower < tol
    print(f"  PASS: {ok}\n")
    return ok

