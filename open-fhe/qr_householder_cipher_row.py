"""Packed-Row Householder QR over CKKS (OpenFHE-Python).

Sign assumed +1, zero decryptions inside the Householder loop.
R packed as row ciphertexts, Q packed as column ciphertexts.
Chebyshev domains derived per-step from plaintext simulation.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

from openfhe import *

from qr_householder_plain import householder_qr, matmul, fro_norm, sub


def setup_crypto_context(mult_depth: int, N: int = 8192) -> Tuple:
    """CKKS context with rotation keys for column extraction (1..150) and slot-0 replication (neg powers of 2)."""
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
    rot_indices = (
        list(range(1, 151))
        + [-(1 << i) for i in range(log_slots)]
    )
    cc.EvalRotateKeyGen(keys.secretKey, rot_indices)

    return cc, keys


def he_sqrt(cc, ct, a: float, b: float, degree: int = 16):
    """Chebyshev sqrt on [a, b]. Clamps to a to avoid sqrt(negative). Depth: ~2*ceil(log2(degree))."""
    return cc.EvalChebyshevFunction(lambda t: math.sqrt(max(t, a)), ct, a, b, degree)


def he_inv(cc, ct, a: float, b: float, degree: int = 16):
    """Chebyshev 1/t on [a, b]. Domain must be strictly positive. Depth: ~2*ceil(log2(degree))."""
    return cc.EvalChebyshevFunction(lambda t: 1.0 / t, ct, a, b, degree)


def safe_rotate(cc, ct, k: int):
    """EvalRotate left by k, no-op when k == 0 (OpenFHE has no automorphism key for index 0)."""
    return cc.EvalRotate(ct, k) if k != 0 else ct


def replicate_slot_0(cc, ct, slots: int):
    """Broadcast slot 0 to all slots via rotation-doubling tree. Depth cost: 0. Requires other slots to be zero."""
    result = ct
    step = 1
    while step < slots:
        result = cc.EvalAdd(result, cc.EvalRotate(result, -step))
        step *= 2
    return result


def householder_step_fhe(
    cc, keys,
    R_cts: list, Q_cols: list,
    k: int, m: int, n: int,
    norm_sq_lo: float, norm_sq_hi: float,
    vtv_lo: float, vtv_hi: float,
    D_sqrt: int = 16, D_inv: int = 16,
):
    """One Householder reflection at pivot column k (sign=+1, zero decryptions). Updates R_cts and Q_cols in-place."""
    slots  = cc.GetRingDimension() // 2
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

    v0_bc  = replicate_slot_0(cc, v0_ct, slots)
    tau_bc = replicate_slot_0(cc, two_over_vtv_ct, slots)

    v_bc: list = [v0_bc]
    for j in range(1, length):
        xj_slot0 = cc.EvalMult(
            safe_rotate(cc, x_masked[j], k),
            ptxt_e0,
        )
        v_bc.append(replicate_slot_0(cc, xj_slot0, slots))

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


def encrypt_row(cc, keys, row: list):
    """Encrypt a single matrix row, zero-padded to fill the slot count."""
    slots   = cc.GetRingDimension() // 2
    padded  = list(row) + [0.0] * (slots - len(row))
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(padded))


def decrypt_vector(cc, keys, ct, length: int) -> list:
    """Decrypt a ciphertext and return the first `length` real values."""
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(length)
    return [v.real for v in pt.GetCKKSPackedValue()]


def encrypt_matrix_rows(cc, keys, A: list) -> list:
    """Encrypt matrix A as a list of row ciphertexts."""
    return [encrypt_row(cc, keys, row) for row in A]


def encrypt_identity_cols(cc, keys, m: int) -> list:
    """Encrypt the m x m identity matrix as a list of column ciphertexts."""
    slots = cc.GetRingDimension() // 2
    cols  = []
    for j in range(m):
        col = [1.0 if i == j else 0.0 for i in range(m)]
        col += [0.0] * (slots - m)
        cols.append(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(col)))
    return cols


def decrypt_matrix_rows(cc, keys, R_cts: list, n: int) -> list:
    """Decrypt row-packed ciphertexts back to an m x n matrix."""
    return [decrypt_vector(cc, keys, ct, n) for ct in R_cts]


def decrypt_matrix_cols(cc, keys, Q_cols: list, m: int) -> list:
    """Decrypt column-packed ciphertexts back to an m x m matrix."""
    cols = [decrypt_vector(cc, keys, ct, m) for ct in Q_cols]
    return [[cols[j][i] for j in range(m)] for i in range(m)]


def _simulate_norms(A: list) -> list:
    """Plaintext Householder QR (sign=+1) returning [(norm_sq_k, vtv_k), ...] for per-step Chebyshev domains."""
    m, n = len(A), len(A[0])
    r = [list(map(float, row)) for row in A]
    info = []

    for k in range(min(m, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_sq = sum(xi * xi for xi in x)
        norm_x = math.sqrt(norm_sq)
        if norm_x < 1e-15:
            info.append((1.0, 1.0))
            continue
        v0 = x[0] + norm_x
        vtv = 2.0 * norm_x * v0
        if vtv < 1e-15:
            info.append((norm_sq, 1.0))
            continue
        info.append((norm_sq, vtv))

        v = [x[0] + norm_x] + x[1:]
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]
        for j in range(k, n):
            dot = sum(v[i] * r[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                r[k + i][j] -= 2.0 * v[i] * dot

    return info


def _cheby_depth(degree: int) -> int:
    """Multiplicative depth for EvalChebyshevFunction: 2 * ceil(log2(degree + 1)) (BSGS estimate)."""
    return 2 * math.ceil(math.log2(degree + 1))


def depth_for_size(m: int, n: int, D_sqrt: int = 64, D_inv: int = 64) -> int:
    """Minimum multiplicative depth for m x n FHE Householder QR with given Chebyshev degrees."""
    steps          = min(m, n)
    depth_per_step = 1 + _cheby_depth(D_sqrt) + 1 + _cheby_depth(D_inv) + 1 + 2 + 1 + 2
    return max(30, steps * depth_per_step + 10)


def householder_qr_cipher(
    cc, keys,
    A: list,
    D_sqrt: int = 64,
    D_inv: int = 64,
) -> Tuple[list, list]:
    """Fully homomorphic Householder QR. Returns decrypted (Q, R). Chebyshev bounds from plaintext simulation with margin=10."""
    m, n = len(A), len(A[0])

    step_norms = _simulate_norms(A)
    margin     = 10.0

    R_cts  = encrypt_matrix_rows(cc, keys, A)
    Q_cols = encrypt_identity_cols(cc, keys, m)

    steps = min(m, n)
    for k in range(steps):
        ns, vt = step_norms[k]
        householder_step_fhe(
            cc, keys, R_cts, Q_cols, k, m, n,
            norm_sq_lo = ns / margin,
            norm_sq_hi = ns * margin,
            vtv_lo     = vt / margin,
            vtv_hi     = vt * margin,
            D_sqrt     = D_sqrt,
            D_inv      = D_inv,
        )

    Q = decrypt_matrix_cols(cc, keys, Q_cols, m)
    R = decrypt_matrix_rows(cc, keys, R_cts, n)
    return Q, R


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


def random_matrix(m: int, n: int, seed: int = 42) -> list:
    """Random m x n matrix with entries in [0.01, 10]."""
    rng = random.Random(seed)
    return [[rng.uniform(0.01, 10.0) for _ in range(n)] for _ in range(m)]


def main():
    D_sqrt, D_inv = 64, 64

    A_main = random_matrix(150, 4, seed=42)
    m_main, n_main = 150, 4

    depth_main = depth_for_size(m_main, n_main, D_sqrt, D_inv)

    cc, keys = setup_crypto_context(depth_main, N=8192)

    t0 = time.perf_counter()
    Q_enc, R_enc = householder_qr_cipher(cc, keys, A_main, D_sqrt=D_sqrt, D_inv=D_inv)
    print(f"150x4 row-packed: {time.perf_counter() - t0:.1f}s")
    verify_fhe(A_main, Q_enc, R_enc, tol=1e-4)

    smoke = [
        ("2x2",  [[3.0, 1.0], [4.0, 1.5]], 1e-2),
        ("4x2",  random_matrix(4, 2, seed=7), 1e-2),
    ]

    for label, A_s, tol_s in smoke:
        m_s, n_s = len(A_s), len(A_s[0])
        d_s = depth_for_size(m_s, n_s, D_sqrt, D_inv)

        cc_s, keys_s = setup_crypto_context(d_s, N=8192)

        t0 = time.perf_counter()
        Q_s, R_s = householder_qr_cipher(cc_s, keys_s, A_s, D_sqrt=D_sqrt, D_inv=D_inv)
        print(f"{label} row-packed: {time.perf_counter() - t0:.1f}s")
        verify_fhe(A_s, Q_s, R_s, tol=tol_s)


if __name__ == "__main__":
    main()
