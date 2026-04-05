"""Debug variant of Packed-Row Householder QR over CKKS (OpenFHE-Python).

Identical algorithm to qr_householder_cipher_row.py but with:
  - Mid-step decrypts after every phase to trace where NaN appears.
  - Per-phase logging with timing and slot-0 values.
  - No matrix dumps in output.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

from openfhe import *

from qr_householder_plain import householder_qr, matmul, fro_norm, sub


# ---------------------------------------------------------------------------
# 1. Crypto context  (unchanged)
# ---------------------------------------------------------------------------

def setup_crypto_context(mult_depth: int, N: int = 8192) -> Tuple:
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


# ---------------------------------------------------------------------------
# 2. Chebyshev primitives  (unchanged)
# ---------------------------------------------------------------------------

def he_sqrt(cc, ct, a: float, b: float, degree: int = 16):
    return cc.EvalChebyshevFunction(lambda t: math.sqrt(max(t, a)), ct, a, b, degree)


def he_inv(cc, ct, a: float, b: float, degree: int = 16):
    return cc.EvalChebyshevFunction(lambda t: 1.0 / t, ct, a, b, degree)


# ---------------------------------------------------------------------------
# 3. Slot replication  (unchanged)
# ---------------------------------------------------------------------------

def safe_rotate(cc, ct, k: int):
    return cc.EvalRotate(ct, k) if k != 0 else ct


def replicate_slot_0(cc, ct, slots: int):
    result = ct
    step = 1
    while step < slots:
        result = cc.EvalAdd(result, cc.EvalRotate(result, -step))
        step *= 2
    return result


# ---------------------------------------------------------------------------
# 4. Debug helpers
# ---------------------------------------------------------------------------

def _decrypt_slot0(cc, keys, ct) -> float:
    """Decrypt and return only slot 0."""
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(1)
    return pt.GetCKKSPackedValue()[0].real


def _decrypt_slots(cc, keys, ct, n: int) -> list:
    """Decrypt and return the first n slots."""
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(n)
    return [v.real for v in pt.GetCKKSPackedValue()]


def _fmt(v) -> str:
    """Format a value, flagging NaN/Inf."""
    if math.isnan(v):
        return "NaN!"
    if math.isinf(v):
        return "Inf!"
    return f"{v:.6e}"


def _check_nan(label: str, values) -> bool:
    """Return True if any value is NaN or Inf, and print a warning."""
    bad = any(math.isnan(v) or math.isinf(v) for v in values)
    if bad:
        print(f"    *** {label}: contains NaN/Inf! ***")
    return bad


def _lvl(ct) -> str:
    """Return ciphertext level as string, or '?' if not available."""
    try:
        return str(ct.GetLevel())
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# 5. Core Householder step — with mid-step decrypts
# ---------------------------------------------------------------------------

def householder_step_fhe_debug(
    cc, keys,
    R_cts: list, Q_cols: list,
    k: int, m: int, n: int,
    norm_sq_lo: float, norm_sq_hi: float,
    vtv_lo: float, vtv_hi: float,
    D_sqrt: int = 16, D_inv: int = 16,
    # plaintext references for comparison
    expected_norm_sq: float = None,
    expected_vtv: float = None,
):
    slots  = cc.GetRingDimension() // 2
    length = m - k

    print(f"\n    [step k={k}] length={length}, slots={slots}")
    print(f"    [step k={k}] Chebyshev domains:")
    print(f"      norm_sq: [{norm_sq_lo:.4e}, {norm_sq_hi:.4e}]")
    print(f"      vtv:     [{vtv_lo:.4e}, {vtv_hi:.4e}]")
    if expected_norm_sq is not None:
        print(f"      expected norm_sq={expected_norm_sq:.6e}  vtv={expected_vtv:.6e}")
    print(f"    [step k={k}] input levels: R_cts[{k}]={_lvl(R_cts[k])}, Q_cols[{k}]={_lvl(Q_cols[k])}")

    # ------------------------------------------------------------------
    # Plaintext selector masks
    # ------------------------------------------------------------------
    e_k_vec = [0.0] * slots
    e_k_vec[k] = 1.0
    ptxt_ek = cc.MakeCKKSPackedPlaintext(e_k_vec)

    e_0_vec = [0.0] * slots
    e_0_vec[0] = 1.0
    ptxt_e0 = cc.MakeCKKSPackedPlaintext(e_0_vec)

    # ------------------------------------------------------------------
    # Phase 1: norm computation
    # ------------------------------------------------------------------
    t_phase = time.perf_counter()

    x_masked = [cc.EvalMult(R_cts[k + j], ptxt_ek) for j in range(length)]

    # Debug: spot-check x_masked[0] and x_masked[-1]
    xm0 = _decrypt_slots(cc, keys, x_masked[0], k + 1)[k]
    xmL = _decrypt_slots(cc, keys, x_masked[-1], k + 1)[k]
    print(f"    [Phase 1] x_masked[0] slot[{k}]={_fmt(xm0)}, "
          f"x_masked[{length-1}] slot[{k}]={_fmt(xmL)}")
    _check_nan("x_masked endpoints", [xm0, xmL])

    norm_sq_ct = cc.EvalMult(x_masked[0], x_masked[0])
    for j in range(1, length):
        sq_j = cc.EvalMult(x_masked[j], x_masked[j])
        norm_sq_ct = cc.EvalAdd(norm_sq_ct, sq_j)

    # Debug: decrypt norm_sq in slot k
    norm_sq_vals = _decrypt_slots(cc, keys, norm_sq_ct, k + 1)
    dec_norm_sq = norm_sq_vals[k]
    print(f"    [Phase 1] norm_sq (slot {k}): {_fmt(dec_norm_sq)}")
    _check_nan("norm_sq_ct slot k", [dec_norm_sq])

    norm_sq_slot0 = cc.EvalMult(
        safe_rotate(cc, norm_sq_ct, k),
        ptxt_e0,
    )

    # Debug: check slot 0 after rotation
    dec_ns0 = _decrypt_slot0(cc, keys, norm_sq_slot0)
    print(f"    [Phase 1] norm_sq after rotate-to-slot0: {_fmt(dec_ns0)}")
    in_domain = norm_sq_lo <= dec_ns0 <= norm_sq_hi
    print(f"    [Phase 1] in Chebyshev domain [{norm_sq_lo:.4e}, {norm_sq_hi:.4e}]? {in_domain}")
    if not in_domain:
        print(f"    *** WARNING: norm_sq value {dec_ns0:.6e} OUTSIDE Chebyshev domain! ***")
    _check_nan("norm_sq_slot0", [dec_ns0])

    norm_ct = he_sqrt(cc, norm_sq_slot0, norm_sq_lo, norm_sq_hi, D_sqrt)
    norm_ct = cc.EvalMult(norm_ct, ptxt_e0)

    dec_norm = _decrypt_slot0(cc, keys, norm_ct)
    expected_norm = math.sqrt(dec_ns0) if dec_ns0 > 0 else float('nan')
    print(f"    [Phase 1] ||x|| (Chebyshev sqrt): {_fmt(dec_norm)}  "
          f"(expected sqrt({_fmt(dec_ns0)})={_fmt(expected_norm)})")
    _check_nan("norm_ct", [dec_norm])

    print(f"    [Phase 1] levels: x_masked[0]={_lvl(x_masked[0])}, "
          f"norm_sq_ct={_lvl(norm_sq_ct)}, norm_sq_slot0={_lvl(norm_sq_slot0)}, "
          f"norm_ct={_lvl(norm_ct)}")
    print(f"    [Phase 1] elapsed: {time.perf_counter() - t_phase:.2f}s")

    # ------------------------------------------------------------------
    # Phase 2: v0, vtv, tau
    # ------------------------------------------------------------------
    t_phase = time.perf_counter()

    x0_slot0 = cc.EvalMult(
        safe_rotate(cc, cc.EvalMult(R_cts[k], ptxt_ek), k),
        ptxt_e0,
    )

    dec_x0 = _decrypt_slot0(cc, keys, x0_slot0)
    print(f"    [Phase 2] x[0] = R[{k},{k}] (slot 0): {_fmt(dec_x0)}")
    _check_nan("x0_slot0", [dec_x0])

    v0_ct = cc.EvalAdd(x0_slot0, norm_ct)
    dec_v0 = _decrypt_slot0(cc, keys, v0_ct)
    print(f"    [Phase 2] v0 = x[0] + ||x|| = {_fmt(dec_x0)} + {_fmt(dec_norm)} = {_fmt(dec_v0)}")
    _check_nan("v0_ct", [dec_v0])

    vtv_ct = cc.EvalMult(cc.EvalMult(norm_ct, 2.0), v0_ct)
    dec_vtv = _decrypt_slot0(cc, keys, vtv_ct)
    expected_vtv_val = 2.0 * dec_norm * dec_v0
    print(f"    [Phase 2] vtv = 2*||x||*v0 = {_fmt(dec_vtv)}  (expected {_fmt(expected_vtv_val)})")
    in_domain_vtv = vtv_lo <= dec_vtv <= vtv_hi
    print(f"    [Phase 2] in Chebyshev domain [{vtv_lo:.4e}, {vtv_hi:.4e}]? {in_domain_vtv}")
    if not in_domain_vtv:
        print(f"    *** WARNING: vtv value {dec_vtv:.6e} OUTSIDE Chebyshev domain! ***")
    _check_nan("vtv_ct", [dec_vtv])

    two_over_vtv_ct = he_inv(cc, vtv_ct, vtv_lo, vtv_hi, D_inv)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, 2.0)       # 1/(v^Tv) → 2/(v^Tv)
    two_over_vtv_ct = cc.EvalMult(two_over_vtv_ct, ptxt_e0)

    dec_tau = _decrypt_slot0(cc, keys, two_over_vtv_ct)
    expected_tau = 2.0 / dec_vtv if dec_vtv != 0 else float('nan')
    print(f"    [Phase 2] tau = 2/vtv (Chebyshev inv * 2): {_fmt(dec_tau)}  "
          f"(expected {_fmt(expected_tau)})")
    _check_nan("two_over_vtv_ct", [dec_tau])

    print(f"    [Phase 2] levels: x0_slot0={_lvl(x0_slot0)}, v0_ct={_lvl(v0_ct)}, "
          f"vtv_ct={_lvl(vtv_ct)}, two_over_vtv_ct={_lvl(two_over_vtv_ct)}")
    print(f"    [Phase 2] elapsed: {time.perf_counter() - t_phase:.2f}s")

    # ------------------------------------------------------------------
    # Phase 3: Broadcast
    # ------------------------------------------------------------------
    t_phase = time.perf_counter()

    v0_bc  = replicate_slot_0(cc, v0_ct, slots)
    tau_bc = replicate_slot_0(cc, two_over_vtv_ct, slots)

    v_bc: list = [v0_bc]
    for j in range(1, length):
        xj_slot0 = cc.EvalMult(
            safe_rotate(cc, x_masked[j], k),
            ptxt_e0,
        )
        v_bc.append(replicate_slot_0(cc, xj_slot0, slots))

    # Debug: verify broadcast consistency (slot 0 vs slot 1)
    v0_s0 = _decrypt_slot0(cc, keys, v_bc[0])
    tau_s0 = _decrypt_slot0(cc, keys, tau_bc)
    print(f"    [Phase 3] v0_bc slot0={_fmt(v0_s0)}, tau_bc slot0={_fmt(tau_s0)}")
    _check_nan("v_bc[0]", [v0_s0])
    _check_nan("tau_bc", [tau_s0])

    print(f"    [Phase 3] levels: v0_bc={_lvl(v0_bc)}, tau_bc={_lvl(tau_bc)}, "
          f"v_bc[0]={_lvl(v_bc[0])}"
          + (f", v_bc[1]={_lvl(v_bc[1])}" if length > 1 else ""))
    print(f"    [Phase 3] elapsed: {time.perf_counter() - t_phase:.2f}s")

    # ------------------------------------------------------------------
    # Phase 4: Update R rows
    # ------------------------------------------------------------------
    t_phase = time.perf_counter()

    w_ct = cc.EvalMult(v_bc[0], R_cts[k])
    for j in range(1, length):
        w_ct = cc.EvalAdd(w_ct, cc.EvalMult(v_bc[j], R_cts[k + j]))

    # Debug: check w_ct summary
    w_vals = _decrypt_slots(cc, keys, w_ct, n)
    w_has_nan = _check_nan("w_ct", w_vals)
    print(f"    [Phase 4] w_ct (v^T @ R): max|val|={max(abs(v) for v in w_vals):.4e}, nan={w_has_nan}")

    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), w_ct)
        R_cts[k + i] = cc.EvalSub(R_cts[k + i], update)

    # Debug: check updated R rows — summary only
    r_nan_count = 0
    r_max_abs = 0.0
    for i in range(min(5, length)):
        r_vals = _decrypt_slots(cc, keys, R_cts[k + i], n)
        r_max_abs = max(r_max_abs, max(abs(v) for v in r_vals))
        if any(math.isnan(v) or math.isinf(v) for v in r_vals):
            r_nan_count += 1
    print(f"    [Phase 4] R after update: max|val|={r_max_abs:.4e}, "
          f"nan_rows(first 5)={r_nan_count}")
    print(f"    [Phase 4] levels: w_ct={_lvl(w_ct)}, "
          f"R_cts[{k}]={_lvl(R_cts[k])}, R_cts[{k + length - 1}]={_lvl(R_cts[k + length - 1])}")

    print(f"    [Phase 4] elapsed: {time.perf_counter() - t_phase:.2f}s")

    # ------------------------------------------------------------------
    # Phase 5: Update Q columns
    # ------------------------------------------------------------------
    t_phase = time.perf_counter()

    d_ct = cc.EvalMult(Q_cols[k], v_bc[0])
    for j in range(1, length):
        d_ct = cc.EvalAdd(d_ct, cc.EvalMult(Q_cols[k + j], v_bc[j]))

    # Debug: check d_ct summary
    d_vals = _decrypt_slots(cc, keys, d_ct, min(5, m))
    d_has_nan = _check_nan("d_ct", d_vals)
    print(f"    [Phase 5] d_ct (Q @ v): max|val|={max(abs(v) for v in d_vals):.4e}, nan={d_has_nan}")

    for i in range(length):
        update = cc.EvalMult(cc.EvalMult(tau_bc, v_bc[i]), d_ct)
        Q_cols[k + i] = cc.EvalSub(Q_cols[k + i], update)

    # Debug: check updated Q columns — summary only
    q_nan_count = 0
    q_max_abs = 0.0
    for i in range(min(5, length)):
        q_vals = _decrypt_slots(cc, keys, Q_cols[k + i], min(5, m))
        q_max_abs = max(q_max_abs, max(abs(v) for v in q_vals))
        if any(math.isnan(v) or math.isinf(v) for v in q_vals):
            q_nan_count += 1
    print(f"    [Phase 5] Q after update: max|val|={q_max_abs:.4e}, "
          f"nan_cols(first 5)={q_nan_count}")
    print(f"    [Phase 5] levels: d_ct={_lvl(d_ct)}, "
          f"Q_cols[{k}]={_lvl(Q_cols[k])}, Q_cols[{k + length - 1}]={_lvl(Q_cols[k + length - 1])}")

    print(f"    [Phase 5] elapsed: {time.perf_counter() - t_phase:.2f}s")


# ---------------------------------------------------------------------------
# 6. Encrypt / decrypt helpers  (unchanged)
# ---------------------------------------------------------------------------

def encrypt_row(cc, keys, row: list):
    slots   = cc.GetRingDimension() // 2
    padded  = list(row) + [0.0] * (slots - len(row))
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(padded))


def decrypt_vector(cc, keys, ct, length: int) -> list:
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(length)
    return [v.real for v in pt.GetCKKSPackedValue()]


def encrypt_matrix_rows(cc, keys, A: list) -> list:
    return [encrypt_row(cc, keys, row) for row in A]


def encrypt_identity_cols(cc, keys, m: int) -> list:
    slots = cc.GetRingDimension() // 2
    cols  = []
    for j in range(m):
        col = [1.0 if i == j else 0.0 for i in range(m)]
        col += [0.0] * (slots - m)
        cols.append(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(col)))
    return cols


def decrypt_matrix_rows(cc, keys, R_cts: list, n: int) -> list:
    return [decrypt_vector(cc, keys, ct, n) for ct in R_cts]


def decrypt_matrix_cols(cc, keys, Q_cols: list, m: int) -> list:
    cols = [decrypt_vector(cc, keys, ct, m) for ct in Q_cols]
    return [[cols[j][i] for j in range(m)] for i in range(m)]


# ---------------------------------------------------------------------------
# 7. Plaintext simulation  (unchanged)
# ---------------------------------------------------------------------------

def _simulate_norms(A: list) -> list:
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


# ---------------------------------------------------------------------------
# 8. Depth helper  (unchanged)
# ---------------------------------------------------------------------------

def _cheby_depth(degree: int) -> int:
    return 2 * math.ceil(math.log2(degree + 1))


def depth_for_size(m: int, n: int, D_sqrt: int = 64, D_inv: int = 64) -> int:
    steps          = min(m, n)
    depth_per_step = 1 + _cheby_depth(D_sqrt) + 1 + _cheby_depth(D_inv) + 1 + 2 + 1 + 2
    return max(30, steps * depth_per_step + 10)


# ---------------------------------------------------------------------------
# 9. Top-level QR driver — debug version
# ---------------------------------------------------------------------------

def householder_qr_cipher_debug(
    cc, keys,
    A: list,
    D_sqrt: int = 64,
    D_inv: int = 64,
) -> Tuple[list, list]:
    """Debug variant: decrypts intermediates at every phase to trace NaN."""
    m, n = len(A), len(A[0])

    step_norms = _simulate_norms(A)
    margin     = 10.0

    # Print the max level budget (fresh ciphertext level = 0, max usable = mult_depth)
    fresh_lvl = _lvl(encrypt_row(cc, keys, [0.0]))
    print(f"\n  Depth budget: fresh ct level={fresh_lvl}")

    print(f"\n  Plaintext simulation results:")
    for i, (ns, vt) in enumerate(step_norms):
        print(f"    step {i}: norm_sq={ns:.6e}  vtv={vt:.6e}")

    R_cts  = encrypt_matrix_rows(cc, keys, A)
    Q_cols = encrypt_identity_cols(cc, keys, m)

    # Sanity check: decrypt initial R rows
    print(f"\n  Initial R sanity check (first 3 rows, {n} cols):")
    for i in range(min(3, m)):
        vals = _decrypt_slots(cc, keys, R_cts[i], n)
        orig = A[i][:n]
        max_err = max(abs(vals[j] - orig[j]) for j in range(n))
        print(f"    R[{i}]: encrypt/decrypt max_err={max_err:.2e}")

    steps = min(m, n)
    for k in range(steps):
        ns, vt = step_norms[k]
        print(f"\n  {'='*50}")
        print(f"  STEP k={k}/{steps-1}")
        print(f"  {'='*50}")
        t0 = time.perf_counter()
        householder_step_fhe_debug(
            cc, keys, R_cts, Q_cols, k, m, n,
            norm_sq_lo = ns / margin,
            norm_sq_hi = ns * margin,
            vtv_lo     = vt / margin,
            vtv_hi     = vt * margin,
            D_sqrt     = D_sqrt,
            D_inv      = D_inv,
            expected_norm_sq = ns,
            expected_vtv     = vt,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  step k={k} total: {elapsed:.1f}s")

        # Post-step sanity: check diagonal of R
        diag_vals = []
        for j in range(n):
            r_row = _decrypt_slots(cc, keys, R_cts[j], n)
            diag_vals.append(r_row[j] if j < n else float('nan'))
        print(f"  R diagonal after step {k}: {', '.join(_fmt(v) for v in diag_vals)}")
        if any(math.isnan(v) or math.isinf(v) for v in diag_vals):
            print(f"  *** NaN/Inf detected in R diagonal — stopping early ***")
            break

    Q = decrypt_matrix_cols(cc, keys, Q_cols, m)
    R = decrypt_matrix_rows(cc, keys, R_cts, n)
    return Q, R


# ---------------------------------------------------------------------------
# 10. Verification  (unchanged, no matrix dump)
# ---------------------------------------------------------------------------

def transpose(a: list) -> list:
    m, n = len(a), len(a[0])
    return [[a[i][j] for i in range(m)] for j in range(n)]


def verify_fhe(A: list, Q: list, R: list, tol: float = 1e-4) -> bool:
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


# ---------------------------------------------------------------------------
# 11. Main — debug run (small matrix first to iterate quickly)
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("Householder QR — DEBUG VARIANT (mid-step decrypts)")
    print("=" * 62)

    D_sqrt, D_inv = 64, 64

    # Start with a small matrix for fast iteration
    print("\n>>> DEBUG RUN: 4x2")
    A = random_matrix(4, 2, seed=7)
    m, n = len(A), len(A[0])

    depth = depth_for_size(m, n, D_sqrt, D_inv)
    # print(f"  Matrix: {m}x{n}  |  Steps: {min(m, n)}  |  mult_depth: {depth}")

    # Plaintext reference
    Q_ref, R_ref = householder_qr(A)
    print(f"  Plaintext R diagonal: {', '.join(_fmt(R_ref[i][i]) for i in range(min(m, n)))}")

    # FHE debug run
    t0 = time.perf_counter()
    cc, keys = setup_crypto_context(depth, N=8192)
    print(f"  Context setup: {time.perf_counter() - t0:.1f}s  "
          f"(slots={cc.GetRingDimension() // 2})")

    t0 = time.perf_counter()
    Q_enc, R_enc = householder_qr_cipher_debug(cc, keys, A, D_sqrt=D_sqrt, D_inv=D_inv)
    print(f"\n  FHE QR total: {time.perf_counter() - t0:.1f}s")

    print("\n  --- Verification ---")
    verify_fhe(A, Q_enc, R_enc, tol=1e-2)

    # Second run: the primary 150x4 target
    print("\n>>> DEBUG RUN: 150x4")
    A_main = random_matrix(150, 4, seed=42)
    m_main, n_main = 150, 4

    depth_main = depth_for_size(m_main, n_main, D_sqrt, D_inv)
    print(f"  Matrix: {m_main}x{n_main}  |  Steps: {min(m_main, n_main)}  |  mult_depth: {depth_main}")

    Q_ref, R_ref = householder_qr(A_main)
    print(f"  Plaintext R diagonal: "
          f"{', '.join(_fmt(R_ref[i][i]) for i in range(min(m_main, n_main)))}")

    t0 = time.perf_counter()
    cc, keys = setup_crypto_context(depth_main, N=8192)
    print(f"  Context setup: {time.perf_counter() - t0:.1f}s  "
          f"(slots={cc.GetRingDimension() // 2})")

    t0 = time.perf_counter()
    Q_enc, R_enc = householder_qr_cipher_debug(cc, keys, A_main, D_sqrt=D_sqrt, D_inv=D_inv)
    print(f"\n  FHE QR total: {time.perf_counter() - t0:.1f}s")

    print("\n  --- Verification ---")
    verify_fhe(A_main, Q_enc, R_enc, tol=1e-2)


if __name__ == "__main__":
    main()
