"""Shared FHE utilities for Householder QR solvers."""

from __future__ import annotations

import math
import random


def encrypt_row(cc, keys, row: list):
    """Encrypt a single matrix row, zero-padded to fill the slot count."""
    slots = cc.GetRingDimension() // 2
    padded = list(row) + [0.0] * (slots - len(row))
    return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(padded))

def encrypt_matrix_rows(cc, keys, A: list) -> list:
    """Encrypt matrix A as a list of row ciphertexts."""
    return [encrypt_row(cc, keys, row) for row in A]


def encrypt_matrix_cols(cc, keys, A: list) -> list:
    """Encrypt m x n matrix A as n column ciphertexts: R_cts[j] = [A[0,j], ..., A[m-1,j], 0, ...]."""
    m, n = len(A), len(A[0])
    slots = cc.GetRingDimension() // 2
    R_cts = []
    for j in range(n):
        col = [A[i][j] for i in range(m)] + [0.0] * (slots - m)
        R_cts.append(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(col)))
    return R_cts



def encrypt_identity_cols(cc, keys, m: int) -> list:
    """Encrypt the m x m identity matrix as a list of column ciphertexts."""
    slots = cc.GetRingDimension() // 2
    cols  = []
    for j in range(m):
        col = [1.0 if i == j else 0.0 for i in range(m)]
        col += [0.0] * (slots - m)
        cols.append(cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(col)))
    return cols


def decrypt_vector(cc, keys, ct, length: int) -> list:
    """Decrypt a ciphertext and return the first `length` real values."""
    pt = cc.Decrypt(ct, keys.secretKey)
    pt.SetLength(length)
    return [v.real for v in pt.GetCKKSPackedValue()]


def decrypt_matrix_rows(cc, keys, R_cts: list, n: int) -> list:
    """Decrypt row-packed ciphertexts back to an m x n matrix."""
    return [decrypt_vector(cc, keys, ct, n) for ct in R_cts]


def decrypt_matrix_cols(cc, keys, Q_cols: list, m: int) -> list:
    """Decrypt column-packed ciphertexts back to an m x m matrix."""
    cols = [decrypt_vector(cc, keys, ct, m) for ct in Q_cols]
    return [[cols[j][i] for j in range(m)] for i in range(m)]


def _cheby_depth(degree: int) -> int:
    """Multiplicative depth for EvalChebyshevFunction: 2 * ceil(log2(degree + 1)) (BSGS estimate)."""
    return 2 * math.ceil(math.log2(degree + 1))


def depth_for_size(m: int, n: int, D_sqrt: int = 64, D_inv: int = 64,
                   D_inv_backsub: int = None, safety_factor: float = 1.15,
                   depth_override: int = None) -> int:
    """Estimated multiplicative depth for m x n FHE Householder QR + encrypted-pivot back-substitution.

    D_inv:          Chebyshev degree for 1/t in QR steps.
    D_inv_backsub:  Chebyshev degree for 1/t in back-sub pivot inversion (defaults to D_inv).
    safety_factor:  multiplicative buffer applied to calibrated core estimate.
    depth_override: if set, bypass estimation and return this depth (floored at 30).
    """
    if depth_override is not None:
        return max(30, int(depth_override))

    if D_inv_backsub is None:
        D_inv_backsub = D_inv

    steps = min(m, n)
    # Calibrated for FLEXIBLEAUTO: ct*pt masks often do not consume full levels.
    qr_depth = steps * (_cheby_depth(D_sqrt) + _cheby_depth(D_inv) + 1)
    backsub_depth = n * (_cheby_depth(D_inv_backsub) + 1)
    base_overhead = 8

    estimate = math.ceil((qr_depth + backsub_depth + base_overhead) * safety_factor)
    return max(30, estimate)


def he_sqrt(cc, ct, a: float, b: float, degree: int = 16):
    """Chebyshev sqrt on [a, b]. Clamps to a to avoid sqrt(negative). Depth: ~2*ceil(log2(degree))."""
    return cc.EvalChebyshevFunction(lambda t: math.sqrt(max(t, a)), ct, a, b, degree)


def he_inv(cc, ct, a: float, b: float, degree: int = 16):
    """Chebyshev 1/t on [a, b]. Domain must not cross zero (both positive or both negative)."""
    return cc.EvalChebyshevFunction(lambda t: 1.0 / t, ct, a, b, degree)


def safe_rotate(cc, ct, k: int):
    """EvalRotate left by k, no-op when k == 0 (OpenFHE has no automorphism key for index 0)."""
    return cc.EvalRotate(ct, k) if k != 0 else ct


def replicate_slot_0(cc, ct, active_slots: int):
    """Broadcast slot 0 to the first active_slots via rotation-doubling tree.

    Depth cost: 0 (additions/rotations only). Requires other slots in ct to be zero.
    """
    result = ct
    step = 1
    while step < active_slots:
        result = cc.EvalAdd(result, cc.EvalRotate(result, -step))
        step *= 2
    return result


def sum_slots(cc, ct, n: int):
    """Tree-based sum of slots 0..n-1 into slot 0. Depth cost: 0 (additions only). Caller must mask with e_0 after."""
    result = ct
    step = 1
    while step < n:
        result = cc.EvalAdd(result, cc.EvalRotate(result, step))
        step *= 2
    return result

def simulate_norms(A: list) -> list:
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

def simulate_diag_values(A: list) -> list:
    """Plaintext QR simulation returning the diagonal values R[i][i] for scalar inverse in back-sub."""
    from qr_householder_plain import householder_qr

    _, R = householder_qr(A)
    return [R[i][i] for i in range(len(R[0]))]


def _simulate_qr_sign_plus(A: list) -> list:
    """Plaintext Householder QR with sign=+1 (matching FHE solver). Returns the R matrix."""
    m, n = len(A), len(A[0])
    r = [list(map(float, row)) for row in A]

    for k in range(min(m, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < 1e-15:
            continue
        v = [x[0] + norm_x] + x[1:]  # sign=+1 always
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < 1e-15:
            continue
        v = [vi / norm_v for vi in v]
        for j in range(k, n):
            dot = sum(v[i] * r[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                r[k + i][j] -= 2.0 * v[i] * dot

    return r


def simulate_diag_bounds(A: list, diag_bounds: list = None, safety_margin: float = 0.1,
                         eps_floor: float = 1e-15) -> list:
    """Compute per-diagonal signed bounds (lo_i, hi_i) for R[i][i] from sign=+1 QR simulation.

    Uses the same sign=+1 Householder convention as the FHE solver so that
    diagonal signs match the encrypted values exactly.

    Bounds preserve the sign of the diagonal. For a negative diagonal d:
        lo = d * (1 + margin)  (more negative), hi = d * (1 - margin)  (less negative).
    For a positive diagonal d:
        lo = d * (1 - margin), hi = d * (1 + margin).
    In both cases lo < hi and the interval does not cross zero.

    Args:
        A: input matrix (m x n).
        diag_bounds: optional precomputed bounds. If provided, returned as-is.
        safety_margin: fraction to expand the interval around the diagonal (default 0.1 = ±10%).
        eps_floor: minimum absolute value to enforce to avoid near-zero domains.

    Returns:
        List of tuples [(lo_0, hi_0), ...] with lo_i < hi_i, preserving sign.
    """
    if diag_bounds is not None:
        return diag_bounds

    R = _simulate_qr_sign_plus(A)
    n = len(R[0])
    bounds = []

    for i in range(n):
        d = R[i][i]
        ad = abs(d)
        if ad < eps_floor:
            bounds.append((eps_floor, eps_floor * (1.0 + safety_margin)))
            continue

        if d > 0:
            lo = max(ad * (1.0 - safety_margin), eps_floor)
            hi = ad * (1.0 + safety_margin)
        else:
            lo = -ad * (1.0 + safety_margin)
            hi = -max(ad * (1.0 - safety_margin), eps_floor)
        bounds.append((lo, hi))

    return bounds


def random_matrix(m: int, n: int, seed: int = 42) -> list:
    """Random m x n matrix with entries in [0.01, 10]."""
    rng = random.Random(seed)
    return [[rng.uniform(0.01, 10.0) for _ in range(n)] for _ in range(m)]


def he_matmul_T_vec(cc, Q_cols: list, rhs: list, m: int, n: int):
    """Compute c = Q^T @ rhs homomorphically. Q_cols encrypted, rhs plaintext.

    Returns a single ciphertext with c[j] in slot j (j = 0..n-1).
    Depth cost: +2 levels beyond Q_cols depth.
    """
    slots = cc.GetRingDimension() // 2
    rhs_padded = list(rhs) + [0.0] * (slots - len(rhs))
    rhs_ptxt = cc.MakeCKKSPackedPlaintext(rhs_padded)

    e0_vec = [0.0] * slots
    e0_vec[0] = 1.0
    e0_ptxt = cc.MakeCKKSPackedPlaintext(e0_vec)

    c_ct = None
    for j in range(n):
        prod = cc.EvalMult(Q_cols[j], rhs_ptxt)
        dot = sum_slots(cc, prod, m)
        dot = cc.EvalMult(dot, e0_ptxt)
        if j != 0:
            dot = safe_rotate(cc, dot, -j)
        c_ct = dot if c_ct is None else cc.EvalAdd(c_ct, dot)

    return c_ct


def he_back_substitute(cc, keys, R_cts: list, c_ct, n: int,
                       diag_bounds: list, D_inv: int = 64):
    """Solve upper-triangular Rx = c homomorphically with encrypted pivot inversion.

    R_cts:      row-packed encrypted R (m ciphertexts).
    c_ct:       encrypted c vector with c[j] in slot j.
    n:          number of unknowns (columns of R).
    diag_bounds: list of tuples [(lo_i, hi_i), ...] for encrypted diagonal bounds.
    D_inv:      Chebyshev degree for encrypted reciprocal approximation.

    Returns ciphertext with x[j] in slot j. Depth cost: ~2*n + n*2*ceil(log2(D_inv+1)) levels.
    """
    if diag_bounds is None or len(diag_bounds) < n:
        raise ValueError("diag_bounds must provide one (lo, hi) pair per unknown for encrypted pivot inversion")

    slots = cc.GetRingDimension() // 2

    e0_vec = [0.0] * slots
    e0_vec[0] = 1.0
    e0_ptxt = cc.MakeCKKSPackedPlaintext(e0_vec)

    # Start with x = 0 (encrypted)
    x_ct = cc.Encrypt(keys.publicKey,
                      cc.MakeCKKSPackedPlaintext([0.0] * slots))

    for i in range(n - 1, -1, -1):
        # 1. Inner product: sum_{j>i} R[i][j] * x[j]
        if i < n - 1:
            # R_cts[i] has row i in slots 0..n-1; x_ct has solution so far
            products = cc.EvalMult(R_cts[i], x_ct)          # depth +1 (ct*ct)
            inner = sum_slots(cc, products, n)               # depth +0
            inner = cc.EvalMult(inner, e0_ptxt)              # depth +1 (ct*pt)
        else:
            inner = None

        # 2. Extract c[i] to slot 0
        c_i = cc.EvalMult(safe_rotate(cc, c_ct, i), e0_ptxt)

        # 3. Numerator: c[i] - inner
        if inner is not None:
            numerator = cc.EvalSub(c_i, inner)
        else:
            numerator = c_i

        # 4. Extract encrypted pivot R[i][i] and compute reciprocal
        # Extract R[i][i] to slot 0: rotate row i by i, then mask with e0
        diag_i_ct = cc.EvalMult(safe_rotate(cc, R_cts[i], i), e0_ptxt)

        # Guard: check that the diagonal bound is not too small
        lo_i, hi_i = diag_bounds[i]
        if abs(hi_i) < 1e-15 and abs(lo_i) < 1e-15:
            raise ValueError(
                f"Diagonal bound too small at position {i}: [{lo_i:.2e}, {hi_i:.2e}]. "
                "Cannot compute encrypted reciprocal. Check QR stability or increase bound margin."
            )

        # Compute encrypted reciprocal: 1/R[i][i] using Chebyshev approximation
        # Domain [lo_i, hi_i] may be negative (sign=+1 Householder produces negative diagonals)
        inv_diag_ct = he_inv(cc, diag_i_ct, lo_i, hi_i, D_inv)

        # Multiply numerator by encrypted reciprocal
        x_i = cc.EvalMult(numerator, inv_diag_ct)

        # 5. Place x[i] into slot i and accumulate
        if i != 0:
            x_i = safe_rotate(cc, x_i, -i)
        x_ct = cc.EvalAdd(x_ct, x_i)

    return x_ct
