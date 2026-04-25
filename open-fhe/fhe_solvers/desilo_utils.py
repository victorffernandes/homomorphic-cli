"""Shared FHE utilities for Householder QR solvers using desilofhe."""

from __future__ import annotations

import gc
import math
import numpy as np


def encrypt_row(engine, keys, row: list):
    """Encrypt a single matrix row, zero-padded to fill the slot count."""
    slots = engine.slot_count
    padded = np.zeros(slots)
    padded[:len(row)] = row
    return engine.encrypt(padded, keys['public_key'])


def encrypt_matrix_rows(engine, keys, A: list) -> list:
    """Encrypt matrix A as a list of row ciphertexts."""
    return [encrypt_row(engine, keys, row) for row in A]


def encrypt_matrix_cols(engine, keys, A: list) -> list:
    """Encrypt m x n matrix A as n column ciphertexts: R_cts[j] = [A[0,j], ..., A[m-1,j], 0, ...]."""
    m, n = len(A), len(A[0])
    slots = engine.slot_count
    R_cts = []
    for j in range(n):
        col = np.zeros(slots)
        col[:m] = [A[i][j] for i in range(m)]
        R_cts.append(engine.encrypt(col, keys['public_key']))
    return R_cts


def encrypt_identity_cols(engine, keys, m: int) -> list:
    """Encrypt the m x m identity matrix as a list of column ciphertexts."""
    slots = engine.slot_count
    cols = []
    for j in range(m):
        col = np.zeros(slots)
        col[j] = 1.0
        cols.append(engine.encrypt(col, keys['public_key']))
    return cols


def decrypt_vector(engine, keys, ct, length: int) -> list:
    """Decrypt a ciphertext and return the first `length` real values."""
    result = engine.decrypt(ct, keys['secret_key']).real
    return result[:length].tolist()


def decrypt_matrix_rows(engine, keys, R_cts: list, n: int) -> list:
    """Decrypt row-packed ciphertexts back to an m x n matrix."""
    return [decrypt_vector(engine, keys, ct, n) for ct in R_cts]


def decrypt_matrix_cols(engine, keys, Q_cols: list, m: int) -> list:
    """Decrypt column-packed ciphertexts back to an m x m matrix."""
    cols = [decrypt_vector(engine, keys, ct, m) for ct in Q_cols]
    return [[cols[j][i] for j in range(m)] for i in range(m)]


def _cheby_depth(degree: int) -> int:
    """Multiplicative depth for Chebyshev polynomial: 2 * ceil(log2(degree + 1))."""
    return 2 * math.ceil(math.log2(degree + 1))


def _cheby_coeffs(f, a: float, b: float, degree: int) -> list:
    """Compute Chebyshev coefficients for function f on domain [a, b].

    Returns coefficients c such that f(x) ≈ Σ c_k T_k(t) where t = (2x - a - b)/(b - a).
    Uses oversampling at 4*(degree+1) points via Chebyshev nodes.
    """
    n = 4 * (degree + 1)
    k = np.arange(n)
    theta = (2 * k + 1) * np.pi / (2 * n)
    t = np.cos(theta)
    x = (b - a) / 2 * t + (a + b) / 2
    y = np.array([f(xi) for xi in x])

    c = np.zeros(degree + 1)
    for j in range(degree + 1):
        c[j] = (2.0 / n) * np.sum(y * np.cos(j * theta))
    c[0] /= 2
    return c.tolist()


def he_sqrt(engine, keys, ct, a: float, b: float, degree: int = 16):
    """Chebyshev sqrt on [a, b]. Clamps to a to avoid sqrt(negative).
    Depth: ~2*ceil(log2(degree+1))."""
    f = lambda t: math.sqrt(max(t, a))
    coeffs = _cheby_coeffs(f, a, b, degree)

    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    t_ct = engine.multiply(ct, scale)
    t_ct = engine.add(t_ct, shift)

    return engine.evaluate_chebyshev_polynomial(t_ct, coeffs, keys['relin_key'])


def he_inv(engine, keys, ct, a: float, b: float, degree: int = 16):
    """Chebyshev 1/t on [a, b]. Domain must not cross zero."""
    f = lambda t: 1.0 / t
    coeffs = _cheby_coeffs(f, a, b, degree)

    scale = 2.0 / (b - a)
    shift = -(a + b) / (b - a)
    t_ct = engine.multiply(ct, scale)
    t_ct = engine.add(t_ct, shift)

    return engine.evaluate_chebyshev_polynomial(t_ct, coeffs, keys['relin_key'])


def he_mul_cc(engine, keys, a, b):
    """Ciphertext × ciphertext multiply with automatic relinearization."""
    return engine.multiply(a, b, keys['relin_key'])


def he_bootstrap(engine, keys, ct):
    """Bootstrap a ciphertext. Requires 'conjugation_key' and 'bootstrap_key' in keys."""
    return engine.bootstrap(
        ct,
        keys['relin_key'],
        keys['conjugation_key'],
        keys['bootstrap_key'],
    )


def maybe_bootstrap(engine, keys, ct, min_level: int):
    """Bootstrap ct only if its remaining level is below min_level."""
    if 'bootstrap_key' not in keys:
        return ct
    if ct.level < min_level:
        return he_bootstrap(engine, keys, ct)
    return ct


def safe_rotate(engine, keys, ct, k: int):
    """Rotate left by k, no-op when k == 0."""
    return engine.rotate(ct, keys['rotation_key'], k) if k != 0 else ct


def replicate_slot_0(engine, keys, ct, active_slots: int):
    """Broadcast slot 0 to the first active_slots via rotation-doubling tree.
    Depth cost: 0 (additions/rotations only)."""
    result = ct
    step = 1
    while step < active_slots:
        result = engine.add(result, safe_rotate(engine, keys, result, -step))
        step *= 2
    return result


def sum_slots(engine, keys, ct, n: int):
    """Tree-based sum of slots 0..n-1 into slot 0. Depth cost: 0."""
    result = ct
    step = 1
    while step < n:
        result = engine.add(result, safe_rotate(engine, keys, result, step))
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


def _simulate_qr_sign_plus(A: list) -> list:
    """Plaintext Householder QR with sign=+1 (matching FHE solver). Returns the R matrix."""
    m, n = len(A), len(A[0])
    r = [list(map(float, row)) for row in A]

    for k in range(min(m, n)):
        x = [r[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < 1e-15:
            continue
        v = [x[0] + norm_x] + x[1:]
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
    """Compute per-diagonal signed bounds (lo_i, hi_i) for R[i][i] from sign=+1 QR simulation."""
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
    rng = np.random.RandomState(seed)
    return [[float(rng.uniform(0.01, 10.0)) for _ in range(n)] for _ in range(m)]


def he_matmul_T_vec(engine, keys, Q_cols: list, rhs: list, m: int, n: int):
    """Compute c = Q^T @ rhs homomorphically. Q_cols encrypted, rhs plaintext.
    Returns a single ciphertext with c[j] in slot j (j = 0..n-1).
    Depth cost: +2 levels."""
    slots = engine.slot_count
    rhs_padded = np.zeros(slots)
    rhs_padded[:len(rhs)] = rhs
    rhs_pt = engine.encode(rhs_padded)

    e0_vec = np.zeros(slots)
    e0_vec[0] = 1.0
    e0_pt = engine.encode(e0_vec)

    c_ct = None
    for j in range(n):
        prod = engine.multiply(Q_cols[j], rhs_pt)
        dot = sum_slots(engine, keys, prod, m)
        dot = engine.multiply(dot, e0_pt)
        if j != 0:
            dot = safe_rotate(engine, keys, dot, -j)
        c_ct = dot if c_ct is None else engine.add(c_ct, dot)

    return c_ct


def he_back_substitute(engine, keys, R_cts: list, c_ct, n: int,
                       diag_bounds: list, D_inv: int = 64):
    """Solve upper-triangular Rx = c homomorphically with encrypted pivot inversion.
    R_cts: row-packed encrypted R (m ciphertexts).
    c_ct: encrypted c vector with c[j] in slot j.
    Returns ciphertext with x[j] in slot j."""
    if diag_bounds is None or len(diag_bounds) < n:
        raise ValueError("diag_bounds must provide one (lo, hi) pair per unknown")

    slots = engine.slot_count

    e0_vec = np.zeros(slots)
    e0_vec[0] = 1.0
    e0_pt = engine.encode(e0_vec)

    x_ct = engine.encrypt(np.zeros(slots), keys['public_key'])
    cheby_inv_depth = _cheby_depth(D_inv)
    c_ct = maybe_bootstrap(engine, keys, c_ct, 4)

    for i in range(n - 1, -1, -1):
        x_ct = maybe_bootstrap(engine, keys, x_ct, cheby_inv_depth + 4)
        R_cts[i] = maybe_bootstrap(engine, keys, R_cts[i], cheby_inv_depth + 4)

        if i < n - 1:
            products = he_mul_cc(engine, keys, R_cts[i], x_ct)
            inner = sum_slots(engine, keys, products, n)
            inner = engine.multiply(inner, e0_pt)
        else:
            inner = None

        c_i = engine.multiply(safe_rotate(engine, keys, c_ct, i), e0_pt)

        if inner is not None:
            numerator = engine.subtract(c_i, inner)
        else:
            numerator = c_i

        diag_i_ct = engine.multiply(safe_rotate(engine, keys, R_cts[i], i), e0_pt)

        lo_i, hi_i = diag_bounds[i]
        if abs(hi_i) < 1e-15 and abs(lo_i) < 1e-15:
            raise ValueError(
                f"Diagonal bound too small at position {i}: [{lo_i:.2e}, {hi_i:.2e}]"
            )

        inv_diag_ct = he_inv(engine, keys, diag_i_ct, lo_i, hi_i, D_inv)
        x_i = he_mul_cc(engine, keys, numerator, inv_diag_ct)

        if i != 0:
            x_i = safe_rotate(engine, keys, x_i, -i)
        x_ct = engine.add(x_ct, x_i)

        if i < n - 1:
            del products, inner
        del c_i, numerator, diag_i_ct, inv_diag_ct, x_i
        if i % 8 == 0:
            gc.collect()

    return x_ct


def he_primal_weights(engine, keys, x_ct, X_train, y_train):
    """Compute primal weight vector w = Σᵢ αᵢ·yᵢ·x_train_i inside FHE.
    x_ct: encrypted [b, α₀, ..., α_{n-1}] with b in slot 0, αᵢ in slot i+1.
    Returns encrypted d-dimensional weight vector in slots 0..d-1."""
    slots = engine.slot_count
    n_train = len(X_train)
    d = len(X_train[0]) if n_train > 0 else 0

    e0_vec = np.zeros(slots)
    e0_vec[0] = 1.0
    e0_pt = engine.encode(e0_vec)

    w_ct = None
    for i in range(n_train):
        alpha_i_ct = engine.multiply(safe_rotate(engine, keys, x_ct, i + 1), e0_pt)
        alpha_i_bc = replicate_slot_0(engine, keys, alpha_i_ct, d)

        sv = np.zeros(slots)
        for k in range(d):
            sv[k] = float(y_train[i] * X_train[i][k])
        sv_pt = engine.encode(sv)

        term = engine.multiply(alpha_i_bc, sv_pt)
        w_ct = term if w_ct is None else engine.add(w_ct, term)

    return w_ct
