"""Packed-Row Householder QR over CKKS (desilofhe).

Sign assumed +1, zero decryptions inside the Householder loop.
R packed as row ciphertexts, Q packed as column ciphertexts.
Chebyshev domains derived per-step from plaintext simulation.
"""

from __future__ import annotations

import gc
import json
import math
import os
from typing import Tuple

import numpy as np
from desilofhe import Engine

from qr_householder_plain import matmul, fro_norm, sub
from .desilo_utils import (
    encrypt_identity_cols,
    he_sqrt,
    he_inv,
    safe_rotate,
    replicate_slot_0,
    encrypt_matrix_rows,
    simulate_norms,
    simulate_diag_bounds,
    he_matmul_T_vec,
    he_back_substitute,
    he_primal_weights,
    maybe_bootstrap,
    he_mul_cc,
)


def setup_crypto_context(mult_depth: int, N: int = None,
                         matrix_size: int = None, n_test: int = None,
                         feature_dim: int = None) -> Tuple:
    """CKKS context with desilofhe bootstrap mode.

    Parameters N, matrix_size, n_test, and feature_dim are accepted only for API
    compatibility. mult_depth is also accepted for ABI compatibility with callers
    but is ignored because bootstrap mode fixes max_level=26 in the library.
    """
    _ = (mult_depth, N, matrix_size, n_test, feature_dim)
    engine = Engine(use_bootstrap=True)
    print(f"  [desilo] bootstrap mode (max_level={engine.max_level}, slots={engine.slot_count})")

    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    relin_key = engine.create_relinearization_key(secret_key)
    rotation_key = engine.create_rotation_key(secret_key)
    conjugation_key = engine.create_conjugation_key(secret_key)
    bootstrap_key = engine.create_bootstrap_key(secret_key)

    keys = {
        'secret_key': secret_key,
        'public_key': public_key,
        'relin_key': relin_key,
        'rotation_key': rotation_key,
        'conjugation_key': conjugation_key,
        'bootstrap_key': bootstrap_key,
    }

    return engine, keys


def householder_step_fhe(
    engine,
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
    """One Householder reflection at pivot column k (sign=+1, zero decryptions).
    Updates R_cts and Q_cols in-place."""
    slots = engine.slot_count
    length = m - k

    e_k_vec = np.zeros(slots)
    e_k_vec[k] = 1.0
    pt_ek = engine.encode(e_k_vec)

    e_0_vec = np.zeros(slots)
    e_0_vec[0] = 1.0
    pt_e0 = engine.encode(e_0_vec)

    x_masked = [engine.multiply(R_cts[k + j], pt_ek) for j in range(length)]

    norm_sq_ct = he_mul_cc(engine, keys, x_masked[0], x_masked[0])
    for j in range(1, length):
        sq_j = he_mul_cc(engine, keys, x_masked[j], x_masked[j])
        norm_sq_ct = engine.add(norm_sq_ct, sq_j)

    norm_sq_slot0 = engine.multiply(
        safe_rotate(engine, keys, norm_sq_ct, k),
        pt_e0,
    )

    norm_ct = he_sqrt(engine, keys, norm_sq_slot0, norm_sq_lo, norm_sq_hi, D_sqrt)
    norm_ct = engine.multiply(norm_ct, pt_e0)
    del norm_sq_ct, norm_sq_slot0
    gc.collect()

    x0_slot0 = engine.multiply(
        safe_rotate(engine, keys, engine.multiply(R_cts[k], pt_ek), k),
        pt_e0,
    )

    v0_ct = engine.add(x0_slot0, norm_ct)
    cheby_inv_depth = 2 * math.ceil(math.log2(D_inv + 1))
    norm_ct = maybe_bootstrap(engine, keys, norm_ct, cheby_inv_depth + 3)
    v0_ct = maybe_bootstrap(engine, keys, v0_ct, cheby_inv_depth + 3)
    del x0_slot0

    vtv_ct = he_mul_cc(engine, keys, engine.multiply(norm_ct, 2.0), v0_ct)

    two_over_vtv_ct = he_inv(engine, keys, vtv_ct, vtv_lo, vtv_hi, D_inv)
    two_over_vtv_ct = engine.multiply(two_over_vtv_ct, 2.0)
    two_over_vtv_ct = engine.multiply(two_over_vtv_ct, pt_e0)

    active_slots = max(1, min(max(m, n), slots))

    v0_bc = replicate_slot_0(engine, keys, v0_ct, active_slots)
    tau_bc = replicate_slot_0(engine, keys, two_over_vtv_ct, active_slots)

    v_bc: list = [v0_bc]
    for j in range(1, length):
        xj_slot0 = engine.multiply(
            safe_rotate(engine, keys, x_masked[j], k),
            pt_e0,
        )
        v_bc.append(replicate_slot_0(engine, keys, xj_slot0, active_slots))

    del x_masked

    w_ct = he_mul_cc(engine, keys, v_bc[0], R_cts[k])
    for j in range(1, length):
        w_ct = engine.add(w_ct, he_mul_cc(engine, keys, v_bc[j], R_cts[k + j]))

    for i in range(length):
        tv = he_mul_cc(engine, keys, tau_bc, v_bc[i])
        update = he_mul_cc(engine, keys, tv, w_ct)
        R_cts[k + i] = engine.subtract(R_cts[k + i], update)

    del w_ct

    d_ct = he_mul_cc(engine, keys, Q_cols[k], v_bc[0])
    for j in range(1, length):
        d_ct = engine.add(d_ct, he_mul_cc(engine, keys, Q_cols[k + j], v_bc[j]))

    for i in range(length):
        tv = he_mul_cc(engine, keys, tau_bc, v_bc[i])
        update = he_mul_cc(engine, keys, tv, d_ct)
        Q_cols[k + i] = engine.subtract(Q_cols[k + i], update)

    del v_bc, d_ct, tau_bc, v0_bc, norm_ct
    gc.collect()


def _qr(
    engine,
    keys,
    A: list,
    D_sqrt: int = 64,
    D_inv: int = 64,
    diag_bounds: list = None,
) -> Tuple[list, list, list]:
    """Fully homomorphic Householder QR. Returns (Q_cols, R_cts, diag_bounds)."""
    m, n = len(A), len(A[0])

    R_cts = encrypt_matrix_rows(engine, keys, A)
    Q_cols = encrypt_identity_cols(engine, keys, m)

    step_norms = simulate_norms(A)
    diag_bounds = simulate_diag_bounds(A, diag_bounds=diag_bounds)
    margin = 2.0

    steps = min(m, n)

    for k in range(steps):
        for j in range(k, m):
            R_cts[j] = maybe_bootstrap(engine, keys, R_cts[j], 5)
            Q_cols[j] = maybe_bootstrap(engine, keys, Q_cols[j], 5)

        ns, vt = step_norms[k]
        householder_step_fhe(
            engine,
            keys,
            R_cts,
            Q_cols,
            k,
            m,
            n,
            norm_sq_lo=(ns / margin),
            norm_sq_hi=(ns * margin),
            vtv_lo=vt / margin,
            vtv_hi=vt * margin,
            D_sqrt=D_sqrt,
            D_inv=D_inv,
        )
        gc.collect()

    return Q_cols, R_cts, diag_bounds


def solver(
    engine,
    keys,
    H: list,
    rhs: list,
    X_train,
    y_train,
    D_sqrt: int = 64,
    D_inv: int = 64,
    D_inv_backsub: int = 64,
) -> Tuple:
    """Solve H @ x = rhs fully in FHE and compute primal weights inside FHE.
    Returns (b_ct, w_ct, n): b_ct — encrypted bias, w_ct — encrypted weights, n — solution size."""
    m, n = len(H), len(H[0])

    Q_cols, R_cts, diag_bounds = _qr(engine, keys, H, D_sqrt=D_sqrt, D_inv=D_inv)

    backsub_budget = 2 * math.ceil(math.log2(D_inv_backsub + 1)) + 5
    Q_cols = [maybe_bootstrap(engine, keys, ct, backsub_budget) for ct in Q_cols]
    R_cts = [maybe_bootstrap(engine, keys, ct, backsub_budget) for ct in R_cts]

    c_ct = he_matmul_T_vec(engine, keys, Q_cols, rhs, m, n)
    del Q_cols
    gc.collect()

    x_ct = he_back_substitute(engine, keys, R_cts, c_ct, n,
                              diag_bounds=diag_bounds, D_inv=D_inv_backsub)
    del R_cts, c_ct
    gc.collect()

    x_ct = maybe_bootstrap(engine, keys, x_ct, 5)

    e0_vec = np.zeros(engine.slot_count)
    e0_vec[0] = 1.0
    e0_pt = engine.encode(e0_vec)
    b_ct = engine.multiply(x_ct, e0_pt)
    w_ct = he_primal_weights(engine, keys, x_ct, X_train, y_train)

    return b_ct, w_ct, n


def serialize_model(engine, keys, b_ct, w_ct, out_dir: str,
                    mode_str: str = "primal:linear") -> None:
    """Serialize model: save secret key, ciphertexts, and config (minimal storage).
    Public/relin/rotation keys are regenerated on load from secret key."""
    os.makedirs(out_dir, exist_ok=True)

    # Save only essential state: secret key, ciphertexts, and context config
    engine.write_secret_key(keys['secret_key'], f"{out_dir}/secret_key.bin")
    engine.write_ciphertext(b_ct, f"{out_dir}/bias.bin")
    engine.write_ciphertext(w_ct, f"{out_dir}/weights.bin")

    config = {
        'max_level': engine.max_level,
        'slot_count': engine.slot_count,
    }
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f)

    with open(f"{out_dir}/mode.txt", "w") as f:
        f.write(mode_str)


def load_model(out_dir: str, d: int, n_test: int = None):
    """Load serialized model and regenerate all derived keys from secret key.
    Returns (engine, keys, b_ct, w_ct, mode_str)."""
    with open(f"{out_dir}/config.json") as f:
        config = json.load(f)

    engine = Engine(max_level=config['max_level'])
    secret_key = engine.read_secret_key(f"{out_dir}/secret_key.bin")

    # Regenerate all derived keys from secret key (never stored on disk)
    public_key = engine.create_public_key(secret_key)
    relin_key = engine.create_relinearization_key(secret_key)
    rotation_key = engine.create_rotation_key(secret_key)

    keys = {
        'secret_key': secret_key,
        'public_key': public_key,
        'relin_key': relin_key,
        'rotation_key': rotation_key,
    }

    b_ct = engine.read_ciphertext(f"{out_dir}/bias.bin")
    w_ct = engine.read_ciphertext(f"{out_dir}/weights.bin")

    with open(f"{out_dir}/mode.txt") as f:
        mode_str = f.read().strip()

    return engine, keys, b_ct, w_ct, mode_str


def transpose(a: list) -> list:
    m, n = len(a), len(a[0])
    return [[a[i][j] for i in range(m)] for j in range(n)]


def verify_fhe(A: list, Q: list, R: list, tol: float = 1e-4) -> bool:
    """Verify ||A - QR||_F / ||A||_F, ||Q^T Q - I||_F, and max |R[i,j]| for i > j."""
    m, n = len(A), len(A[0])
    norm_A = fro_norm(A)

    rel_recon = fro_norm(sub(A, matmul(Q, R))) / norm_A if norm_A > 0 else 0.0
    print(f"  ||A - QR||_F / ||A||_F = {rel_recon:.2e}  (target < {tol:.0e})")

    identity = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    ortho_err = fro_norm(sub(matmul(transpose(Q), Q), identity))
    print(f"  ||Q^T Q - I||_F       = {ortho_err:.2e}  (target < {tol:.0e})")

    max_lower = max(
        (abs(R[i][j]) for i in range(m) for j in range(min(i, n))),
        default=0.0,
    )
    print(f"  max |R[i,j]| (i > j)  = {max_lower:.2e}  (target < {tol:.0e})")

    ok = rel_recon < tol and ortho_err < tol and max_lower < tol
    print(f"  PASS: {ok}\n")
    return ok


# ─── Compatibility exports for lssvm_cipher.py ───

def decrypt_vector(engine, keys, ct, length: int) -> list:
    """Decrypt and extract first length values (compatible with lssvm_cipher.py)."""
    result = engine.decrypt(ct, keys['secret_key']).real
    return result[:length].tolist()


def get_slot_count(engine) -> int:
    """Get slot count (compatible with lssvm_cipher.py)."""
    return engine.slot_count


def predict_cipher(engine, keys, b_ct, w_ct, X_test):
    """Score test samples using encrypted primal weights (compatible with lssvm_cipher.py)."""
    from .desilo_utils import sum_slots, safe_rotate

    slot_count = engine.slot_count
    n_test, d = X_test.shape
    e0_vec = np.zeros(slot_count)
    e0_vec[0] = 1.0
    e0_pt = engine.encode(e0_vec)

    scores_ct = None
    for j in range(n_test):
        xj = np.zeros(slot_count)
        xj[:d] = X_test[j]
        xj_pt = engine.encode(xj)

        dot = engine.multiply(w_ct, xj_pt)
        score = sum_slots(engine, keys, dot, d)
        score = engine.add(score, b_ct)
        score = engine.multiply(score, e0_pt)
        if j != 0:
            score = safe_rotate(engine, keys, score, -j)
        scores_ct = score if scores_ct is None else engine.add(scores_ct, score)

    return scores_ct


if __name__ == "__main__":
    from .desilo_utils import decrypt_matrix_cols, decrypt_matrix_rows

    print("=== Householder QR via desilofhe ===\n")

    test_sizes = [
        (2, 2),
        (3, 3),
        (5, 5),
        (10, 10),
    ]

    D_sqrt = 16
    D_inv = 16
    engine, keys = setup_crypto_context(0)

    for m, n in test_sizes:
        print(f"Test: {m}×{n}")
        A = [[float(i + j + 1) for j in range(n)] for i in range(m)]

        Q_cols, R_cts, _ = _qr(engine, keys, A, D_sqrt=D_sqrt, D_inv=D_inv)

        Q = decrypt_matrix_cols(engine, keys, Q_cols, m)
        R = decrypt_matrix_rows(engine, keys, R_cts, n)

        verify_fhe(A, Q, R, tol=1e-4)
