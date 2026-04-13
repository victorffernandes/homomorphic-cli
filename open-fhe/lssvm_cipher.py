"""LSSVM solver over CKKS-encrypted data using Householder QR.

Pipeline:
  1. Subsample Iris to keep H small enough for FHE.
  2. Encrypt H, run Householder QR in CKKS → encrypted Q, R.
  3. Compute c = Q^T rhs in FHE (Q encrypted, rhs plaintext).
  4. Back-substitute Rx = c fully in FHE → decrypt only the final solution.
  5. Predict on the full test set and compare with the plaintext solver.
"""

from __future__ import annotations

import sys
import time
import importlib
import numpy as np

from fhe_solvers.utils import sum_slots, safe_rotate
from fhe_solvers.utils import decrypt_vector, depth_for_size
from fhe_solvers.utils import he_matmul_T_vec, he_back_substitute
from lssvm_preprocessing import prepare_iris_binary, build_lssvm_matrix
from lssvm_plain import predict_lssvm

solver_name = sys.argv[1] if len(sys.argv) > 1 else "qr_householder_cipher_row"
solv = importlib.import_module(f"fhe_solvers.{solver_name}")

# ── configuration ──────────────────────────────────────────────────
N_PER_CLASS    = 10    # samples per binary class → H is (2*N+1) x (2*N+1)
D_SQRT         = 2    # Chebyshev degree for sqrt (QR step)
D_INV          = 2    # Chebyshev degree for 1/t (QR step — vtv reciprocal)
D_INV_BACKSUB  = 8    # Chebyshev degree for 1/t in back-sub (tight bounds → lower degree sufficient)
DEPTH_SAFETY   = 1.15 # calibrated FLEXIBLEAUTO safety factor
DEPTH_OVERRIDE = None # set int to bypass estimator during experimentation
GAMMA          = 1.0  # regularisation


def solve_lssvm_fhe(cc, keys, H: list, rhs: list,
                    D_sqrt: int, D_inv: int, D_inv_backsub: int):
    """Solve H @ x = rhs fully in FHE: QR + encrypted Q^T@rhs + encrypted back-sub with encrypted pivots.

    All stages use fully homomorphic operations:
      - QR: Householder reflections with encrypted norm and reciprocal (Chebyshev √ and 1/t).
      - Q^T@rhs: encrypted Q × plaintext rhs, sum of slots.
      - Back-substitute: encrypted pivot extraction + Chebyshev reciprocal, no plaintext decryption.

    Returns (x_ct, n) where x_ct is the encrypted solution [b, alpha_0, ..., alpha_{n-1}].
    """
    m, n = len(H), len(H[0])

    t0 = time.perf_counter()
    Q_cols, R_cts, diag_bounds = solv.solver(
        cc, keys, H, D_sqrt=D_sqrt, D_inv=D_inv,
    )
    print(f"    QR factorisation: {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    c_ct = he_matmul_T_vec(cc, Q_cols, rhs, m, n)
    print(f"    Q^T @ rhs (FHE): {time.perf_counter() - t1:.4f}s")

    t2 = time.perf_counter()
    x_ct = he_back_substitute(cc, keys, R_cts, c_ct, n,
                              diag_bounds=diag_bounds, D_inv=D_inv_backsub)
    print(f"    Back-sub (FHE, encrypted pivots): {time.perf_counter() - t2:.1f}s")

    return x_ct, n


def predict_lssvm_cipher(cc, x_ct, X_test: np.ndarray, X_train: np.ndarray,
                         y_train: np.ndarray, n: int):
    """Predict using encrypted LSSVM weights and plaintext input.

    x_ct:    encrypted solution [b, alpha_0, ..., alpha_{N-1}] in slots 0..n-1.
    X_test:  plaintext test data (n_test, d).
    X_train: plaintext training data (n_train, d).
    y_train: plaintext labels (n_train,).
    n:       number of elements in the solution vector (1 + n_train).

    For each test sample j the score is:
        score_j = b + sum_i alpha_i * y_i * (x_test_j . x_train_i)

    This is computed as an inner product of x_ct with a plaintext vector
    [1, K_{j,0}*y_0, K_{j,1}*y_1, ...] followed by a slot sum.

    Returns an encrypted ciphertext with score_j in slot j.
    """
    slots = cc.GetRingDimension() // 2
    n_test = len(X_test)
    n_train = len(X_train)

    K = X_test @ X_train.T  # (n_test, n_train)

    e0_vec = [0.0] * slots
    e0_vec[0] = 1.0
    e0_ptxt = cc.MakeCKKSPackedPlaintext(e0_vec)

    scores_ct = None
    for j in range(n_test):
        pv = [0.0] * slots
        pv[0] = 1.0  # multiplied by b
        for i in range(n_train):
            pv[i + 1] = float(K[j, i] * y_train[i])
        ptxt = cc.MakeCKKSPackedPlaintext(pv)

        prod = cc.EvalMult(x_ct, ptxt)
        score = sum_slots(cc, prod, n)
        score = cc.EvalMult(score, e0_ptxt)

        if j != 0:
            score = safe_rotate(cc, score, -j)
        scores_ct = score if scores_ct is None else cc.EvalAdd(scores_ct, score)

    return scores_ct


def subsample_for_fhe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select n_per_class samples from each binary class (+1 / -1)."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_train == 1.0)[0]
    idx_neg = np.where(y_train == -1.0)[0]

    n_pos = min(n_per_class, len(idx_pos))
    n_neg = min(n_per_class, len(idx_neg))
    if n_pos < n_per_class or n_neg < n_per_class:
        print(f"  WARNING: not enough samples, using {n_pos} pos / {n_neg} neg")

    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    sel = np.sort(np.concatenate([sel_pos, sel_neg]))

    return X_train[sel], y_train[sel]


def main():
    splits = prepare_iris_binary()
    classifiers = []

    # All sub-problems have the same H size → set up context once
    n = 2 * N_PER_CLASS + 1
    depth = depth_for_size(
        n,
        n,
        D_SQRT,
        D_INV,
        D_INV_BACKSUB,
        safety_factor=DEPTH_SAFETY,
        depth_override=DEPTH_OVERRIDE,
    )
    n_test = len(splits[0][1]) if splits else n
    print(f"=== LSSVM FHE Solver (Iris OvR) ===")
    print(f"Gamma={GAMMA}  N_per_class={N_PER_CLASS}  H size={n}x{n}  depth={depth}")
    print(f"Chebyshev degrees: sqrt={D_SQRT}, inv_qr={D_INV}, inv_backsub={D_INV_BACKSUB}\n")

    print("Setting up crypto context ...")
    t_ctx = time.perf_counter()
    if solver_name == "qr_householder_cipher_row":
        cc, keys = solv.setup_crypto_context(depth, matrix_size=n, n_test=n_test)
    else:
        cc, keys = solv.setup_crypto_context(depth)
    ring_dim = cc.GetRingDimension()
    print(f"Context ready in {time.perf_counter() - t_ctx:.1f}s  (N={ring_dim}, slots={ring_dim//2})\n")

    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        print(f"--- Class {class_idx} ({name} vs rest) ---")

        t_sub = time.perf_counter()
        X_sub, y_sub = subsample_for_fhe(X_tr, y_tr, N_PER_CLASS)
        print(f"  Subsampled {len(y_sub)} points in {time.perf_counter() - t_sub:.3f}s")

        t_mat = time.perf_counter()
        H_np, rhs_np = build_lssvm_matrix(X_sub, y_sub, GAMMA)
        print(f"  Built H ({H_np.shape[0]}x{H_np.shape[1]}), cond={np.linalg.cond(H_np):.1f} in {time.perf_counter() - t_mat:.3f}s")
        H_list = H_np.tolist()
        rhs_list = rhs_np.tolist()

        # ── FHE solve ──
        print(f"  Starting FHE QR solve ({n}x{n}, depth={depth}) ...")
        t0 = time.perf_counter()
        x_ct, n_sol = solve_lssvm_fhe(cc, keys, H_list, rhs_list,
                                      D_SQRT, D_INV, D_INV_BACKSUB)
        elapsed = time.perf_counter() - t0
        print(f"  FHE QR solve: {elapsed:.1f}s")

        # Decrypt solution
        solution = decrypt_vector(cc, keys, x_ct, n_sol)
        b_fhe = solution[0]
        alpha_fhe = np.array(solution[1:])

        preds_fhe, scores_fhe = predict_lssvm(X_te, X_sub, alpha_fhe, y_sub, b_fhe)
        acc_fhe = np.mean(preds_fhe == y_te) * 100

        # ── plaintext reference on same subsample ──
        sol_plain = np.linalg.solve(H_np, rhs_np)
        b_plain = sol_plain[0]
        alpha_plain = sol_plain[1:]
        preds_plain, scores_plain = predict_lssvm(X_te, X_sub, alpha_plain, y_sub, b_plain)
        acc_plain = np.mean(preds_plain == y_te) * 100

        # ── cipher prediction ──
        t_cp = time.perf_counter()
        scores_cipher_ct = predict_lssvm_cipher(cc, x_ct, X_te, X_sub, y_sub, n_sol)
        print(f"  Cipher predict: {time.perf_counter() - t_cp:.4f}s")
        scores_cipher = np.array(decrypt_vector(cc, keys, scores_cipher_ct, len(X_te)))
        preds_cipher = np.sign(scores_cipher)
        preds_cipher[preds_cipher == 0] = 1.0

        score_err = np.linalg.norm(scores_cipher - scores_plain) / max(np.linalg.norm(scores_plain), 1e-15)
        pred_match = np.mean(preds_cipher == preds_plain) * 100

        # ── comparison ──
        b_err = abs(b_fhe - b_plain) / max(abs(b_plain), 1e-15)
        alpha_err = np.linalg.norm(alpha_fhe - alpha_plain) / max(np.linalg.norm(alpha_plain), 1e-15)

        print(f"  FHE  accuracy: {acc_fhe:.2f}%")
        print(f"  Plain accuracy: {acc_plain:.2f}%  (same subsample)")
        print(f"  |b_err|/|b|  = {b_err:.4e}")
        print(f"  ||a_err||/||a|| = {alpha_err:.4e}")
        print(f"  Cipher vs plain scores ||err||/||s|| = {score_err:.4e}")
        print(f"  Cipher vs plain prediction match = {pred_match:.2f}%")
        print()

        classifiers.append({
            "class_idx": class_idx,
            "scores": scores_fhe,
        })

    # ── OvR multiclass accuracy ──
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    _, _, _, y_test_raw = train_test_split(
        iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
    )
    score_matrix = np.column_stack([c["scores"] for c in classifiers])
    class_indices = np.array([c["class_idx"] for c in classifiers])
    predicted = class_indices[score_matrix.argmax(axis=1)]
    ovr_acc = np.mean(predicted == y_test_raw) * 100
    print(f"OvR Multiclass Accuracy (FHE): {ovr_acc:.2f}%")


if __name__ == "__main__":
    main()
