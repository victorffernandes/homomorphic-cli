"""LSSVM plaintext solver using numpy.linalg.solve.

Loads Iris, builds the block matrix via lssvm_preprocessing, solves each
one-vs-rest binary problem, and reports per-class + multiclass accuracy.
"""

from __future__ import annotations

import numpy as np

from lssvm_preprocessing import prepare_iris_binary, build_lssvm_matrix

GAMMA = 1.0


def back_substitute(R: list, c: list) -> list:
    """Solve upper-triangular Rx = c in O(n^2)."""
    n = len(c)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = c[i] - sum(R[i][j] * x[j] for j in range(i + 1, n))
        diag = R[i][i]
        if abs(diag) < 1e-10:
            print(f"  WARNING: near-zero pivot R[{i},{i}]={diag:.2e}, setting x[{i}]=0")
            x[i] = 0.0
        else:
            x[i] = s / diag
    return x


def matmul_T_vec(Q: list, rhs: list) -> list:
    """Compute c = Q^T @ rhs (plaintext, Q is list-of-rows)."""
    m = len(Q)
    n = len(Q[0])
    return [sum(Q[i][j] * rhs[i] for i in range(m)) for j in range(n)]


def solve_lssvm_plain(
    H: np.ndarray, rhs: np.ndarray
) -> tuple[float, np.ndarray]:
    """Solve H @ [b; alpha] = rhs.

    Returns (b, alpha).  Falls back to least-squares if H is singular.
    """
    cond = np.linalg.cond(H)
    if cond > 1e12:
        print(f"  WARNING: H is ill-conditioned (cond={cond:.2e}), using lstsq")
        solution, *_ = np.linalg.lstsq(H, rhs, rcond=None)
    else:
        solution = np.linalg.solve(H, rhs)

    b = float(solution[0])
    alpha = solution[1:]
    return b, alpha


def predict_lssvm(
    X_test: np.ndarray,
    X_train: np.ndarray,
    alpha: np.ndarray,
    y_train: np.ndarray,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using a primal-weight LSSVM model.

    For non-linear kernels with explicit feature maps (polynomial,
    homogeneous polynomial), pass phi(X_train) and phi(X_test) instead
    of raw features — the primal weight formula is kernel-agnostic.

    Returns (predictions, raw_scores).
    predictions: signs in {-1, +1}.
    raw_scores: continuous decision values (used for OvR argmax).
    """
    w = X_train.T @ (alpha * y_train)
    scores = X_test @ w + b
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    return preds, scores


def evaluate_ovr(
    classifiers: list[dict],
    X_test: np.ndarray,
    y_test_multiclass: np.ndarray,
) -> float:
    """Multiclass accuracy via argmax over OvR decision scores."""
    score_matrix = np.column_stack([c["scores"] for c in classifiers])
    predicted_classes = np.array([c["class_idx"] for c in classifiers])[
        score_matrix.argmax(axis=1)
    ]
    return float(np.mean(predicted_classes == y_test_multiclass))


def main():
    splits = prepare_iris_binary()
    classifiers = []

    print(f"=== LSSVM Plaintext Solver (Iris OvR) ===")
    print(f"Gamma = {GAMMA}\n")

    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        H, rhs = build_lssvm_matrix(X_tr, y_tr, GAMMA)

        b, alpha = solve_lssvm_plain(H, rhs)
        preds, scores = predict_lssvm(X_te, X_tr, alpha, y_tr, b)

        correct = int(np.sum(preds == y_te))
        total = len(y_te)
        acc = correct / total * 100
        n_sv = int(np.sum(np.abs(alpha) > 1e-6))

        print(f"Class {class_idx} ({name} vs rest):")
        print(f"  Binary accuracy: {acc:.2f}%  ({correct}/{total} correct)")
        print(f"  Support vectors (|alpha| > 1e-6): {n_sv}/{len(alpha)}")
        print()

        classifiers.append({
            "class_idx": class_idx,
            "scores": scores,
            "b": b,
            "alpha": alpha,
            "X_train": X_tr,
            "y_train": y_tr,
        })

    # Recover original integer test labels for multiclass eval
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    _, _, _, y_test_raw = train_test_split(
        iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
    )
    ovr_acc = evaluate_ovr(classifiers, X_te, y_test_raw)
    print(f"OvR Multiclass Accuracy: {ovr_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
