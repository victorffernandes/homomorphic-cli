"""LSSVM preprocessing: build the symmetric block matrix for Least Squares SVM.

Given dataset X, labels y, and regularisation parameter gamma, assembles:

    H = [0       1_N^T          ]   (N+1 x N+1)
        [1_N   Omega + (1/g)*I_N]

    rhs = [0, y_1, ..., y_N]^T

where Omega_ij = y_i * K(x_i, x_j) * y_j.

Kernel options
--------------
linear_kernel            : K(x,y) = x·y                    (default)
polynomial_kernel        : K(x,y) = (x·y + c)^degree
homogeneous_poly_kernel  : K(x,y) = (x·y)^degree

For kernels with explicit finite feature maps (linear, polynomial,
homogeneous_poly) the corresponding feature map functions allow full
primal-weight computation — no training data needed at inference.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Kernel functions ──────────────────────────────────────────────


def linear_kernel(X: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
    """K[i,j] = x_i · x_j"""
    if X2 is None:
        X2 = X
    return X @ X2.T


def polynomial_kernel(
    X: np.ndarray, X2: np.ndarray = None, degree: int = 2, c: float = 1.0
) -> np.ndarray:
    """K[i,j] = (x_i · x_j + c)^degree"""
    if X2 is None:
        X2 = X
    return (X @ X2.T + c) ** degree


def homogeneous_poly_kernel(
    X: np.ndarray, X2: np.ndarray = None, degree: int = 2
) -> np.ndarray:
    """K[i,j] = (x_i · x_j)^degree"""
    if X2 is None:
        X2 = X
    return (X @ X2.T) ** degree


# ── Feature maps (explicit φ such that K(x,y) = φ(x)·φ(y)) ───────


def poly_feature_map(X: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    """Explicit feature map for polynomial kernel (x·y + c)^degree.

    Uses sklearn PolynomialFeatures which scales interaction terms so that
    φ(x)·φ(y) == (x·y + c)^degree exactly when c=1 (standard normalisation).
    For c != 1, features are pre-scaled by sqrt(c) for the bias column.
    Output shape: (N, C(d+degree, degree)).
    """
    from sklearn.preprocessing import PolynomialFeatures

    phi = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(X)
    if c != 1.0:
        phi[:, 0] *= np.sqrt(c)  # scale bias term to match (x·y + c)^degree
    return phi


def homogeneous_poly_feature_map(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Explicit feature map for homogeneous polynomial kernel (x·y)^degree.

    No bias term. Output shape: (N, C(d+degree-1, degree)).
    """
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)


# ── Matrix assembly ───────────────────────────────────────────────


def build_omega(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Omega_ij = y_i * K_ij * y_j"""
    y_col = y.reshape(-1, 1)
    return y_col * K * y_col.T


def build_lssvm_matrix(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float,
    kernel=linear_kernel,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (N+1)x(N+1) block matrix H and the rhs vector.

    Parameters
    ----------
    X : (N, d) feature matrix
    y : (N,)   labels in {-1, +1}
    gamma : regularisation parameter (> 0)

    Returns
    -------
    H   : (N+1, N+1) symmetric block matrix
    rhs : (N+1,)     right-hand side vector [0, y_1, ..., y_N]
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    N = len(y)
    K = kernel(X)
    Omega = build_omega(K, y)

    H = np.zeros((N + 1, N + 1))
    H[0, 1:] = 1.0
    H[1:, 0] = 1.0
    H[1:, 1:] = Omega + (1.0 / gamma) * np.eye(N)

    rhs = np.zeros(N + 1)
    rhs[1:] = y

    return H, rhs


def prepare_iris_binary(
    class_idx: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Load Iris, split, scale, and return OvR binary sub-problems.

    Parameters
    ----------
    class_idx : If given, return only that class-vs-rest problem.
                If None, return all three OvR problems.
    test_size : Fraction of data held out for testing.
    random_state : Seed for reproducible splits.

    Returns
    -------
    List of (X_train, X_test, y_train, y_test, class_name) tuples.
    y values are +1 (target class) or -1 (rest).
    """
    iris = load_iris()
    X_all, y_all = iris.data, iris.target
    class_names = iris.target_names

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    indices = [class_idx] if class_idx is not None else range(len(class_names))

    results = []
    for c in indices:
        y_tr = np.where(y_train_raw == c, 1.0, -1.0)
        y_te = np.where(y_test_raw == c, 1.0, -1.0)
        results.append((X_train, X_test, y_tr, y_te, class_names[c]))

    return results


def prepare_dataset(
    X: np.ndarray, y: np.ndarray, gamma: float
) -> list[tuple[list, list, dict]]:
    """Build LSSVM matrices for binary or multi-class (OvR) data.

    Parameters
    ----------
    X : (N, d) feature matrix (already scaled)
    y : (N,)   integer class labels

    Returns
    -------
    List of (H_list, rhs_list, meta) where H_list/rhs_list are plain
    Python lists (compatible with FHE functions) and meta is a dict
    with 'y_binary' and 'class_label'.
    """
    classes = np.unique(y)

    if len(classes) == 2:
        y_binary = np.where(y == classes[1], 1.0, -1.0)
        H, rhs = build_lssvm_matrix(X, y_binary, gamma)
        meta = {"y_binary": y_binary, "class_label": f"{classes[1]} vs {classes[0]}"}
        return [(H.tolist(), rhs.tolist(), meta)]

    problems = []
    for c in classes:
        y_binary = np.where(y == c, 1.0, -1.0)
        H, rhs = build_lssvm_matrix(X, y_binary, gamma)
        meta = {"y_binary": y_binary, "class_label": f"class {c} vs rest"}
        problems.append((H.tolist(), rhs.tolist(), meta))
    return problems


if __name__ == "__main__":
    splits = prepare_iris_binary()
    for X_tr, X_te, y_tr, y_te, name in splits:
        H, rhs = build_lssvm_matrix(X_tr, y_tr, gamma=1.0)
        sym = np.allclose(H, H.T)
        cond = np.linalg.cond(H)
        print(
            f"{name} vs rest:  H shape {H.shape},  "
            f"symmetric: {sym},  cond: {cond:.1f}"
        )
