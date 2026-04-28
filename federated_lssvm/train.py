"""Federated LSSVM training via One-Shot FedAvg over CKKS-encrypted weights.

Pipeline:
  1. Load Iris (120 train / 30 test, stratified split).
  2. Partition all 120 training samples disjointly across k clients.
  3. Each client trains a local FHE LS-SVM on their partition.
  4. Server aggregates encrypted weight vectors: w_global = (1/k) * sum(w_ct_i).
  5. Evaluate global model on the 30-sample test set.
  6. Serialize to models/k={k}/class_{i}/.
"""

from __future__ import annotations

import sys
from config.parallel import bootstrap as _init_parallel

_init_parallel()

import os
import time
import importlib
import numpy as np
from openfhe import SerializeToFile, DeserializeCiphertext, BINARY

from lssvm.solvers.utils import depth_for_size
from lssvm.preprocessing import (
    prepare_iris_binary,
    build_lssvm_matrix,
    linear_kernel,
    polynomial_kernel,
    homogeneous_poly_kernel,
    poly_feature_map,
    homogeneous_poly_feature_map,
)
from lssvm.plain import predict_lssvm
from config.metrics import weight_relative_error

# ── solver backend ─────────────────────────────────────────────────
solv = importlib.import_module("lssvm.solvers.cg_cipher")

# ── configuration ──────────────────────────────────────────────────
D_SQRT = 4
D_INV = 4
D_INV_BACKSUB = 4
DEPTH_SAFETY = 1.15
DEPTH_OVERRIDE = None
N_OVERRIDE = None
GAMMA = 1.1
N_PER_CLASS_BASELINE = (
    2  # used only for the single-client FHE baseline (matches lssvm_cipher.py)
)

# ── kernel registry (local copy — avoids lssvm_cipher sys.argv side effect) ──
CLASS_KERNEL_SELECTION = {0: "linear", 1: "homo_poly", 2: "homo_poly"}
_KERNEL_REGISTRY = {
    "linear": (linear_kernel, None, "primal:linear"),
    "poly": (polynomial_kernel, poly_feature_map, "primal:poly:degree=2:c=1.0"),
    "homo_poly": (
        homogeneous_poly_kernel,
        homogeneous_poly_feature_map,
        "primal:homo_poly:degree=2",
    ),
}
CLASS_KERNELS = {
    idx: (name,) + _KERNEL_REGISTRY[name]
    for idx, name in CLASS_KERNEL_SELECTION.items()
}


# ── per-client checkpoint helpers ──────────────────────────────────


def _cts_exist(out_dir: str) -> bool:
    return os.path.exists(f"{out_dir}/bias.bin") and os.path.exists(
        f"{out_dir}/weights.bin"
    )


def _save_cts(out_dir: str, b_ct, w_ct) -> None:
    os.makedirs(out_dir, exist_ok=True)
    assert SerializeToFile(
        f"{out_dir}/bias.bin", b_ct, BINARY
    ), f"Failed to serialize bias to {out_dir}"
    assert SerializeToFile(
        f"{out_dir}/weights.bin", w_ct, BINARY
    ), f"Failed to serialize weights to {out_dir}"


def _load_cts(out_dir: str):
    b_ct, ok = DeserializeCiphertext(f"{out_dir}/bias.bin", BINARY)
    assert ok, f"Failed to deserialize bias from {out_dir}"
    w_ct, ok = DeserializeCiphertext(f"{out_dir}/weights.bin", BINARY)
    assert ok, f"Failed to deserialize weights from {out_dir}"
    return b_ct, w_ct


# ── subsample helper (inline copy from lssvm_cipher.py) ────────────
def _subsample_for_fhe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_train == 1.0)[0]
    idx_neg = np.where(y_train == -1.0)[0]
    n_pos = min(n_per_class, len(idx_pos))
    n_neg = min(n_per_class, len(idx_neg))
    sel_pos = rng.choice(idx_pos, size=n_pos, replace=False)
    sel_neg = rng.choice(idx_neg, size=n_neg, replace=False)
    sel = np.sort(np.concatenate([sel_pos, sel_neg]))
    return X_train[sel], y_train[sel]


# ── core federated functions ────────────────────────────────────────


def partition_all(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    base_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Distribute all training samples disjointly across k clients.

    Uses np.array_split so remainders are spread evenly (some clients get
    one extra sample). Each client's partition preserves the pos/neg ratio.
    """
    rng = np.random.default_rng(base_seed)

    pos_idx = np.where(y == 1.0)[0].copy()
    neg_idx = np.where(y == -1.0)[0].copy()

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_chunks = np.array_split(pos_idx, k)
    neg_chunks = np.array_split(neg_idx, k)

    partitions = []
    for i in range(k):
        indices = np.sort(np.concatenate([pos_chunks[i], neg_chunks[i]]))
        partitions.append((X[indices], y[indices]))
    return partitions


def fhe_aggregate(cc, b_cts: list, w_cts: list) -> tuple:
    """Average encrypted client models via FedAvg inside CKKS.

    Only uses EvalAdd (noise-cheap) and one EvalMult by plaintext scalar 1/k.
    No decryption is performed.
    """
    k = len(b_cts)

    b_sum = b_cts[0]
    for ct in b_cts[1:]:
        b_sum = cc.EvalAdd(b_sum, ct)
    b_global = cc.EvalMult(b_sum, 1.0 / k)

    w_sum = w_cts[0]
    for ct in w_cts[1:]:
        w_sum = cc.EvalAdd(w_sum, ct)
    w_global = cc.EvalMult(w_sum, 1.0 / k)

    return b_global, w_global


def plaintext_federated_reference(
    partitions_feat: list[tuple[np.ndarray, np.ndarray]],
    X_te_feat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute FedAvg in plaintext numpy for validation.

    Returns (predictions, w_avg, b_avg).
    """
    w_sum = None
    b_sum = 0.0
    for X_c, y_c in partitions_feat:
        H, rhs = build_lssvm_matrix(X_c, y_c, GAMMA)
        try:
            sol = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(H, rhs, rcond=None)[0]
        b_i = sol[0]
        alpha_i = sol[1:]
        w_i = X_c.T @ (alpha_i * y_c)
        b_sum += b_i
        w_sum = w_i if w_sum is None else w_sum + w_i

    k = len(partitions_feat)
    w_avg = w_sum / k
    b_avg = b_sum / k
    scores = X_te_feat @ w_avg + b_avg
    preds = np.sign(scores)
    preds[preds == 0] = 1.0
    return preds, w_avg, b_avg


def smoke_test_fedavg() -> None:
    """Validate FedAvg aggregation on two synthetic 7x7 LSSVM systems.

    Uses only 7x7 matrices (6 training samples → 7x7 H) to stay within the
    memory budget. Builds LSSVM-structured matrices via build_lssvm_matrix so
    the FHE solver's sign=+1 Householder assumption holds. No Iris data needed.
    Callable via --smoke-test flag.
    """
    print("=== smoke_test_fedavg: two synthetic 7x7 LSSVM systems ===")

    n_per_class = 3  # 3 pos + 3 neg = 6 training samples → 7x7 H
    n_feat = 4  # same dimensionality as Iris linear features

    # Synthetic client datasets with fixed seeds for reproducibility
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(10)

    X_c1 = rng0.random((2 * n_per_class, n_feat))
    y_c1 = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    H1_np, rhs1_np = build_lssvm_matrix(X_c1, y_c1, GAMMA)

    X_c2 = rng1.random((2 * n_per_class, n_feat))
    y_c2 = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    H2_np, rhs2_np = build_lssvm_matrix(X_c2, y_c2, GAMMA)

    # Plaintext expected averages of FedAvg
    sol1 = np.linalg.solve(H1_np, rhs1_np)
    sol2 = np.linalg.solve(H2_np, rhs2_np)

    b1, alpha1 = sol1[0], sol1[1:]
    b2, alpha2 = sol2[0], sol2[1:]
    w1 = X_c1.T @ (alpha1 * y_c1)
    w2 = X_c2.T @ (alpha2 * y_c2)

    expected_b_avg = (b1 + b2) / 2.0
    expected_w_avg = (w1 + w2) / 2.0  # shape: (n_feat,)

    # Set up 7x7 crypto context with feature_dim=n_feat for rotation keys
    depth = depth_for_size(
        7,
        7,
        D_SQRT,
        D_INV,
        D_INV_BACKSUB,
        safety_factor=DEPTH_SAFETY,
        depth_override=DEPTH_OVERRIDE,
    )
    print(f"  Context depth={depth} for 7x7 matrix ...")
    t0 = time.perf_counter()
    cc, keys = solv.setup_crypto_context(
        depth, matrix_size=7, n_test=1, feature_dim=n_feat, N=N_OVERRIDE
    )
    print(f"  Context ready in {time.perf_counter() - t0:.1f}s")

    # FHE solve for each synthetic client
    print("  FHE solve client 1 ...")
    t1 = time.perf_counter()
    b_ct1, w_ct1, _ = solv.solver(
        cc,
        keys,
        H1_np.tolist(),
        rhs1_np.tolist(),
        X_c1,
        y_c1,
        D_sqrt=D_SQRT,
        D_inv=D_INV,
        D_inv_backsub=D_INV_BACKSUB,
    )
    print(f"  Client 1 done in {time.perf_counter() - t1:.1f}s")

    print("  FHE solve client 2 ...")
    t2 = time.perf_counter()
    b_ct2, w_ct2, _ = solv.solver(
        cc,
        keys,
        H2_np.tolist(),
        rhs2_np.tolist(),
        X_c2,
        y_c2,
        D_sqrt=D_SQRT,
        D_inv=D_INV,
        D_inv_backsub=D_INV_BACKSUB,
    )
    print(f"  Client 2 done in {time.perf_counter() - t2:.1f}s")

    # FedAvg aggregation (the function under test)
    b_global, w_global = fhe_aggregate(cc, [b_ct1, b_ct2], [w_ct1, w_ct2])

    # Decrypt and compare against plaintext average
    w_fhe_avg = np.array(solv.decrypt_vector(cc, keys, w_global, n_feat))
    b_fhe_avg = solv.decrypt_vector(cc, keys, b_global, 1)[0]

    w_err = float(
        np.linalg.norm(w_fhe_avg - expected_w_avg)
        / (np.linalg.norm(expected_w_avg) + 1e-15)
    )
    # Combined [b, w] relative error avoids inflated ratios when b ≈ 0
    fhe_full = np.concatenate([[b_fhe_avg], w_fhe_avg])
    expected_full = np.concatenate([[expected_b_avg], expected_w_avg])
    full_err = float(
        np.linalg.norm(fhe_full - expected_full)
        / (np.linalg.norm(expected_full) + 1e-15)
    )

    print(f"  bias  (expected={expected_b_avg:.4f}, fhe={b_fhe_avg:.4f})")
    print(f"  weight relative error: {w_err:.2e}")
    print(f"  full [b,w] relative error: {full_err:.2e}")

    assert w_err < 0.05, f"Smoke test FAILED: weight error {w_err:.2e} > 5%"
    assert full_err < 0.05, f"Smoke test FAILED: full [b,w] error {full_err:.2e} > 5%"
    print("=== smoke_test_fedavg PASSED ===\n")


def _print_comparison_table(
    class_idx: int,
    name: str,
    y_te: np.ndarray,
    preds_single: np.ndarray,
    preds_fed_fhe: np.ndarray,
    preds_fed_plain: np.ndarray,
    preds_full_plain: np.ndarray,
    w_fhe_fed: np.ndarray,
    w_plain_fed: np.ndarray,
    k: int,
) -> None:
    def acc(p):
        return float(np.mean(p == y_te) * 100)

    print(f"  --- Class {class_idx} ({name} vs rest) comparison (k={k}) ---")
    print(f"  {'Approach':<40} | Accuracy")
    print(f"  {'-'*40}-+---------")
    print(
        f"  {'Single-client FHE (N_per_class=2, seed=42)':<40} | {acc(preds_single):.2f}%"
    )
    print(f"  {f'Federated FHE  ({k} clients avg.)':<40} | {acc(preds_fed_fhe):.2f}%")
    print(f"  {'Federated plaintext reference':<40} | {acc(preds_fed_plain):.2f}%")
    print(f"  {'Full-data plaintext reference':<40} | {acc(preds_full_plain):.2f}%")
    w_err = weight_relative_error(w_fhe_fed, w_plain_fed)
    print(f"  FHE fed weights vs plaintext fed weights: {w_err:.4e}")
    print()


def main(k: int = 3, serialize: bool = True, n_per_class: int | None = None) -> None:
    splits = prepare_iris_binary()
    n_test = len(splits[0][1])  # 30

    print(f"=== Federated FHE LS-SVM (Iris OvR, k={k} clients) ===")
    print(f"Dataset: 120 train / {n_test} test  |  GAMMA={GAMMA}")
    print(f"Chebyshev: sqrt={D_SQRT}, inv_qr={D_INV}, inv_backsub={D_INV_BACKSUB}\n")

    # Step 1: Compute all partitions (plaintext, no FHE) to determine context size
    all_partitions = {}
    max_client_n = 1
    for class_idx, (X_tr, _, y_tr, _, _) in enumerate(splits):
        parts = partition_all(X_tr, y_tr, k)
        if n_per_class is not None:
            parts = [
                _subsample_for_fhe(Xc, yc, n_per_class, seed=42 + class_idx * 1000 + i)
                for i, (Xc, yc) in enumerate(parts)
            ]
        all_partitions[class_idx] = parts
        max_n = max(len(y_c) for _, y_c in parts) + 1
        max_client_n = max(max_client_n, max_n)
    print(f"Max client H size: {max_client_n}x{max_client_n}  (k={k})\n")

    # Step 2: Feature dimension for rotation key generation
    sample_X = splits[0][0][:1]
    max_feat_dim = max_client_n
    for _, _, feature_map_fn, _ in CLASS_KERNELS.values():
        d = feature_map_fn(sample_X).shape[1] if feature_map_fn else sample_X.shape[1]
        max_feat_dim = max(max_feat_dim, d)

    # Step 3: One shared crypto context for all clients and all classes
    depth = (
        depth_for_size(
            max_client_n,
            max_client_n,
            D_SQRT,
            D_INV,
            D_INV_BACKSUB,
            safety_factor=DEPTH_SAFETY,
            depth_override=DEPTH_OVERRIDE,
        )
        # fhe_aggregate EvalMult(1) + predict_cipher 2×EvalMult(2)
        # + implicit ModDown(1) + decrypt margin(2)
        + 6
    )
    print(f"Setting up shared crypto context (depth={depth}) ...")
    t_ctx = time.perf_counter()
    cc, keys = solv.setup_crypto_context(
        depth,
        matrix_size=max_client_n,
        n_test=n_test,
        feature_dim=max_feat_dim,
        N=N_OVERRIDE,
    )
    slot_count = solv.get_slot_count(cc)
    print(
        f"Context ready in {time.perf_counter() - t_ctx:.1f}s  (slots={slot_count})\n"
    )

    classifiers_fed = []
    classifiers_single = []

    for class_idx, (X_tr, X_te, y_tr, y_te, name) in enumerate(splits):
        kernel_name, _, feature_map, mode_str = CLASS_KERNELS.get(
            class_idx, ("linear", linear_kernel, None, "primal:linear")
        )
        print(f"--- Class {class_idx} ({name} vs rest) [kernel={kernel_name}] ---")

        # Apply feature map to test set
        X_te_feat = feature_map(X_te) if feature_map else X_te
        X_tr_feat = feature_map(X_tr) if feature_map else X_tr

        # Full-data plaintext reference
        H_full, rhs_full = build_lssvm_matrix(X_tr_feat, y_tr, GAMMA)
        try:
            sol_full = np.linalg.solve(H_full, rhs_full)
        except np.linalg.LinAlgError:
            sol_full = np.linalg.lstsq(H_full, rhs_full, rcond=None)[0]
        alpha_full = sol_full[1:]
        preds_plain_full, _ = predict_lssvm(
            X_te_feat, X_tr_feat, alpha_full, y_tr, sol_full[0]
        )

        # Per-client FHE training (with checkpointing)
        parts = all_partitions[class_idx]
        b_cts, w_cts = [], []
        parts_feat = []
        for client_id, (X_c, y_c) in enumerate(parts):
            X_c_feat = feature_map(X_c) if feature_map else X_c
            parts_feat.append((X_c_feat, y_c))
            ckpt_dir = f"models/k={k}/class_{class_idx}/client_{client_id}"
            if _cts_exist(ckpt_dir):
                print(f"  [client {client_id}] Resuming from checkpoint {ckpt_dir}")
                b_ct_i, w_ct_i = _load_cts(ckpt_dir)
            else:
                H_c, rhs_c = build_lssvm_matrix(X_c_feat, y_c, GAMMA)
                print(
                    f"  [client {client_id}] H={H_c.shape}, cond={np.linalg.cond(H_c):.1f} ..."
                )
                t0 = time.perf_counter()
                b_ct_i, w_ct_i, _ = solv.solver(
                    cc,
                    keys,
                    H_c.tolist(),
                    rhs_c.tolist(),
                    X_c_feat,
                    y_c,
                    D_sqrt=D_SQRT,
                    D_inv=D_INV,
                    D_inv_backsub=D_INV_BACKSUB,
                )
                print(
                    f"  [client {client_id}] FHE solve: {time.perf_counter() - t0:.1f}s"
                )
                _save_cts(ckpt_dir, b_ct_i, w_ct_i)
                print(f"  [client {client_id}] Checkpoint saved.")
            b_cts.append(b_ct_i)
            w_cts.append(w_ct_i)

            # DEBUG: decrypt client weights immediately after solve
            _d = X_te_feat.shape[1]
            _b_i = solv.decrypt_vector(cc, keys, b_ct_i, 1)[0]
            _w_i = np.array(solv.decrypt_vector(cc, keys, w_ct_i, _d))
            _has_nan_b = np.isnan(_b_i)
            _has_nan_w = np.any(np.isnan(_w_i))
            print(
                f"  [DEBUG client {client_id}] b={_b_i:.4f}  "
                f"w_norm={np.linalg.norm(_w_i):.4f}  "
                f"NaN: b={_has_nan_b}, w={_has_nan_w}  "
                f"b_ct.level={b_ct_i.GetLevel()}  w_ct.level={w_ct_i.GetLevel()}"
            )

        # FHE aggregation
        print(f"  Aggregating {k} encrypted models ...")
        t_agg = time.perf_counter()
        b_global, w_global = fhe_aggregate(cc, b_cts, w_cts)
        print(f"  Aggregation: {time.perf_counter() - t_agg:.3f}s")

        # DEBUG: decrypt aggregated global model
        d = X_te_feat.shape[1]
        _b_g = solv.decrypt_vector(cc, keys, b_global, 1)[0]
        _w_g = np.array(solv.decrypt_vector(cc, keys, w_global, d))
        print(
            f"  [DEBUG global] b={_b_g:.4f}  w_norm={np.linalg.norm(_w_g):.4f}  "
            f"NaN: b={np.isnan(_b_g)}, w={np.any(np.isnan(_w_g))}  "
            f"b_global.level={b_global.GetLevel()}  w_global.level={w_global.GetLevel()}"
        )

        # Encrypted inference with global model
        t_inf = time.perf_counter()
        scores_ct = solv.predict_cipher(cc, keys, b_global, w_global, X_te_feat)
        print(f"  Cipher predict: {time.perf_counter() - t_inf:.4f}s")
        scores_fed = np.array(solv.decrypt_vector(cc, keys, scores_ct, n_test))
        # DEBUG: raw scores before sign()
        print(
            f"  [DEBUG scores] raw={np.round(scores_fed[:6], 4)}  "
            f"NaN={np.any(np.isnan(scores_fed))}  "
            f"scores_ct.level={scores_ct.GetLevel()}"
        )
        preds_fed = np.sign(scores_fed)
        preds_fed[preds_fed == 0] = 1.0

        # Decrypt federated weights for error reporting
        w_fhe_fed = np.array(solv.decrypt_vector(cc, keys, w_global, d))

        # Plaintext federated reference
        preds_plain_fed, w_plain_fed, _ = plaintext_federated_reference(
            parts_feat, X_te_feat
        )

        # Single-client FHE baseline (matches lssvm_cipher.py, N_PER_CLASS=2)
        X_s, y_s = _subsample_for_fhe(X_tr, y_tr, N_PER_CLASS_BASELINE, seed=42)
        X_s_feat = feature_map(X_s) if feature_map else X_s
        H_s, rhs_s = build_lssvm_matrix(X_s_feat, y_s, GAMMA)
        print(f"  Single-client baseline H={H_s.shape} ...")
        t_sc = time.perf_counter()
        b_ct_s, w_ct_s, _ = solv.solver(
            cc,
            keys,
            H_s.tolist(),
            rhs_s.tolist(),
            X_s_feat,
            y_s,
            D_sqrt=D_SQRT,
            D_inv=D_INV,
            D_inv_backsub=D_INV_BACKSUB,
        )
        print(f"  Single-client solve: {time.perf_counter() - t_sc:.1f}s")
        scores_ct_s = solv.predict_cipher(cc, keys, b_ct_s, w_ct_s, X_te_feat)
        scores_s = np.array(solv.decrypt_vector(cc, keys, scores_ct_s, n_test))
        preds_single = np.sign(scores_s)
        preds_single[preds_single == 0] = 1.0

        _print_comparison_table(
            class_idx,
            name,
            y_te,
            preds_single,
            preds_fed,
            preds_plain_fed,
            preds_plain_full,
            w_fhe_fed,
            w_plain_fed,
            k,
        )

        # Serialize global model
        if serialize:
            out_dir = f"models/k={k}/class_{class_idx}"
            solv.serialize_model(
                cc, keys, b_global, w_global, out_dir, mode_str=mode_str
            )
            print(f"  Global model serialized to {out_dir}/  [{mode_str}]")

        # # Delete per-client checkpoints now that aggregation succeeded
        # for client_id in range(k):
        #     ckpt_dir = f"models/k={k}/class_{class_idx}/client_{client_id}"
        #     if os.path.exists(ckpt_dir):
        #         shutil.rmtree(ckpt_dir)
        # print(f"  Intermediate checkpoints deleted.")

        classifiers_fed.append({"class_idx": class_idx, "scores": scores_fed})
        classifiers_single.append({"class_idx": class_idx, "scores": scores_s})

    # OvR multiclass accuracy
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split as tts

    iris = load_iris()
    _, _, _, y_test_raw = tts(
        iris.data,
        iris.target,
        test_size=0.2,
        stratify=iris.target,
        random_state=42,
    )

    def _ovr_acc(classifiers):
        score_matrix = np.column_stack([c["scores"] for c in classifiers])
        class_indices = np.array([c["class_idx"] for c in classifiers])
        predicted = class_indices[score_matrix.argmax(axis=1)]
        return np.mean(predicted == y_test_raw) * 100

    print(
        f"OvR Multiclass Accuracy (Federated FHE, k={k}): {_ovr_acc(classifiers_fed):.2f}%"
    )
    print(
        f"OvR Multiclass Accuracy (Single-client FHE):      {_ovr_acc(classifiers_single):.2f}%"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke-test" in args:
        smoke_test_fedavg()
        sys.exit(0)

    k = 3
    serialize = "--no-serialize" not in args
    n_per_class = next(
        (int(a.split("=", 1)[1]) for a in args if a.startswith("--n-per-class=")),
        None,
    )
    numeric_args = [a for a in args if a.lstrip("-").isdigit()]
    if numeric_args:
        k = int(numeric_args[0])

    main(k=k, serialize=serialize, n_per_class=n_per_class)
