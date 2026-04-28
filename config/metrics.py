"""Classification and weight-error metrics for FHE LSSVM evaluation.

All computations are in plaintext numpy — inputs are decrypted predictions
and weights produced by the FHE pipeline.

Binary labels convention: positive class = +1, negative class = -1.
"""

from __future__ import annotations

import numpy as np


# ── Weight comparison ─────────────────────────────────────────────────────────

def weight_relative_error(w_fhe: np.ndarray, w_plain: np.ndarray) -> float:
    """||w_fhe - w_plain|| / ||w_plain||.  Returns inf if w_plain is zero."""
    denom = np.linalg.norm(w_plain)
    if denom < 1e-15:
        return float("inf")
    return float(np.linalg.norm(w_fhe - w_plain) / denom)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def confusion_matrix(preds: np.ndarray, labels: np.ndarray):
    """Return (TP, FP, FN, TN) for binary ±1 labels."""
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == -1)))
    fn = int(np.sum((preds == -1) & (labels == 1)))
    tn = int(np.sum((preds == -1) & (labels == -1)))
    return tp, fp, fn, tn


# ── Per-class metrics ─────────────────────────────────────────────────────────

def precision(preds: np.ndarray, labels: np.ndarray) -> float:
    """TP / (TP + FP).  Returns 0 if no positive predictions."""
    tp, fp, _, _ = confusion_matrix(preds, labels)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(preds: np.ndarray, labels: np.ndarray) -> float:
    """TP / (TP + FN).  Returns 0 if no positive ground-truth samples."""
    tp, _, fn, _ = confusion_matrix(preds, labels)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """2 * precision * recall / (precision + recall).  Returns 0 if both are 0."""
    p = precision(preds, labels)
    r = recall(preds, labels)
    denom = p + r
    return 2 * p * r / denom if denom > 0 else 0.0


# ── Summary report ────────────────────────────────────────────────────────────

def print_class_report(
    class_idx: int,
    name: str,
    preds_cipher: np.ndarray,
    preds_plain: np.ndarray,
    y_te: np.ndarray,
    w_fhe: np.ndarray,
    w_plain_sub: np.ndarray,
    w_plain_full: np.ndarray,
) -> None:
    """Print the full per-class metrics block."""
    acc_cipher = float(np.mean(preds_cipher == y_te) * 100)
    acc_plain  = float(np.mean(preds_plain  == y_te) * 100)
    pred_match = float(np.mean(preds_cipher == preds_plain) * 100)

    w_err_sub  = weight_relative_error(w_fhe, w_plain_sub)
    w_err_full = weight_relative_error(w_fhe, w_plain_full)

    prec = precision(preds_cipher, y_te)
    rec  = recall(preds_cipher, y_te)
    f1   = f1_score(preds_cipher, y_te)

    tp, fp, fn, tn = confusion_matrix(preds_cipher, y_te)

    print(f"  FHE  accuracy:  {acc_cipher:.2f}%")
    print(f"  Plain accuracy: {acc_plain:.2f}%  (same subsample)")
    print(f"  ||w_fhe - w_plain_sub|| / ||w_plain_sub||   = {w_err_sub:.4e}  (same subsample)")
    print(f"  ||w_fhe - w_plain_full|| / ||w_plain_full|| = {w_err_full:.4e}  (full train set)")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Cipher vs plain prediction match = {pred_match:.2f}%")
