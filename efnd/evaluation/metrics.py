"""
efnd.evaluation.metrics
==========================

Shared evaluation metrics for both the proposed hybrid model
(`evaluate_generalization.py`) and the shallow ML baselines
(`efnd.baselines.shallow_ml`), so all experiments in Section 5 report
directly comparable numbers.

Metrics implemented:
    - Macro-F1
    - Balanced Accuracy
    - ROC-AUC
    - Brier Score
    - Expected Calibration Error (ECE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    macro_f1: float
    balanced_accuracy: float
    roc_auc: Optional[float]
    brier_score: Optional[float]
    ece: Optional[float]
    n_examples: int

    def to_dict(self) -> dict:
        return {
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "n_examples": self.n_examples,
        }


def compute_ece(
    labels: np.ndarray,
    pos_probs: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Expected Calibration Error over equal-width confidence bins.

    For binary classification, "confidence" is the predicted probability of
    the predicted class (i.e. max(p, 1-p)), and "accuracy" within a bin is
    the fraction of correct predictions in that bin. ECE is the confidence-
    weighted average absolute gap between per-bin accuracy and confidence:

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|
    """
    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray(pos_probs).astype(np.float64)

    preds = (pos_probs >= 0.5).astype(int)
    confidences = np.where(preds == 1, pos_probs, 1.0 - pos_probs)
    correctness = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    n = len(labels)
    ece = 0.0

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == num_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        n_b = int(in_bin.sum())
        if n_b == 0:
            continue

        acc_b = correctness[in_bin].mean()
        conf_b = confidences[in_bin].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)

    return float(ece)


def compute_classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    pos_probs: Optional[np.ndarray] = None,
    ece_num_bins: int = 15,
) -> ClassificationMetrics:
    """
    Computes the full Section-5 metric suite for a binary classification
    task (0 = LIKELY_REAL, 1 = LIKELY_FAKE).

    `pos_probs` (predicted probability of the positive/"fake" class) is
    required for ROC-AUC, Brier Score, and ECE; if omitted (or the label
    set is degenerate — only one class present), those fields are `None`.
    """
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    n = len(labels)

    macro_f1 = float(f1_score(labels, preds, average="macro")) if n > 0 else 0.0
    balanced_acc = float(balanced_accuracy_score(labels, preds)) if n > 0 else 0.0

    roc_auc: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None

    if pos_probs is not None and n > 0:
        pos_probs = np.asarray(pos_probs).astype(np.float64)
        if len(set(labels.tolist())) > 1:
            roc_auc = float(roc_auc_score(labels, pos_probs))
        brier = float(brier_score_loss(labels, pos_probs))
        ece = compute_ece(labels, pos_probs, num_bins=ece_num_bins)

    return ClassificationMetrics(
        macro_f1=macro_f1,
        balanced_accuracy=balanced_acc,
        roc_auc=roc_auc,
        brier_score=brier,
        ece=ece,
        n_examples=n,
    )
