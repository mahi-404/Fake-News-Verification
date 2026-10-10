"""
efnd.baselines.shallow_ml
============================

Section 5.1, Baseline 1: Shallow ML.

TF-IDF (unigrams + bigrams) feature extraction followed by either
Logistic Regression or a Linear SVM classifier, reported using the same
metric suite (Macro-F1, Balanced Accuracy, ROC-AUC, Brier Score, ECE) as
the proposed hybrid model, for direct comparability in Section 5's results
tables.

`LinearSVC` has no native `predict_proba`; to still report ROC-AUC, Brier
Score, and ECE for the SVM variant, it is wrapped in
`CalibratedClassifierCV` (sigmoid/Platt scaling) so both baselines expose
well-defined class probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from efnd.evaluation.metrics import ClassificationMetrics, compute_classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class ShallowMLConfig:
    model_type: Literal["logistic_regression", "linear_svm"] = "logistic_regression"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: Optional[int] = 50_000
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    stop_words: Optional[str] = "english"

    # Logistic Regression
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_class_weight: Optional[str] = "balanced"

    # Linear SVM
    svm_C: float = 1.0
    svm_max_iter: int = 5000
    svm_class_weight: Optional[str] = "balanced"
    svm_calibration_cv: int = 5

    ece_num_bins: int = 15
    random_state: int = 42


class ShallowMLBaseline:
    """
    TF-IDF (1-2 ngrams) -> {Logistic Regression | Linear SVM} baseline.

    Usage
    -----
    >>> config = ShallowMLConfig(model_type="logistic_regression")
    >>> baseline = ShallowMLBaseline(config)
    >>> baseline.fit(train_texts, train_labels)
    >>> metrics = baseline.evaluate(test_texts, test_labels)
    >>> baseline.save("baselines/tfidf_lr.pkl")
    """

    def __init__(self, config: ShallowMLConfig) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            ngram_range=config.ngram_range,
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            sublinear_tf=config.sublinear_tf,
            lowercase=config.lowercase,
            stop_words=config.stop_words,
        )
        self.classifier = self._build_classifier()
        self._is_fitted = False

    def _build_classifier(self):
        if self.config.model_type == "logistic_regression":
            return LogisticRegression(
                C=self.config.lr_C,
                max_iter=self.config.lr_max_iter,
                class_weight=self.config.lr_class_weight,
                random_state=self.config.random_state,
            )
        if self.config.model_type == "linear_svm":
            base_svm = LinearSVC(
                C=self.config.svm_C,
                max_iter=self.config.svm_max_iter,
                class_weight=self.config.svm_class_weight,
                random_state=self.config.random_state,
            )
            # Wrap for calibrated probability outputs (needed for
            # ROC-AUC / Brier / ECE, which LinearSVC cannot provide natively).
            return CalibratedClassifierCV(
                base_svm, method="sigmoid", cv=self.config.svm_calibration_cv
            )
        raise ValueError(f"Unsupported model_type '{self.config.model_type}'.")

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------
    def fit(self, texts: list[str], labels: list[int]) -> "ShallowMLBaseline":
        logger.info(
            "Fitting TF-IDF vectorizer (ngram_range=%s) over %d documents...",
            self.config.ngram_range, len(texts),
        )
        X = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF vocabulary size: %d", len(self.vectorizer.vocabulary_))

        logger.info("Fitting %s classifier...", self.config.model_type)
        self.classifier.fit(X, labels)
        self._is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Returns P(class) for both classes, shape (N, 2)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: list[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, texts: list[str], labels: list[int]) -> ClassificationMetrics:
        probs = self.predict_proba(texts)
        preds = np.argmax(probs, axis=1)
        pos_probs = probs[:, 1]

        metrics = compute_classification_metrics(
            labels=np.asarray(labels),
            preds=preds,
            pos_probs=pos_probs,
            ece_num_bins=self.config.ece_num_bins,
        )
        logger.info(
            "[%s] macro_f1=%.4f | balanced_acc=%.4f | roc_auc=%s | brier=%s | ece=%s",
            self.config.model_type, metrics.macro_f1, metrics.balanced_accuracy,
            f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A",
            f"{metrics.brier_score:.4f}" if metrics.brier_score is not None else "N/A",
            f"{metrics.ece:.4f}" if metrics.ece is not None else "N/A",
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier,
                    "is_fitted": self._is_fitted,
                },
                f,
            )
        logger.info("Saved baseline artifact to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ShallowMLBaseline":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(config=state["config"])
        instance.vectorizer = state["vectorizer"]
        instance.classifier = state["classifier"]
        instance._is_fitted = state["is_fitted"]
        return instance


# ============================================================================
# Data loading helper
# ============================================================================
def load_text_label_dataset(path: str | Path, text_col: str = "text", label_col: str = "label") -> tuple[list[str], list[int]]:
    """
    Loads a CSV/JSONL/Parquet dataset with `text_col` and `label_col`
    columns (label: 0 = LIKELY_REAL, 1 = LIKELY_FAKE).
    """
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=(path.suffix == ".jsonl"))
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Dataset at {path} must contain columns '{text_col}' and '{label_col}'; "
            f"found columns: {list(df.columns)}"
        )

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


# ============================================================================
# CLI entry point
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 1 (Section 5.1): TF-IDF + LogisticRegression/LinearSVM "
                    "shallow ML baseline for fake-news classification."
    )
    parser.add_argument("--train_data", type=str, required=True, help="CSV/JSONL/Parquet path")
    parser.add_argument("--test_data", type=str, required=True, help="CSV/JSONL/Parquet path")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--model_type", type=str, default="logistic_regression",
        choices=["logistic_regression", "linear_svm"],
    )
    parser.add_argument("--max_features", type=int, default=50_000)
    parser.add_argument("--ece_num_bins", type=int, default=15)
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to persist the fitted baseline")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write the metrics report")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = build_arg_parser()
    args = parser.parse_args()

    train_texts, train_labels = load_text_label_dataset(args.train_data, args.text_col, args.label_col)
    test_texts, test_labels = load_text_label_dataset(args.test_data, args.text_col, args.label_col)

    config = ShallowMLConfig(
        model_type=args.model_type,
        max_features=args.max_features,
        ece_num_bins=args.ece_num_bins,
    )
    baseline = ShallowMLBaseline(config)
    baseline.fit(train_texts, train_labels)
    metrics = baseline.evaluate(test_texts, test_labels)

    report = {
        "model_type": args.model_type,
        "train_data": args.train_data,
        "test_data": args.test_data,
        "ngram_range": list(config.ngram_range),
        "metrics": metrics.to_dict(),
    }
    print(json.dumps(report, indent=2))

    if args.save_path:
        baseline.save(args.save_path)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote baseline report to %s", output_path)


if __name__ == "__main__":
    main()
