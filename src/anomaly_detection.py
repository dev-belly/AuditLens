"""Machine-learning anomaly detection for AuditLens.

The rule engine can only find the patterns somebody thought to write down. An
Isolation Forest is used here to surface *previously unknown* combinations of
circumstances - a payment that is unremarkable on every individual dimension but
strange in combination.

Why Isolation Forest
--------------------
* **It is unsupervised.** There is no reliable supply of labelled fraud in
  practice: confirmed cases are rare, reporting is biased towards what has
  already been caught, and a model trained on past cases only learns to find
  more of the same. The model therefore learns what *normal* looks like and
  flags departures from it.
* **It isolates rather than profiles.** The algorithm builds random decision
  trees that split the data on random features at random thresholds. Anomalies
  are points that get isolated in few splits - they sit in sparse regions of the
  feature space. Normal points need many splits to separate.
* **It scales and needs no distributional assumptions.** Ledger features are
  heavily skewed and correlated; distance-based methods such as k-NN or
  Mahalanobis distance behave badly here.

Every score is accompanied by the engineered features that drove it, so the
output is never a bare ``anomaly = 1``.

Usage::

    python src/anomaly_detection.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/anomaly_detection.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import ML_FEATURE_COLUMNS, prepare_model_matrix
from src.utils import (
    MODEL_METRICS_JSON,
    RANDOM_SEED,
    TRANSACTIONS_FEATURES,
    Timer,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_json,
)

LOGGER = get_logger(__name__)

#: Expected first-pass exception rate. This is a *planning assumption*, not a
#: fitted parameter: an audit team decides in advance roughly how many items it
#: is willing to review, and the Isolation Forest is told the same thing. Setting
#: it from the injected labels would be leakage.
DEFAULT_CONTAMINATION: float = 0.03

#: Review budgets used for precision@k. An auditor can only look at a finite
#: number of items, so "if we review the top N riskiest vouchers, how many are
#: real?" is a far more useful question than accuracy over the whole ledger.
REVIEW_BUDGETS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.10)


@dataclass
class ModelEvaluation:
    """Performance of the anomaly detector against the labelled benchmark.

    Attributes:
        n_samples: Vouchers scored.
        n_anomalies: Labelled anomalies in the benchmark.
        n_flagged: Vouchers the model flagged.
        precision: Of the flagged vouchers, the share that are labelled anomalies.
        recall: Of the labelled anomalies, the share that were flagged.
        f1: Harmonic mean of precision and recall.
        roc_auc: Ranking quality of the anomaly score (threshold-free).
        average_precision: Area under the precision-recall curve. More informative
            than ROC-AUC when anomalies are rare, as they are here.
        confusion_matrix: ``[[TN, FP], [FN, TP]]``.
        precision_at_k: Precision when only the top *k* fraction by anomaly score
            is reviewed.
        recall_at_k: Recall at the same review budgets.
        threshold: The decision threshold applied to the anomaly score.
    """

    n_samples: int
    n_anomalies: int
    n_flagged: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    confusion_matrix: list[list[int]]
    precision_at_k: dict[str, float] = field(default_factory=dict)
    recall_at_k: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    @property
    def true_positives(self) -> int:
        """Number of correctly flagged anomalies."""
        return int(self.confusion_matrix[1][1])

    @property
    def false_positives(self) -> int:
        """Number of flagged vouchers that are not labelled anomalies."""
        return int(self.confusion_matrix[0][1])

    @property
    def false_negatives(self) -> int:
        """Number of labelled anomalies the model missed."""
        return int(self.confusion_matrix[1][0])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation with derived counts."""
        return {
            "n_samples": self.n_samples,
            "n_anomalies": self.n_anomalies,
            "n_flagged": self.n_flagged,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "confusion_matrix": self.confusion_matrix,
            "precision_at_k": {key: round(value, 4) for key, value in self.precision_at_k.items()},
            "recall_at_k": {key: round(value, 4) for key, value in self.recall_at_k.items()},
            "threshold": round(self.threshold, 6),
        }


def evaluate_detector(
    y_true: pd.Series,
    y_pred: pd.Series,
    anomaly_scores: pd.Series,
    threshold: float = 0.0,
) -> ModelEvaluation:
    """Evaluate the detector against the labelled benchmark.

    Args:
        y_true: Ground-truth ``anomaly_label`` (1 = injected anomaly).
        y_pred: Model flags (True/1 = flagged).
        anomaly_scores: Continuous 0-1 anomaly score.
        threshold: Decision threshold applied to the score.

    Returns:
        A populated :class:`ModelEvaluation`.
    """
    truth = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)
    scores = np.asarray(anomaly_scores, dtype=float)

    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).tolist()

    # ROC-AUC is undefined when only one class is present.
    if len(np.unique(truth)) > 1:
        roc_auc = float(roc_auc_score(truth, scores))
        average_precision = float(average_precision_score(truth, scores))
    else:
        roc_auc = float("nan")
        average_precision = float("nan")

    # Precision@k: sort by score, review the top k fraction.
    order = np.argsort(-scores)
    sorted_truth = truth[order]
    total_anomalies = max(1, int(truth.sum()))
    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    for budget in REVIEW_BUDGETS:
        k = max(1, int(round(len(sorted_truth) * budget)))
        hits = int(sorted_truth[:k].sum())
        precision_at_k[f"top_{budget:.1%}"] = hits / k
        recall_at_k[f"top_{budget:.1%}"] = hits / total_anomalies

    return ModelEvaluation(
        n_samples=int(len(truth)),
        n_anomalies=int(truth.sum()),
        n_flagged=int(predicted.sum()),
        precision=float(precision_score(truth, predicted, zero_division=0)),
        recall=float(recall_score(truth, predicted, zero_division=0)),
        f1=float(f1_score(truth, predicted, zero_division=0)),
        roc_auc=roc_auc,
        average_precision=average_precision,
        confusion_matrix=matrix,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        threshold=threshold,
    )


def run_anomaly_detection(
    transactions: pd.DataFrame | None = None,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Fit the Isolation Forest and score every voucher.

    Args:
        transactions: Feature table. Loaded from disk when omitted.
        contamination: Expected share of outliers. See
            :data:`DEFAULT_CONTAMINATION` for why this is a planning assumption.
        random_state: Seed for reproducibility.

    Returns:
        Mapping with the scored table, the fitted detector, the scaler, the
        evaluation and the imputation map.
    """
    ensure_directories()
    if transactions is None:
        if not TRANSACTIONS_FEATURES.exists():
            raise FileNotFoundError(
                f"{TRANSACTIONS_FEATURES} not found. Run `python src/feature_engineering.py` first."
            )
        transactions = load_dataframe(TRANSACTIONS_FEATURES)

    df = transactions.reset_index(drop=True).copy()

    # ---- 1. Build the model matrix -----------------------------------------
    matrix, fill_values = prepare_model_matrix(df, ML_FEATURE_COLUMNS)

    # ---- 2. Scale ----------------------------------------------------------
    # Isolation Forest splits on individual features, so it does not *require*
    # scaling the way a distance-based method does. It is scaled anyway because
    # the project specification calls for StandardScaler, and because it makes
    # the feature space interpretable: every feature is then measured in
    # standard deviations from its mean.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    # ---- 3. Fit the Isolation Forest ---------------------------------------
    detector = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    detector.fit(scaled)

    # ``score_samples`` returns the anomaly score: lower means more anomalous.
    # Negating it gives an "anomalousness" score where higher is more suspicious.
    raw_score = -detector.score_samples(scaled)
    flags = detector.predict(scaled) == -1

    # Map onto 0-1 using the observed range so the value is interpretable and
    # stable for the downstream weighted score.
    lower, upper = float(raw_score.min()), float(raw_score.max())
    anomaly_score = (raw_score - lower) / max(upper - lower, 1e-12)

    df["ml_anomaly_raw_score"] = raw_score.round(6)
    df["anomaly_score"] = anomaly_score.round(6)
    df["ml_anomaly_flag"] = flags
    df["ml_anomaly_rank"] = pd.Series(anomaly_score, index=df.index).rank(ascending=False, method="first").astype(int)

    # ---- 4. Evaluate against the labelled benchmark ------------------------
    evaluation: ModelEvaluation | None = None
    if "anomaly_label" in df.columns:
        threshold = float(np.min(anomaly_score[flags])) if flags.any() else 0.0
        evaluation = evaluate_detector(
            df["anomaly_label"], pd.Series(flags, index=df.index), pd.Series(anomaly_score, index=df.index), threshold
        )
        LOGGER.info(
            "Isolation Forest: precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f (flagged %s of %s)",
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
            evaluation.roc_auc,
            evaluation.n_flagged,
            evaluation.n_samples,
        )

    # ---- 5. Persist metrics ------------------------------------------------
    payload: dict[str, Any] = {
        "model": "IsolationForest",
        "hyperparameters": {
            "n_estimators": 300,
            "max_samples": "auto",
            "contamination": contamination,
            "random_state": random_state,
            "scaler": "StandardScaler",
        },
        "feature_columns": list(ML_FEATURE_COLUMNS),
        "n_features": len(ML_FEATURE_COLUMNS),
        "imputed_features": fill_values,
        "evaluation": evaluation.to_dict() if evaluation else None,
        "note": (
            "Metrics are measured against anomalies deliberately injected into the "
            "synthetic ledger. They are an upper bound on real-world performance: "
            "injected anomalies follow known patterns, whereas genuine irregularities "
            "do not announce themselves."
        ),
    }
    save_json(payload, MODEL_METRICS_JSON)
    LOGGER.info("Wrote %s", MODEL_METRICS_JSON)

    return {
        "transactions": df,
        "detector": detector,
        "scaler": scaler,
        "evaluation": evaluation,
        "metrics": payload,
    }


def main() -> int:
    """CLI entry point."""
    with Timer("anomaly detection"):
        result = run_anomaly_detection()

    evaluation: ModelEvaluation | None = result["evaluation"]
    if evaluation is None:
        LOGGER.warning("No ground-truth labels found; skipping evaluation output.")
        return 0

    LOGGER.info("--- Isolation Forest evaluation ---")
    LOGGER.info("  samples            %s", evaluation.n_samples)
    LOGGER.info("  labelled anomalies %s", evaluation.n_anomalies)
    LOGGER.info("  flagged            %s", evaluation.n_flagged)
    LOGGER.info("  confusion matrix   TN=%s FP=%s FN=%s TP=%s",
                evaluation.confusion_matrix[0][0], evaluation.false_positives,
                evaluation.false_negatives, evaluation.true_positives)
    LOGGER.info("  precision          %.4f", evaluation.precision)
    LOGGER.info("  recall             %.4f", evaluation.recall)
    LOGGER.info("  f1                 %.4f", evaluation.f1)
    LOGGER.info("  roc_auc            %.4f", evaluation.roc_auc)
    LOGGER.info("  average precision  %.4f", evaluation.average_precision)
    LOGGER.info("--- Precision / recall by review budget ---")
    for key, value in evaluation.precision_at_k.items():
        LOGGER.info("  %-10s precision=%.4f  recall=%.4f", key, value, evaluation.recall_at_k[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
