"""The model's feature matrix, and the one mistake that would invalidate everything.

The central claim of this project is that the Isolation Forest is *unsupervised* and
graded against labels it never saw. If a ground-truth column ever reached
``ML_FEATURE_COLUMNS`` the model would be reading the answer key: precision, recall and
ROC-AUC would all improve, every other test would still pass, and the resulting numbers
would be meaningless. Nothing checked for that.

Leakage is guarded in three other places - the dashboard grid builder, the rendered
pages, and the SQL query results - but not at the model, which is where it matters most.
The three existing guards all protect the *presentation* of the answer key; this file
protects the *use* of it.

Two contract invariants keep the count honest as well. ``ML_FEATURE_COLUMNS`` and
``FEATURE_DESCRIPTIONS`` must describe exactly the same set, and the committed
``outputs/reports/model_metrics.json`` must record the features the code actually used.
The README quotes "20 features" in three places and `docs/architecture.md` and
`docs/methodology.md` quote it too, so a drift here is a drift in the documentation.

Nothing here reads ``data/``, which is gitignored. Everything is either a module
constant or a committed report, so the file runs on a clean checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.database import GROUND_TRUTH_COLUMNS
from src.feature_engineering import FEATURE_DESCRIPTIONS, ML_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_METRICS = PROJECT_ROOT / "outputs" / "reports" / "model_metrics.json"

#: Only the artefact checks need the pipeline to have run. The leakage guards are
#: constants-versus-constants and must never skip - a skipped test is a test that
#: cannot fail, which is the whole problem this file exists to solve.
requires_metrics = pytest.mark.skipif(
    not MODEL_METRICS.exists(),
    reason="Run `python src/run_pipeline.py` to write model_metrics.json.",
)


def _metrics() -> dict:
    return json.loads(MODEL_METRICS.read_text("utf-8"))


class TestNoLeakage:
    """The model must not be able to see the answer key."""

    def test_the_ground_truth_columns_are_the_two_we_expect(self) -> None:
        """Guards the check below: an empty tuple would make it vacuously true."""
        assert set(GROUND_TRUTH_COLUMNS) == {"anomaly_label", "anomaly_type"}, (
            "GROUND_TRUTH_COLUMNS changed, so the leakage check no longer covers "
            f"what it was written for: {GROUND_TRUTH_COLUMNS}"
        )

    def test_no_ground_truth_column_is_a_model_feature(self) -> None:
        """The single most damaging silent bug this project could ship."""
        leaked = sorted(set(ML_FEATURE_COLUMNS) & set(GROUND_TRUTH_COLUMNS))
        assert not leaked, (
            "the model is trained on ground truth, which invalidates every metric "
            f"in outputs/reports/model_metrics.json: {leaked}"
        )

    def test_no_feature_name_looks_like_an_injection_artefact(self) -> None:
        """A future injection bookkeeping column must not drift into the matrix."""
        suspicious = [
            name
            for name in ML_FEATURE_COLUMNS
            if name.startswith(("anomaly_", "injected_", "is_anomal", "ground_truth"))
        ]
        assert not suspicious, f"injection bookkeeping reached the model features: {suspicious}"


class TestTheFeatureContract:
    """The matrix and its documentation must describe the same features."""

    def test_the_feature_names_are_unique(self) -> None:
        duplicates = sorted({name for name in ML_FEATURE_COLUMNS if ML_FEATURE_COLUMNS.count(name) > 1})
        assert not duplicates, f"duplicate feature names: {duplicates}"

    def test_every_feature_is_described(self) -> None:
        missing = sorted(set(ML_FEATURE_COLUMNS) - set(FEATURE_DESCRIPTIONS))
        assert not missing, f"model features with no description: {missing}"

    def test_the_descriptions_describe_nothing_that_is_not_a_feature(self) -> None:
        """A description for a column the model never sees is a stale leftover."""
        extra = sorted(set(FEATURE_DESCRIPTIONS) - set(ML_FEATURE_COLUMNS))
        assert not extra, f"FEATURE_DESCRIPTIONS describes columns the model does not use: {extra}"

    def test_no_description_is_blank(self) -> None:
        blank = sorted(name for name, text in FEATURE_DESCRIPTIONS.items() if not text.strip())
        assert not blank, f"features described by an empty string: {blank}"


@requires_metrics
class TestTheCommittedMetrics:
    """The report must record the matrix the code actually built."""

    def test_the_metrics_record_the_same_feature_columns(self) -> None:
        recorded = list(_metrics()["feature_columns"])
        assert recorded == list(ML_FEATURE_COLUMNS), (
            "outputs/reports/model_metrics.json was produced from a different feature "
            "list than the code now defines; re-run the pipeline"
        )

    def test_the_metrics_record_the_same_feature_count(self) -> None:
        assert _metrics()["n_features"] == len(ML_FEATURE_COLUMNS)
