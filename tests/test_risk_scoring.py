"""Tests for the composite Audit Risk Score.

The score is the deliverable the whole pipeline exists to produce, so these tests
are less about arithmetic than about the properties an auditor would rely on:

* the weights are a complete, normalised allocation of the score;
* a median-sized voucher with no indicators scores near zero, not near the middle;
* more evidence never lowers the score;
* every high score comes with an explanation.

The last one matters most. A risk score that cannot be explained is unusable in an
audit file, so "a reason exists for every flagged voucher" is asserted as a hard
invariant rather than a nicety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_scoring import (
    VENDOR_RISK_WEIGHTS,
    build_risk_reasons,
    build_summary,
    build_vendor_risk_table,
    compute_amount_risk,
    compute_audit_risk_score,
    compute_risk_components,
)
from src.utils import (
    MATERIALITY_THRESHOLD_CNY,
    RISK_BANDS,
    RISK_LEVEL_ORDER,
    RISK_WEIGHTS,
    risk_level_from_score,
)

COMPONENT_COLUMNS: tuple[str, ...] = (
    "risk_component_rule",
    "risk_component_ml",
    "risk_component_vendor",
    "risk_component_amount",
    "risk_component_statistical",
)


def _scored(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal scored transaction table.

    Every component column defaults to 0.0 so a test only has to state the
    components it cares about.
    """
    frame = pd.DataFrame(rows)
    for column in COMPONENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    for column in ("rule_alert_count", "debit_amount"):
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


# --------------------------------------------------------------------------- #
# Weight configuration
# --------------------------------------------------------------------------- #
class TestWeightConfiguration:
    """The composite must be a complete, normalised allocation."""

    def test_weights_sum_to_one(self) -> None:
        assert sum(RISK_WEIGHTS.values()) == pytest.approx(1.0)

    def test_every_weight_maps_to_a_component_column(self) -> None:
        expected = {f"risk_component_{key.replace('_risk', '')}" for key in RISK_WEIGHTS}
        assert expected == set(COMPONENT_COLUMNS)

    def test_rule_risk_carries_the_largest_weight(self) -> None:
        """Documented, defensible rules should outweigh the unsupervised model."""
        assert RISK_WEIGHTS["rule_risk"] > RISK_WEIGHTS["ml_risk"]
        assert RISK_WEIGHTS["ml_risk"] > RISK_WEIGHTS["amount_risk"]

    def test_vendor_weights_sum_to_one(self) -> None:
        assert sum(VENDOR_RISK_WEIGHTS.values()) == pytest.approx(1.0)

    def test_risk_bands_are_contiguous_and_cover_the_range(self) -> None:
        assert [band[0] for band in RISK_BANDS] == list(RISK_LEVEL_ORDER)
        for index, (_, _, upper) in enumerate(RISK_BANDS[:-1]):
            assert upper == RISK_BANDS[index + 1][1]


# --------------------------------------------------------------------------- #
# Band mapping
# --------------------------------------------------------------------------- #
class TestRiskBands:
    """Boundary behaviour of the Low / Medium / High / Critical mapping."""

    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.0, "Low"), (29.99, "Low"),
            (30.0, "Medium"), (59.99, "Medium"),
            (60.0, "High"), (79.99, "High"),
            (80.0, "Critical"), (100.0, "Critical"),
        ],
    )
    def test_boundaries_are_lower_inclusive(self, score: float, expected: str) -> None:
        assert risk_level_from_score(score) == expected

    def test_scores_above_one_hundred_are_critical(self) -> None:
        assert risk_level_from_score(150.0) == "Critical"


# --------------------------------------------------------------------------- #
# Amount component
# --------------------------------------------------------------------------- #
class TestAmountRisk:
    """Size contributes, but weakly."""

    def test_voucher_at_the_median_scores_zero_on_size(self) -> None:
        """The size term is percentile *above* the median, so the median is the zero point.

        With an even number of values one row sits exactly at the 50th percentile.
        """
        amounts = pd.Series([100.0, 200.0, 300.0, 400.0])
        risk = compute_amount_risk(pd.DataFrame({"debit_amount": amounts}))

        assert risk.iloc[1] == pytest.approx(0.0)  # 200, the 50th percentile

    def test_the_smallest_voucher_scores_zero(self) -> None:
        amounts = pd.Series([1_000.0, 50_000.0, 500_000.0, 5_000_000.0])
        risk = compute_amount_risk(pd.DataFrame({"debit_amount": amounts}))

        assert risk.iloc[0] == pytest.approx(0.0)

    def test_risk_increases_with_amount(self) -> None:
        amounts = pd.Series([1_000.0, 50_000.0, 500_000.0, 5_000_000.0])
        risk = compute_amount_risk(pd.DataFrame({"debit_amount": amounts}))

        assert risk.is_monotonic_increasing

    def test_round_amount_adds_to_the_score(self) -> None:
        """Compare two vouchers at the same percentile, differing only in roundness."""
        filler = [float(value) + 1.0 for value in range(1_000, 100_000, 1_000)]
        round_case = compute_amount_risk(pd.DataFrame({"debit_amount": pd.Series(filler + [200_000.0])}))
        odd_case = compute_amount_risk(pd.DataFrame({"debit_amount": pd.Series(filler + [200_001.0])}))

        assert round_case.iloc[-1] - odd_case.iloc[-1] == pytest.approx(0.20, abs=1e-9)

    def test_materiality_floor_adds_to_the_score(self) -> None:
        """Compare two vouchers at the same percentile, straddling the materiality floor."""
        filler = [float(value) + 1.0 for value in range(1_000, 100_000, 1_000)]
        above = compute_amount_risk(
            pd.DataFrame({"debit_amount": pd.Series(filler + [MATERIALITY_THRESHOLD_CNY + 1.0])})
        )
        below = compute_amount_risk(
            pd.DataFrame({"debit_amount": pd.Series(filler + [MATERIALITY_THRESHOLD_CNY - 1.0])})
        )

        assert above.iloc[-1] > below.iloc[-1]

    def test_round_amount_below_the_reporting_floor_does_not_count(self) -> None:
        """A CNY 5,000 round payment is not a manual-entry signature."""
        filler = [float(value) + 1.0 for value in range(1_000, 100_000, 1_000)]
        small_round = compute_amount_risk(pd.DataFrame({"debit_amount": pd.Series(filler + [5_000.0])}))
        small_odd = compute_amount_risk(pd.DataFrame({"debit_amount": pd.Series(filler + [5_001.0])}))

        assert small_round.iloc[-1] == pytest.approx(small_odd.iloc[-1], abs=1e-9)

    def test_score_is_bounded(self) -> None:
        amounts = pd.Series([1.0, 10_000_000_000.0, 100_000_000.0])
        risk = compute_amount_risk(pd.DataFrame({"debit_amount": amounts}))

        assert risk.between(0.0, 1.0).all()


# --------------------------------------------------------------------------- #
# Composite score
# --------------------------------------------------------------------------- #
class TestCompositeScore:
    """Combining the components."""

    def test_all_components_zero_gives_a_zero_score(self) -> None:
        frame = _scored([{column: 0.0 for column in COMPONENT_COLUMNS}])
        assert compute_audit_risk_score(frame).iloc[0] == pytest.approx(0.0)

    def test_all_components_one_gives_one_hundred(self) -> None:
        frame = _scored([{column: 1.0 for column in COMPONENT_COLUMNS}])
        assert compute_audit_risk_score(frame).iloc[0] == pytest.approx(100.0)

    def test_score_is_the_weighted_sum_of_the_components(self) -> None:
        components = {
            "risk_component_rule": 0.8,
            "risk_component_ml": 0.5,
            "risk_component_vendor": 0.2,
            "risk_component_amount": 0.4,
            "risk_component_statistical": 0.1,
        }
        expected = 100 * sum(
            RISK_WEIGHTS[key] * value
            for key, value in zip(
                ("rule_risk", "ml_risk", "vendor_risk", "amount_risk", "statistical_risk"),
                components.values(),
            )
        )
        frame = _scored([components])
        assert compute_audit_risk_score(frame).iloc[0] == pytest.approx(expected, abs=0.01)

    def test_score_never_exceeds_one_hundred(self) -> None:
        frame = _scored([{column: 5.0 for column in COMPONENT_COLUMNS}])
        assert compute_audit_risk_score(frame).iloc[0] == pytest.approx(100.0)

    def test_score_is_bounded_for_negative_components(self) -> None:
        frame = _scored([{column: -3.0 for column in COMPONENT_COLUMNS}])
        assert compute_audit_risk_score(frame).iloc[0] == pytest.approx(0.0)

    def test_more_evidence_never_lowers_the_score(self) -> None:
        """Monotonicity: the property an auditor relies on when triaging."""
        base = {column: 0.0 for column in COMPONENT_COLUMNS}
        scores = []
        for column in COMPONENT_COLUMNS:
            base = {**base, column: 0.6}
            scores.append(compute_audit_risk_score(_scored([base])).iloc[0])

        assert scores == sorted(scores)
        assert len(set(scores)) == len(scores), "each added indicator must move the score"

    def test_missing_component_raises(self) -> None:
        frame = pd.DataFrame({"risk_component_rule": [0.5]})
        with pytest.raises(KeyError):
            compute_audit_risk_score(frame)

    def test_nan_components_are_treated_as_zero(self) -> None:
        frame = _scored([{column: np.nan for column in COMPONENT_COLUMNS}])
        assert compute_audit_risk_score(frame).iloc[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Component assembly
# --------------------------------------------------------------------------- #
class TestComputeRiskComponents:
    """Assembling the five components onto the transaction table."""

    @staticmethod
    def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        scored = pd.DataFrame(
            {
                "transaction_id": ["TX1", "TX2"],
                "vendor_id": ["V1", "V2"],
                "debit_amount": [10_000.0, 900_000.0],
                "rule_risk_score": [0.0, 0.8],
                "anomaly_score": [0.1, 0.7],
            }
        )
        vendor_risk = pd.DataFrame(
            {
                "vendor_id": ["V1", "V2"],
                "vendor_risk_score": [20.0, 90.0],
            }
        )
        return scored, vendor_risk

    def test_all_five_components_are_produced(self) -> None:
        scored, vendor_risk = self._inputs()
        result = compute_risk_components(scored, vendor_risk)

        for column in COMPONENT_COLUMNS:
            assert column in result.columns

    def test_components_are_normalised_to_zero_one(self) -> None:
        scored, vendor_risk = self._inputs()
        result = compute_risk_components(scored, vendor_risk)

        for column in COMPONENT_COLUMNS:
            assert result[column].between(0.0, 1.0).all(), column

    def test_rule_and_ml_components_pass_through(self) -> None:
        scored, vendor_risk = self._inputs()
        result = compute_risk_components(scored, vendor_risk)

        assert result["risk_component_rule"].tolist() == pytest.approx([0.0, 0.8])
        assert result["risk_component_ml"].tolist() == pytest.approx([0.1, 0.7])

    def test_vendor_component_rescales_the_hundred_point_score(self) -> None:
        scored, vendor_risk = self._inputs()
        result = compute_risk_components(scored, vendor_risk)

        assert result["risk_component_vendor"].tolist() == pytest.approx([0.2, 0.9])

    def test_unknown_vendor_falls_back_to_zero(self) -> None:
        scored, _ = self._inputs()
        scored.loc[0, "vendor_id"] = "V-UNKNOWN"
        result = compute_risk_components(scored, pd.DataFrame({"vendor_id": ["V2"], "vendor_risk_score": [90.0]}))

        assert result["risk_component_vendor"].iloc[0] == pytest.approx(0.0)

    def test_a_high_risk_voucher_outscores_a_quiet_one(self) -> None:
        scored, vendor_risk = self._inputs()
        result = compute_risk_components(scored, vendor_risk)
        scores = compute_audit_risk_score(result)

        assert scores.iloc[1] > scores.iloc[0]


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
class TestRiskReasons:
    """Every flag must answer the question 'why?'."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": ["TX1", "TX2"],
                "debit_amount": [900_000.0, 10_000.0],
                "vendor_name": ["Vendor A", "Vendor B"],
                "rule_reasons": [
                    ["Duplicate payment: invoice INV-9 paid more than once"],
                    [],
                ],
                "rule_alert_count": [1, 0],
                "anomaly_score": [0.72, 0.10],
                "ml_anomaly_flag": [True, False],
                "audit_risk_score": [78.0, 12.0],
                "risk_level": ["High", "Low"],
            }
        )

    def test_a_reason_list_is_produced_for_every_voucher(self) -> None:
        reasons = build_risk_reasons(self._frame())

        assert len(reasons) == 2
        assert all(isinstance(items, (list, np.ndarray)) for items in reasons)

    def test_rule_reasons_are_carried_through(self) -> None:
        reasons = build_risk_reasons(self._frame())

        assert any("Duplicate payment" in item for item in reasons.iloc[0])

    def test_a_model_flag_is_never_emitted_without_its_features(self) -> None:
        """'The model said so' is not an audit explanation."""
        reasons = build_risk_reasons(self._frame())
        ml_only = [item for item in reasons.iloc[0] if "Isolation Forest" in item or "model" in item.lower()]

        for item in ml_only:
            assert len(item) > len("Isolation Forest flagged this voucher")

    def test_a_clean_voucher_produces_no_reasons(self) -> None:
        reasons = build_risk_reasons(self._frame())
        assert len(reasons.iloc[1]) == 0

    def test_reasons_are_capped(self) -> None:
        """A wall of text is as useless as no text."""
        frame = self._frame()
        frame["rule_reasons"] = [
            [f"Indicator {i}" for i in range(40)],
            [],
        ]
        reasons = build_risk_reasons(frame, max_reasons=5)

        assert len(reasons.iloc[0]) <= 5

    def test_capping_keeps_the_strongest_evidence(self) -> None:
        """Reasons are ordered strongest-first, so truncation drops the weakest."""
        frame = self._frame()
        frame["rule_reasons"] = [["Strongest indicator", "Weakest indicator"], []]
        reasons = build_risk_reasons(frame, max_reasons=1)

        assert "Strongest indicator" in reasons.iloc[0]

    def test_nan_rule_reasons_are_tolerated(self) -> None:
        frame = self._frame()
        frame["rule_reasons"] = [np.nan, np.nan]
        reasons = build_risk_reasons(frame)

        assert len(reasons) == 2

    def test_every_high_scoring_voucher_has_at_least_one_reason(self) -> None:
        """Hard invariant: a High or Critical score with no explanation is a bug."""
        frame = pd.DataFrame(
            {
                "transaction_id": ["TX1", "TX2", "TX3"],
                "debit_amount": [900_000.0, 950_000.0, 990_000.0],
                "vendor_name": ["A", "B", "C"],
                "rule_reasons": [[], [], []],
                "rule_alert_count": [0, 0, 0],
                "anomaly_score": [0.95, 0.97, 0.99],
                "ml_anomaly_flag": [True, True, True],
                "audit_risk_score": [62.0, 71.0, 88.0],
                "risk_level": ["High", "High", "Critical"],
            }
        )
        reasons = build_risk_reasons(frame)

        for index, row in frame.iterrows():
            if row["risk_level"] in {"High", "Critical"}:
                assert len(reasons.iloc[index]) > 0, f"{row['transaction_id']} has no explanation"


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
class TestBuildSummary:
    """Headline metrics must reconcile with the table they came from."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": [f"TX{i}" for i in range(10)],
                "vendor_id": ["V1"] * 6 + ["V2"] * 4,
                "debit_amount": [1_000.0] * 10,
                "audit_risk_score": [10.0, 20.0, 35.0, 40.0, 65.0, 70.0, 85.0, 15.0, 25.0, 30.0],
                "risk_level": [
                    "Low", "Low", "Medium", "Medium", "High",
                    "High", "Critical", "Low", "Low", "Medium",
                ],
                "rule_alert_count": [0, 0, 1, 0, 2, 1, 3, 0, 0, 1],
                "duplicate_payment_flag": [False] * 9 + [True],
                "weekend_posting_flag": [False] * 8 + [True, False],
            }
        )

    def test_counts_reconcile_with_the_table(self) -> None:
        frame = self._frame()
        summary = build_summary(frame)

        assert summary.total_transactions == 10
        assert summary.total_amount == pytest.approx(10_000.0)
        assert summary.unique_vendors == 2
        assert summary.flagged_transactions == int((frame["rule_alert_count"] > 0).sum())

    def test_risk_level_counts_cover_all_four_bands(self) -> None:
        summary = build_summary(self._frame())

        assert set(summary.risk_level_counts) == set(RISK_LEVEL_ORDER)
        assert sum(summary.risk_level_counts.values()) == 10

    def test_high_risk_count_includes_critical(self) -> None:
        summary = build_summary(self._frame())

        assert summary.high_risk_transactions == 3  # 2 High + 1 Critical
        assert summary.critical_transactions == 1

    def test_level_amounts_sum_to_the_total(self) -> None:
        summary = build_summary(self._frame())

        assert sum(summary.risk_level_amounts.values()) == pytest.approx(summary.total_amount)

    def test_average_score_is_the_table_mean(self) -> None:
        frame = self._frame()
        summary = build_summary(frame)

        assert summary.average_risk_score == pytest.approx(frame["audit_risk_score"].mean())


# --------------------------------------------------------------------------- #
# Vendor risk table
# --------------------------------------------------------------------------- #
class TestVendorRiskTable:
    """Aggregating voucher findings up to the counterparty."""

    @staticmethod
    def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        scored = pd.DataFrame(
            {
                "transaction_id": [f"TX{i}" for i in range(6)],
                "vendor_id": ["V1", "V1", "V1", "V2", "V2", "V3"],
                "vendor_name": ["Vendor One"] * 3 + ["Vendor Two"] * 2 + ["Vendor Three"],
                "debit_amount": [10_000.0, 20_000.0, 30_000.0, 500_000.0, 600_000.0, 1_000.0],
                "transaction_date": pd.to_datetime(["2025-03-01"] * 6),
                "rule_alert_count": [2, 1, 0, 1, 1, 0],
                "duplicate_payment_flag": [True, False, False, False, False, False],
                "weekend_posting_flag": [True, True, False, False, False, False],
                "self_approval_flag": [False, False, False, True, True, False],
                "ml_anomaly_flag": [False, False, False, True, True, False],
            }
        )
        vendor_features = pd.DataFrame(
            {
                "vendor_id": ["V1", "V2", "V3"],
                "vendor_name": ["Vendor One", "Vendor Two", "Vendor Three"],
                "vendor_category": ["Materials", "Consulting", "Office"],
                "vendor_transaction_count": [3, 2, 1],
                "vendor_total_amount": [60_000.0, 1_100_000.0, 1_000.0],
                "vendor_average_amount": [20_000.0, 550_000.0, 1_000.0],
                "vendor_max_amount": [30_000.0, 600_000.0, 1_000.0],
                "vendor_weekend_share": [0.66, 0.0, 0.0],
                "vendor_self_approval_share": [0.0, 1.0, 0.0],
                # Master risk is normalised to 0-1 by the generator, not 0-100.
                "vendor_master_risk_score": [0.20, 0.85, 0.05],
                "shared_bank_account": [False, True, False],
                "registration_date": pd.to_datetime(["2020-01-01", "2025-02-01", "2019-06-01"]),
            }
        )
        return scored, vendor_features

    def test_one_row_per_vendor(self) -> None:
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features)

        assert len(table) == 3
        assert set(table["vendor_id"]) == {"V1", "V2", "V3"}

    def test_sorted_by_descending_risk(self) -> None:
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features)

        assert table["vendor_risk_score"].is_monotonic_decreasing

    def test_scores_are_bounded_and_banded(self) -> None:
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features)

        assert table["vendor_risk_score"].between(0.0, 100.0).all()
        assert set(table["vendor_risk_level"]).issubset(set(RISK_LEVEL_ORDER))

    def test_alert_counts_reflect_the_transaction_flags(self) -> None:
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features).set_index("vendor_id")

        assert table.loc["V1", "alert_count"] == 3   # duplicate + 2 weekend
        assert table.loc["V2", "alert_count"] == 2   # 2 self approvals
        assert table.loc["V3", "alert_count"] == 0

    def test_the_worst_vendor_ranks_first(self) -> None:
        """V2 has self-approvals, a shared bank account and a new registration."""
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features)

        assert table.iloc[0]["vendor_id"] == "V2"

    def test_component_sub_scores_are_present_and_bounded(self) -> None:
        """Every component is normalised to 0-1 before it is weighted."""
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features)

        for name in VENDOR_RISK_WEIGHTS:
            column = f"vendor_risk_{name}"
            assert column in table.columns, column
            assert table[column].between(0.0, 1.0).all(), column

    def test_vendors_with_no_alerts_are_not_banded_high(self) -> None:
        scored, features = self._inputs()
        table = build_vendor_risk_table(scored, features).set_index("vendor_id")

        assert table.loc["V3", "vendor_risk_level"] in {"Low", "Medium"}
