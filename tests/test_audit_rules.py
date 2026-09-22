"""Tests for the audit rule engine.

Each rule is tested on a hand-built ledger containing exactly the pattern the rule
claims to detect, plus enough innocuous vouchers that a rule which simply flagged
everything would fail. That second part matters: a rule that returns all-True is
not detecting anything, and several of these tests assert the *absence* of a flag
precisely to catch that failure mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.audit_rules import (
    DUPLICATE_WINDOW_DAYS,
    AuditRuleEngine,
    RULE_DEFINITIONS,
)
from src.utils import (
    APPROVAL_THRESHOLD_CNY,
    RAPID_PAYMENT_HOURS,
    SPLIT_THRESHOLD_HIGH_RATIO,
    SPLIT_THRESHOLD_LOW_RATIO,
)
from tests.conftest import HOLIDAY, SATURDAY, WEEKDAY, make_transactions


def _engine(rows: list[dict]) -> AuditRuleEngine:
    """Build a rule engine over a small hand-built ledger."""
    return AuditRuleEngine(make_transactions(rows))


# --------------------------------------------------------------------------- #
# Rule 1 - duplicate payment
# --------------------------------------------------------------------------- #
class TestDuplicatePayment:
    """Same vendor, same invoice, same amount, paid twice."""

    def test_same_invoice_paid_twice_is_flagged(self) -> None:
        engine = _engine(
            [
                {"invoice_id": "INV-DUP", "debit_amount": 88_000.0, "amount": 88_000.0,
                 "credit_amount": 88_000.0},
                {"invoice_id": "INV-DUP", "debit_amount": 88_000.0, "amount": 88_000.0,
                 "credit_amount": 88_000.0},
                {"invoice_id": "INV-OTHER", "debit_amount": 42_000.0, "amount": 42_000.0,
                 "credit_amount": 42_000.0},
            ]
        )
        result = engine.rule_duplicate_payment()

        assert result.flagged.tolist() == [True, True, False]
        assert result.scores.iloc[0] == pytest.approx(1.0)
        assert "INV-DUP" in result.reasons.iloc[0]

    def test_different_amounts_on_one_invoice_are_not_duplicates(self) -> None:
        """A part-payment against one invoice is normal, not a duplicate."""
        engine = _engine(
            [
                {"invoice_id": "INV-PART", "debit_amount": 30_000.0, "amount": 30_000.0,
                 "credit_amount": 30_000.0},
                {"invoice_id": "INV-PART", "debit_amount": 12_000.0, "amount": 12_000.0,
                 "credit_amount": 12_000.0},
            ]
        )
        result = engine.rule_duplicate_payment()

        # The repeat-amount branch must not fire either: the amounts differ.
        assert not result.flagged.any()

    def test_repeat_amount_within_window_is_flagged_at_lower_confidence(self) -> None:
        """Same vendor and amount a few days apart, with different invoices."""
        engine = _engine(
            [
                {"invoice_id": "INV-A", "debit_amount": 25_000.0, "amount": 25_000.0,
                 "credit_amount": 25_000.0, "transaction_date": WEEKDAY},
                {"invoice_id": "INV-B", "debit_amount": 25_000.0, "amount": 25_000.0,
                 "credit_amount": 25_000.0,
                 "transaction_date": WEEKDAY + pd.Timedelta(days=3)},
            ]
        )
        result = engine.rule_duplicate_payment()

        assert result.flagged.all()
        # Weaker evidence than a literal repeat of the same invoice.
        assert result.scores.iloc[0] == pytest.approx(0.65)
        assert "within" in result.reasons.iloc[0]

    def test_repeat_amount_outside_window_is_not_flagged(self) -> None:
        engine = _engine(
            [
                {"invoice_id": "INV-A", "debit_amount": 25_000.0, "amount": 25_000.0,
                 "credit_amount": 25_000.0, "transaction_date": WEEKDAY},
                {"invoice_id": "INV-B", "debit_amount": 25_000.0, "amount": 25_000.0,
                 "credit_amount": 25_000.0,
                 "transaction_date": WEEKDAY + pd.Timedelta(days=DUPLICATE_WINDOW_DAYS + 5)},
            ]
        )
        result = engine.rule_duplicate_payment()

        assert not result.flagged.any()

    def test_distinct_vendors_with_identical_amounts_are_not_duplicates(self) -> None:
        engine = _engine(
            [
                {"invoice_id": "INV-A", "vendor_id": "V0001", "debit_amount": 25_000.0,
                 "amount": 25_000.0, "credit_amount": 25_000.0},
                {"invoice_id": "INV-B", "vendor_id": "V0002", "debit_amount": 25_000.0,
                 "amount": 25_000.0, "credit_amount": 25_000.0},
            ]
        )
        result = engine.rule_duplicate_payment()

        assert not result.flagged.any()


# --------------------------------------------------------------------------- #
# Rule 2 - weekend / holiday posting
# --------------------------------------------------------------------------- #
class TestWeekendPosting:
    """Vouchers dated outside the working week."""

    def test_saturday_is_flagged_and_tuesday_is_not(self) -> None:
        engine = _engine(
            [
                {"transaction_date": SATURDAY, "posting_date": SATURDAY},
                {"transaction_date": WEEKDAY, "posting_date": WEEKDAY},
            ]
        )
        result = engine.rule_weekend_posting()

        assert result.flagged.tolist() == [True, False]
        assert "Saturday" in result.reasons.iloc[0]

    def test_public_holiday_on_a_weekday_is_flagged(self) -> None:
        """Labour Day 2025 fell on a Thursday - a holiday, not a weekend."""
        engine = _engine([{"transaction_date": HOLIDAY, "posting_date": HOLIDAY}])
        result = engine.rule_weekend_posting()

        assert result.flagged.iloc[0]
        assert "public holiday" in result.reasons.iloc[0]
        assert HOLIDAY.day_name() == "Thursday", "fixture must not be a weekend"

    def test_holiday_scores_higher_than_a_plain_weekend(self) -> None:
        """A deliberate holiday posting is a stronger signal than a Saturday one."""
        engine = _engine(
            [
                {"transaction_date": HOLIDAY, "posting_date": HOLIDAY},
                {"transaction_date": SATURDAY, "posting_date": SATURDAY},
            ]
        )
        result = engine.rule_weekend_posting()

        assert result.scores.iloc[0] > result.scores.iloc[1]

    def test_score_is_bounded(self) -> None:
        engine = _engine([{"transaction_date": SATURDAY, "posting_date": SATURDAY}])
        result = engine.rule_weekend_posting()

        assert 0.0 < result.scores.iloc[0] <= 1.0


# --------------------------------------------------------------------------- #
# Rule 3 - large round amount
# --------------------------------------------------------------------------- #
class TestLargeRoundAmount:
    """Round amounts, but only where they are material and in the tail."""

    @staticmethod
    def _ledger_with_round_outlier() -> list[dict]:
        # 30 unremarkable, non-round vouchers so that percentile and materiality
        # thresholds are meaningful, then one large round payment.
        rows = [
            {
                "transaction_id": f"TX{i:04d}",
                "invoice_id": f"INV{i:04d}",
                "debit_amount": 3_000.0 + i * 311.0,
                "amount": 3_000.0 + i * 311.0,
                "credit_amount": 3_000.0 + i * 311.0,
            }
            for i in range(1, 31)
        ]
        rows.append(
            {
                "transaction_id": "TX-ROUND",
                "invoice_id": "INV-ROUND",
                "debit_amount": 500_000.0,
                "amount": 500_000.0,
                "credit_amount": 500_000.0,
                "description": "Supplier settlement per contract 2025-114",
            }
        )
        return rows

    def test_large_round_amount_is_flagged(self) -> None:
        engine = _engine(self._ledger_with_round_outlier())
        result = engine.rule_large_round_amount()

        flagged_ids = engine.df.loc[result.flagged, "transaction_id"].tolist()
        assert flagged_ids == ["TX-ROUND"]
        assert "multiple of" in result.reasons.iloc[-1]

    def test_round_but_immaterial_amount_is_not_flagged(self) -> None:
        """A CNY 12,000 stationery order is round and nobody cares."""
        rows = [
            {
                "transaction_id": f"TX{i:04d}",
                "invoice_id": f"INV{i:04d}",
                "debit_amount": 40_000.0 + i * 4_013.0,
                "amount": 40_000.0 + i * 4_013.0,
                "credit_amount": 40_000.0 + i * 4_013.0,
            }
            for i in range(1, 31)
        ]
        rows.append(
            {
                "transaction_id": "TX-SMALL-ROUND",
                "invoice_id": "INV-SMALL",
                "debit_amount": 12_000.0,
                "amount": 12_000.0,
                "credit_amount": 12_000.0,
            }
        )
        engine = _engine(rows)
        result = engine.rule_large_round_amount()

        assert "TX-SMALL-ROUND" not in engine.df.loc[result.flagged, "transaction_id"].tolist()

    def test_non_round_large_amount_is_not_flagged(self) -> None:
        """Being large is not the test; being large *and* round is."""
        rows = [
            {
                "transaction_id": f"TX{i:04d}",
                "invoice_id": f"INV{i:04d}",
                "debit_amount": 3_000.0 + i * 311.0,
                "amount": 3_000.0 + i * 311.0,
                "credit_amount": 3_000.0 + i * 311.0,
            }
            for i in range(1, 31)
        ]
        rows.append(
            {
                "transaction_id": "TX-ODD",
                "invoice_id": "INV-ODD",
                "debit_amount": 487_361.42,
                "amount": 487_361.42,
                "credit_amount": 487_361.42,
            }
        )
        engine = _engine(rows)
        result = engine.rule_large_round_amount()

        assert "TX-ODD" not in engine.df.loc[result.flagged, "transaction_id"].tolist()

    def test_modulo_alone_is_not_sufficient(self) -> None:
        """The specification explicitly forbids a bare ``amount % 1000 == 0`` test."""
        rows = [
            {
                "transaction_id": f"TX{i:04d}",
                "invoice_id": f"INV{i:04d}",
                "debit_amount": 500_000.0,  # every row round, every row large
                "amount": 500_000.0,
                "credit_amount": 500_000.0,
            }
            for i in range(1, 31)
        ]
        engine = _engine(rows)
        result = engine.rule_large_round_amount()

        # When *everything* is round, nothing is unusual. A bare modulo test would
        # flag all 30 vouchers.
        assert result.n_flagged == 0


# --------------------------------------------------------------------------- #
# Rule 5 - split transaction
# --------------------------------------------------------------------------- #
class TestSplitTransaction:
    """Payments clustered just below the approval threshold."""

    @staticmethod
    def _near_threshold(offset: float) -> float:
        """An amount inside the split-detection band."""
        low = APPROVAL_THRESHOLD_CNY * SPLIT_THRESHOLD_LOW_RATIO
        high = APPROVAL_THRESHOLD_CNY * SPLIT_THRESHOLD_HIGH_RATIO
        assert low < offset < high
        return offset

    def test_two_payments_just_below_threshold_on_one_day_are_flagged(self) -> None:
        first = self._near_threshold(APPROVAL_THRESHOLD_CNY * 0.95)
        second = self._near_threshold(APPROVAL_THRESHOLD_CNY * 0.92)
        engine = _engine(
            [
                {"invoice_id": "INV-S1", "debit_amount": first, "amount": first,
                 "credit_amount": first, "transaction_date": WEEKDAY},
                {"invoice_id": "INV-S2", "debit_amount": second, "amount": second,
                 "credit_amount": second, "transaction_date": WEEKDAY},
            ]
        )
        result = engine.rule_split_transaction()

        assert result.flagged.all()
        # Combined value exceeds the threshold, so the score escalates past 0.65.
        assert result.scores.iloc[0] == pytest.approx(0.90)
        assert "approval threshold" in result.reasons.iloc[0]

    def test_single_payment_below_threshold_is_not_flagged(self) -> None:
        amount = self._near_threshold(APPROVAL_THRESHOLD_CNY * 0.95)
        engine = _engine(
            [{"debit_amount": amount, "amount": amount, "credit_amount": amount}]
        )
        result = engine.rule_split_transaction()

        assert result.n_flagged == 0

    def test_payments_on_different_days_are_not_flagged(self) -> None:
        first = self._near_threshold(APPROVAL_THRESHOLD_CNY * 0.95)
        second = self._near_threshold(APPROVAL_THRESHOLD_CNY * 0.92)
        engine = _engine(
            [
                {"debit_amount": first, "amount": first, "credit_amount": first,
                 "transaction_date": WEEKDAY},
                {"debit_amount": second, "amount": second, "credit_amount": second,
                 "transaction_date": WEEKDAY + pd.Timedelta(days=1)},
            ]
        )
        result = engine.rule_split_transaction()

        assert result.n_flagged == 0

    def test_payments_across_different_vendors_are_not_flagged(self) -> None:
        first = self._near_threshold(APPROVAL_THRESHOLD_CNY * 0.95)
        engine = _engine(
            [
                {"vendor_id": "V0001", "debit_amount": first, "amount": first,
                 "credit_amount": first},
                {"vendor_id": "V0002", "debit_amount": first, "amount": first,
                 "credit_amount": first},
            ]
        )
        result = engine.rule_split_transaction()

        assert result.n_flagged == 0

    def test_three_way_split_scores_at_ceiling(self) -> None:
        amounts = [
            self._near_threshold(APPROVAL_THRESHOLD_CNY * ratio)
            for ratio in (0.95, 0.93, 0.91)
        ]
        engine = _engine(
            [
                {"invoice_id": f"INV-T{i}", "debit_amount": amount, "amount": amount,
                 "credit_amount": amount, "transaction_date": WEEKDAY}
                for i, amount in enumerate(amounts, start=1)
            ]
        )
        result = engine.rule_split_transaction()

        assert result.n_flagged == 3
        assert result.scores.max() == pytest.approx(1.0)

    def test_amount_above_the_threshold_is_out_of_scope(self) -> None:
        """A payment over the threshold needs approval anyway - not a split."""
        amount = APPROVAL_THRESHOLD_CNY * 1.5
        engine = _engine(
            [
                {"debit_amount": amount, "amount": amount, "credit_amount": amount},
                {"debit_amount": amount, "amount": amount, "credit_amount": amount},
            ]
        )
        result = engine.rule_split_transaction()

        assert result.n_flagged == 0


# --------------------------------------------------------------------------- #
# Rule 6 - self approval
# --------------------------------------------------------------------------- #
class TestSelfApproval:
    """Segregation of duties: the creator must not be the approver."""

    def test_creator_approving_own_voucher_is_flagged(self) -> None:
        engine = _engine(
            [
                {"created_by": "E0001", "approved_by": "E0001"},
                {"created_by": "E0001", "approved_by": "E0002"},
            ]
        )
        result = engine.rule_self_approval()

        assert result.flagged.tolist() == [True, False]
        assert "E0001" in result.reasons.iloc[0]


# --------------------------------------------------------------------------- #
# Rule 7 - rapid payment
# --------------------------------------------------------------------------- #
class TestRapidPayment:
    """Payment settled suspiciously soon after the invoice arrived.

    The rule reads ``payment_delay_hours`` - the gap between invoice receipt and
    payment - rather than the approval-to-payment gap. That is the control the rule
    is about: normal supplier terms are 30-60 days, so a payment settled the same
    day is either an error or an override.
    """

    def test_payment_settled_same_day_is_flagged(self) -> None:
        engine = _engine(
            [
                {"payment_delay_hours": 1.5},
                {"payment_delay_hours": 720.0},  # 30 days, entirely normal
            ]
        )
        result = engine.rule_rapid_payment()

        assert result.flagged.tolist() == [True, False]
        assert "hours" in result.reasons.iloc[0]
        assert "30-60 days" in result.reasons.iloc[0]

    def test_score_decreases_as_the_delay_grows(self) -> None:
        engine = _engine(
            [
                {"payment_delay_hours": 0.5},
                {"payment_delay_hours": RAPID_PAYMENT_HOURS - 0.5},
            ]
        )
        result = engine.rule_rapid_payment()

        assert result.flagged.all()
        assert result.scores.iloc[0] > result.scores.iloc[1]

    def test_delay_exactly_at_the_threshold_is_not_flagged(self) -> None:
        """The comparison is strict, so the boundary itself is out of scope."""
        engine = _engine([{"payment_delay_hours": RAPID_PAYMENT_HOURS}])
        result = engine.rule_rapid_payment()

        assert result.n_flagged == 0

    def test_missing_delay_is_not_treated_as_instant(self) -> None:
        """A voucher with no invoice has no delay - it must not look instantaneous.

        Imputing 0.0 here flagged 48% of the ledger, because every non-invoice
        voucher appeared to be paid instantly. The rule now requires the feature to
        be present.
        """
        engine = _engine([{"payment_delay_hours": float("nan")}])
        result = engine.rule_rapid_payment()

        assert result.n_flagged == 0


# --------------------------------------------------------------------------- #
# Rule 8 - suspicious description
# --------------------------------------------------------------------------- #
class TestSuspiciousDescription:
    """Narration containing fraud-indicative language."""

    def test_keyword_bearing_description_is_flagged(self) -> None:
        engine = _engine(
            [
                {"description": "Consulting fee - urgent payment, please expedite"},
                {"description": "Purchase of raw materials per PO 2025-0311"},
            ]
        )
        result = engine.rule_suspicious_description()

        assert result.flagged.tolist() == [True, False]
        assert result.reasons.iloc[0]

    def test_repeated_boilerplate_alone_is_not_scored(self) -> None:
        """System-generated narration repeats legitimately and must not be scored.

        The repetition signal was measured at precision 0.02 when it fed the score,
        so it is reported separately. This test pins that decision.
        """
        rows = [
            {
                "description": "Monthly depreciation charge - system generated",
                "vendor_id": f"V{i:04d}",
            }
            for i in range(1, 15)
        ]
        engine = _engine(rows)
        result = engine.rule_suspicious_description()
        repetition = engine.description_repetition_flag()

        assert result.n_flagged == 0
        assert repetition.any(), "the repetition signal should still be reported"


# --------------------------------------------------------------------------- #
# Rule 9 - rare account usage
# --------------------------------------------------------------------------- #
class TestRareAccountUsage:
    """Accounts used far less often than the rest of the chart of accounts."""

    def test_rare_account_is_flagged(self) -> None:
        rows = [
            {
                "transaction_id": f"TX{i:04d}",
                "account_code": "6401",
                "account_transaction_frequency": 0.10,
                "debit_amount": 20_000.0 + i,
                "amount": 20_000.0 + i,
                "credit_amount": 20_000.0 + i,
            }
            for i in range(1, 40)
        ]
        rows.append(
            {
                "transaction_id": "TX-RARE",
                "account_code": "1602",
                "account_transaction_frequency": 0.0005,
                "debit_amount": 150_000.0,
                "amount": 150_000.0,
                "credit_amount": 150_000.0,
            }
        )
        engine = _engine(rows)
        result = engine.rule_rare_account_usage()

        assert "TX-RARE" in engine.df.loc[result.flagged, "transaction_id"].tolist()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class TestEngineOrchestration:
    """The combined run: noisy-OR scoring, reason collection and evaluation."""

    def test_run_produces_one_score_and_flag_column_per_rule(self) -> None:
        engine = _engine([{"debit_amount": 1_000.0 + i, "amount": 1_000.0 + i,
                            "credit_amount": 1_000.0 + i} for i in range(5)])
        scored, alerts = engine.run()

        for definition in RULE_DEFINITIONS:
            assert definition.column in scored.columns
            assert f"{definition.key}_flag" in scored.columns

        assert "rule_risk_score" in scored.columns
        assert "rule_alert_count" in scored.columns
        assert "rule_reasons" in scored.columns
        assert set(alerts.columns) >= {
            "transaction_id", "rule_key", "rule_label", "rule_score", "risk_reason"
        }

    def test_rule_risk_score_stays_within_bounds(self) -> None:
        engine = _engine(
            [
                # One voucher that breaks several controls at once.
                {
                    "transaction_date": SATURDAY,
                    "posting_date": SATURDAY,
                    "created_by": "E0001",
                    "approved_by": "E0001",
                    "description": "Urgent payment - do not delay",
                    "debit_amount": 49_000.0,
                    "amount": 49_000.0,
                    "credit_amount": 49_000.0,
                },
                {"debit_amount": 3_100.0, "amount": 3_100.0, "credit_amount": 3_100.0},
            ]
        )
        scored, _ = engine.run()

        assert scored["rule_risk_score"].between(0.0, 1.0).all()

    def test_noisy_or_is_monotonic_in_the_number_of_indicators(self) -> None:
        """More independent indicators must never lower the combined score."""
        engine = _engine(
            [
                {
                    "transaction_date": SATURDAY,
                    "posting_date": SATURDAY,
                    "created_by": "E0001",
                    "approved_by": "E0001",
                    "description": "Urgent payment - do not delay",
                    "debit_amount": 49_000.0,
                    "amount": 49_000.0,
                    "credit_amount": 49_000.0,
                },
                {"debit_amount": 3_100.0, "amount": 3_100.0, "credit_amount": 3_100.0},
            ]
        )
        scored, _ = engine.run()

        assert scored["rule_risk_score"].iloc[0] > scored["rule_risk_score"].iloc[1]
        assert scored["rule_alert_count"].iloc[0] > 0
        assert scored["rule_alert_count"].iloc[1] == 0

    def test_every_alert_carries_a_human_readable_reason(self) -> None:
        """Explainability is a hard requirement: a flag with no reason is a bug."""
        engine = _engine(
            [
                {
                    "transaction_date": SATURDAY,
                    "posting_date": SATURDAY,
                    "created_by": "E0001",
                    "approved_by": "E0001",
                    "description": "Urgent payment - do not delay",
                    "debit_amount": 49_000.0,
                    "amount": 49_000.0,
                    "credit_amount": 49_000.0,
                }
            ]
        )
        scored, alerts = engine.run()

        assert not alerts.empty
        assert (alerts["risk_reason"].astype(str).str.len() > 10).all()
        assert len(scored["rule_reasons"].iloc[0]) > 0

    def test_evaluation_reports_precision_recall_and_f1(self) -> None:
        engine = _engine(
            [
                {"transaction_date": SATURDAY, "posting_date": SATURDAY},
                {"transaction_date": WEEKDAY, "posting_date": WEEKDAY},
            ]
        )
        scored, _ = engine.run()
        scored["anomaly_label"] = [1, 0]
        scored["anomaly_type"] = ["weekend_posting", None]

        evaluation = engine.evaluate(scored)

        assert set(evaluation["rule_key"]) == {definition.key for definition in RULE_DEFINITIONS}
        assert {"precision", "recall", "f1"}.issubset(evaluation.columns)
        assert evaluation["precision"].between(0.0, 1.0).all()
        assert evaluation["recall"].between(0.0, 1.0).all()

    def test_evaluation_falls_back_to_a_text_encoded_label(self) -> None:
        """The label arrives from parquet as text, not as an integer.

        Without ``anomaly_type`` the evaluation falls back to ``anomaly_label``.
        That column round-trips through parquet as ``"0"``/``"1"``, so comparing
        it against the integer ``1`` matches nothing and reports a silent zero.
        """
        engine = _engine(
            [
                {"transaction_date": SATURDAY, "posting_date": SATURDAY},
                {"transaction_date": WEEKDAY, "posting_date": WEEKDAY},
            ]
        )
        scored, _ = engine.run()
        scored["anomaly_label"] = pd.Series(["1", "0"], index=scored.index, dtype="string")

        evaluation = engine.evaluate(scored)
        weekend = evaluation.loc[evaluation["rule_key"] == "weekend_posting"].iloc[0]

        assert weekend["true_positives"] == 1
        assert weekend["recall"] == pytest.approx(1.0)

    def test_a_text_encoded_zero_is_not_read_as_an_anomaly(self) -> None:
        """``astype(bool)`` on the string ``"0"`` is ``True``.

        That is the more dangerous half of the same trap: it fails in the direction
        of marking everything anomalous, so the evaluation looks plausible while
        being meaningless. The negative case has to be asserted explicitly.
        """
        engine = _engine(
            [
                {"transaction_date": SATURDAY, "posting_date": SATURDAY},
                {"transaction_date": WEEKDAY, "posting_date": WEEKDAY},
            ]
        )
        scored, _ = engine.run()
        # Both vouchers are labelled normal, even though one triggers a rule.
        scored["anomaly_label"] = pd.Series(["0", "0"], index=scored.index, dtype="string")

        evaluation = engine.evaluate(scored)
        weekend = evaluation.loc[evaluation["rule_key"] == "weekend_posting"].iloc[0]

        assert weekend["true_positives"] == 0
        assert weekend["precision"] == pytest.approx(0.0)

    def test_engine_rejects_an_empty_table(self) -> None:
        with pytest.raises(ValueError):
            AuditRuleEngine(pd.DataFrame({"debit_amount": []}))

    def test_engine_rejects_a_table_without_an_amount_column(self) -> None:
        with pytest.raises(KeyError):
            AuditRuleEngine(pd.DataFrame({"transaction_id": ["TX1"]}))
