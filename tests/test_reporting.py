"""Tests for the summary reporting helpers.

The important test here is the one that keeps the two bases of detector overlap
apart. An earlier version of this project reported only the flag-level overlap under
the name ``detections_not_caught_by_rules``, which reads as "anomalies the rules
missed". It was wrong by two orders of magnitude in spirit: 242 of the 244 vouchers it
counted were false positives. These tests pin the distinction so it cannot quietly
collapse again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import reporting
from src.reporting import detector_overlap


def test_high_risk_extract_orders_equal_scores_consistently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "high_risk.csv"
    monkeypatch.setattr(reporting, "HIGH_RISK_CSV", output)
    vouchers = pd.DataFrame(
        [
            {"transaction_id": "TX003", "audit_risk_score": 73.15, "risk_level": "High"},
            {"transaction_id": "TX001", "audit_risk_score": 73.15, "risk_level": "Critical"},
            {"transaction_id": "TX002", "audit_risk_score": 90.0, "risk_level": "High"},
            {"transaction_id": "TX004", "audit_risk_score": 99.0, "risk_level": "Low"},
        ]
    )

    reporting.write_high_risk_extract(vouchers)
    first = output.read_bytes()
    reporting.write_high_risk_extract(vouchers.iloc[::-1])

    assert output.read_bytes() == first
    assert pd.read_csv(output)["transaction_id"].tolist() == ["TX002", "TX001", "TX003"]


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal scored frame from (rule, model, anomaly) triples."""
    return pd.DataFrame(
        [
            {
                "transaction_id": f"TX{index:03d}",
                "rule_alert_count": int(rule),
                "ml_anomaly_flag": bool(model),
                "anomaly_type": anomaly,
            }
            for index, (rule, model, anomaly) in enumerate(rows)
        ]
    )


class TestDetectorOverlap:
    """Flag-level and anomaly-level overlap must not be conflated."""

    def test_flag_and_anomaly_overlap_are_reported_separately(self) -> None:
        """A model-only flag is not an anomaly the rules missed."""
        frame = _frame(
            [
                # An anomaly both detectors catch.
                (1, True, "duplicate_payment"),
                # An anomaly only the rules catch.
                (1, False, "weekend_posting"),
                # An anomaly only the model catches.
                (0, True, "unusual_vendor"),
                # An anomaly neither catches.
                (0, False, "rapid_payment"),
                # A model flag that is NOT an anomaly - the case that caused the bug.
                (0, True, None),
                (0, True, None),
                # A clean voucher nothing flagged.
                (0, False, None),
            ]
        )

        flag_overlap, anomaly_overlap = detector_overlap(frame)

        assert flag_overlap["flagged_by_rule"] == 2
        assert flag_overlap["flagged_by_model"] == 4
        # Three model-only flags: one is a real anomaly, two are not.
        assert flag_overlap["flagged_by_model_only"] == 3

        assert anomaly_overlap["injected_anomalies"] == 4
        assert anomaly_overlap["caught_by_rule"] == 2
        assert anomaly_overlap["caught_by_model"] == 2
        assert anomaly_overlap["caught_by_both"] == 1
        assert anomaly_overlap["caught_by_model_only"] == 1
        assert anomaly_overlap["caught_by_rule_only"] == 1
        assert anomaly_overlap["caught_by_neither"] == 1

    def test_model_only_flags_are_not_counted_as_missed_anomalies(self) -> None:
        """The specific error: two model-only *flags*, one model-only *anomaly*."""
        frame = _frame([(0, True, None), (0, True, None), (0, True, "rapid_payment")])

        flag_overlap, anomaly_overlap = detector_overlap(frame)

        assert flag_overlap["flagged_by_model_only"] == 3
        assert anomaly_overlap["caught_by_model_only"] == 1
        assert anomaly_overlap["caught_by_model_only"] < flag_overlap["flagged_by_model_only"]

    def test_anomaly_overlap_partitions_the_injected_population(self) -> None:
        """Every injected anomaly lands in exactly one of the four buckets."""
        frame = _frame(
            [
                (1, True, "duplicate_payment"),
                (1, True, "duplicate_payment"),
                (1, False, "weekend_posting"),
                (0, True, "unusual_vendor"),
                (0, False, "rapid_payment"),
                (0, False, "rapid_payment"),
                (0, False, None),
            ]
        )

        _, overlap = detector_overlap(frame)

        total = (
            overlap["caught_by_both"]
            + overlap["caught_by_model_only"]
            + overlap["caught_by_rule_only"]
            + overlap["caught_by_neither"]
        )
        assert total == overlap["injected_anomalies"]

    def test_coverage_percentage_matches_the_rule_bucket(self) -> None:
        frame = _frame(
            [(1, False, "a"), (1, False, "b"), (0, False, "c"), (0, False, "d")]
        )

        _, overlap = detector_overlap(frame)

        assert overlap["rule_coverage_pct"] == pytest.approx(50.0)
        assert overlap["caught_by_neither"] == 2

    def test_anomaly_overlap_is_empty_without_ground_truth(self) -> None:
        """Without labels the comparison is undefined, and saying so beats guessing."""
        frame = _frame([(1, True, None), (0, True, None)]).drop(columns=["anomaly_type"])

        flag_overlap, anomaly_overlap = detector_overlap(frame)

        assert anomaly_overlap == {}
        assert flag_overlap["flagged_by_model_only"] == 1

    def test_a_frame_with_no_flags_produces_zeroes(self) -> None:
        frame = _frame([(0, False, None), (0, False, None)])

        flag_overlap, anomaly_overlap = detector_overlap(frame)

        assert set(flag_overlap.values()) == {0}
        assert anomaly_overlap["injected_anomalies"] == 0
        assert anomaly_overlap["rule_coverage_pct"] == 0.0
