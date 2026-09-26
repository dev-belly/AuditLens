"""The review queue is a selection workpaper, not a synthetic answer key."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from src.review_plan import ReviewPolicy, benchmark_review_plan, build_review_plan, write_review_plan


@pytest.fixture
def population() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.DataFrame({
        "transaction_id": [f"TX{i}" for i in range(1, 9)],
        "transaction_date": pd.date_range("2025-01-01", periods=8),
        "vendor_id": ["V1"] * 8,
        "debit_amount": [100, 200, 300, 400, 500, 600, 700, 800],
        "audit_risk_score": [90, 80, 70, 99, 20, 10, 5, 0],
        "risk_level": ["High", "High", "High", "Critical", "Low", "Low", "Low", "Low"],
        "rule_alert_count": [1, 2, 1, 0, 0, 0, 0, 0],
        "risk_reason_text": ["Auditable finding"] * 8,
        "anomaly_label": [1, 1, 1, 0, 0, 0, 0, 0],
        "anomaly_type": ["synthetic"] * 8,
    })
    alerts = pd.DataFrame({
        "transaction_id": ["TX1", "TX2", "TX2", "TX3"],
        "rule_key": ["A", "A", "B", "C"],
        "rule_label": ["Rule A", "Rule A", "Rule B", "Rule C"],
    })
    return transactions, alerts


def test_rule_coverage_precedes_risk_ranking_and_preserves_the_budget(population) -> None:
    transactions, alerts = population
    plan = build_review_plan(transactions, alerts, ReviewPolicy(5, 0.2, 42))

    assert len(plan.queue) == 5
    assert plan.queue["transaction_id"].is_unique
    assert plan.queue["selection_route"].tolist() == [
        "rule_coverage", "rule_coverage", "risk_priority", "risk_priority", "random_control"
    ]
    assert plan.queue["transaction_id"].iloc[:4].tolist() == ["TX2", "TX3", "TX4", "TX1"]
    assert plan.summary["selected_rule_keys"] == ["A", "B", "C"]
    assert plan.summary["uncovered_rule_keys"] == []


def test_probability_applies_only_to_random_sample_of_remaining_frame(population) -> None:
    plan = build_review_plan(*population, ReviewPolicy(5, 0.2, 42))
    targeted = plan.queue.iloc[:4]
    control = plan.queue.iloc[-1]

    assert plan.summary["random_frame_count"] == 4
    assert control["random_inclusion_probability"] == pytest.approx(0.25)
    assert control["random_sampling_weight"] == pytest.approx(4.0)
    assert (targeted["random_inclusion_probability"] == "").all()
    assert (targeted["random_sampling_weight"] == "").all()


def test_input_order_and_answer_key_cannot_change_the_plan(population) -> None:
    transactions, alerts = population
    policy = ReviewPolicy(5, 0.2, 2026)
    original = build_review_plan(transactions, alerts, policy)
    shuffled = transactions.sample(frac=1, random_state=9).reset_index(drop=True)
    shuffled["anomaly_label"] = 1 - shuffled["anomaly_label"]
    shuffled["anomaly_type"] = "altered ground truth"
    changed = build_review_plan(shuffled, alerts.sample(frac=1, random_state=7), policy)

    pd.testing.assert_frame_equal(original.queue, changed.queue)
    assert original.summary == changed.summary
    assert not {"anomaly_label", "anomaly_type"} & set(original.queue.columns)


def test_budget_too_small_reports_the_uncovered_procedure(population) -> None:
    plan = build_review_plan(*population, ReviewPolicy(1, 0, 42))
    assert plan.queue["transaction_id"].tolist() == ["TX2"]
    assert plan.summary["uncovered_rule_keys"] == ["C"]


@pytest.mark.parametrize("budget,share", [(0, 0.2), (9, 0.2), (5, -0.1), (5, 1.1)])
def test_invalid_review_policy_fails_before_selection(population, budget, share) -> None:
    with pytest.raises(ValueError):
        build_review_plan(*population, ReviewPolicy(budget, share, 42))


@pytest.mark.parametrize("corruption", ["duplicate_voucher", "orphan_alert", "missing_alert"])
def test_broken_evidence_cannot_produce_a_review_workpaper(population, corruption) -> None:
    transactions, alerts = population
    if corruption == "duplicate_voucher":
        transactions.loc[7, "transaction_id"] = "TX1"
    elif corruption == "orphan_alert":
        alerts.loc[0, "transaction_id"] = "MISSING"
    else:
        alerts = alerts.iloc[1:].copy()
    with pytest.raises(ValueError):
        build_review_plan(transactions, alerts, ReviewPolicy(5, 0.2, 42))


def test_written_workpaper_checksum_matches_the_actual_csv(population, tmp_path) -> None:
    plan = build_review_plan(*population, ReviewPolicy(5, 0.2, 42))
    csv_path, json_path = tmp_path / "review.csv", tmp_path / "review.json"
    summary = write_review_plan(plan, csv_path, json_path)

    assert summary["queue_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert pd.read_csv(csv_path)["transaction_id"].is_unique
    assert '"queue_sha256"' in json_path.read_text()
    assert "anomaly_label" not in csv_path.read_text()


def test_synthetic_benchmark_is_computed_only_after_the_label_blind_selection(population) -> None:
    transactions, alerts = population
    plan = build_review_plan(transactions, alerts, ReviewPolicy(5, 0.2, 42))
    evaluation = benchmark_review_plan(plan, transactions)
    assert evaluation["injected_anomalies_in_population"] == 3
    assert evaluation["plan_injected_anomalies_found"] == 3

    changed = transactions.copy()
    changed["anomaly_label"] = "0"  # parquet also stores labels as strings
    assert build_review_plan(changed, alerts, ReviewPolicy(5, 0.2, 42)).queue.equals(plan.queue)
    assert benchmark_review_plan(plan, changed)["plan_injected_anomalies_found"] == 0
