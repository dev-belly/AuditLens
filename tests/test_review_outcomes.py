"""A human review can change conclusions, but cannot rewrite the selection."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.review_outcomes import main, summarize_completed_review
from src.review_plan import ReviewPolicy, build_review_plan, write_review_plan


@pytest.fixture
def workpaper(tmp_path):
    transactions = pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "transaction_date": pd.date_range("2025-01-01", periods=4),
            "vendor_id": ["V1"] * 4,
            "debit_amount": [10, 20, 30, 40],
            "audit_risk_score": [95, 80, 30, 10],
            "risk_level": ["High", "High", "Low", "Low"],
            "rule_alert_count": [1, 0, 0, 0],
            "risk_reason_text": ["Review"] * 4,
        }
    )
    alerts = pd.DataFrame({"transaction_id": ["T1"], "rule_key": ["A"]})
    plan = build_review_plan(transactions, alerts, ReviewPolicy(3, 1 / 3, 42))
    original, record, completed = (
        tmp_path / name for name in ("selection.csv", "record.json", "completed.csv")
    )
    write_review_plan(plan, original, record)
    original_df = pd.read_csv(original, dtype=str, keep_default_na=False)
    assert original_df["selection_route"].tolist() == [
        "rule_coverage",
        "risk_priority",
        "random_control",
    ]
    original_df.to_csv(completed, index=False)
    return original, record, completed


def test_review_results_keep_targeted_and_random_evidence_separate(workpaper):
    original, record, completed = workpaper
    frame = pd.read_csv(completed, dtype=str, keep_default_na=False)
    frame.loc[0, ["review_outcome", "evidence_reference"]] = ["exception", "invoice-1"]
    frame.loc[2, ["review_outcome", "evidence_reference"]] = [
        "no_exception",
        "invoice-3",
    ]
    frame.to_csv(completed, index=False)

    result = summarize_completed_review(completed, original, record)

    assert (
        result["selection_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    )
    assert result["by_route"]["rule_coverage"]["exception"] == 1
    assert result["by_route"]["risk_priority"]["pending"] == 1
    assert result["by_route"]["random_control"]["no_exception"] == 1
    assert "population exception or fraud rate is inferred" in result["interpretation"]
    assert "invoice-1" not in json.dumps(result)


@pytest.mark.parametrize(
    "column,value",
    [
        ("transaction_id", "OTHER"),
        ("selection_route", "random_control"),
        ("random_inclusion_probability", "1"),
    ],
)
def test_review_cannot_change_the_sample_or_its_probabilities(workpaper, column, value):
    original, record, completed = workpaper
    frame = pd.read_csv(completed, dtype=str, keep_default_na=False)
    frame.loc[0, column] = value
    frame.to_csv(completed, index=False)
    with pytest.raises(ValueError, match="changed selection field"):
        summarize_completed_review(completed, original, record)


def test_original_selection_must_match_its_checksum(workpaper):
    original, record, completed = workpaper
    original.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="selection record"):
        summarize_completed_review(completed, original, record)


@pytest.mark.parametrize(
    "outcome,evidence,error",
    [
        ("suspected fraud", "file-1", "unrecognized"),
        ("exception", "", "evidence reference"),
    ],
)
def test_outcomes_need_an_allowed_value_and_evidence(
    workpaper, outcome, evidence, error
):
    original, record, completed = workpaper
    frame = pd.read_csv(completed, dtype=str, keep_default_na=False)
    frame.loc[0, ["review_outcome", "evidence_reference"]] = [outcome, evidence]
    frame.to_csv(completed, index=False)
    with pytest.raises(ValueError, match=error):
        summarize_completed_review(completed, original, record)


def test_cli_writes_only_counts_and_provenance(workpaper, tmp_path):
    original, record, completed = workpaper
    output = tmp_path / "review_result.json"
    assert (
        main(
            [
                str(completed),
                "--selection",
                str(original),
                "--record",
                str(record),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["selected_count"] == 3
