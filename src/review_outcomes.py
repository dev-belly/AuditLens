"""Validate a completed audit workpaper and summarize observed review outcomes.

The generated selection file is the immutable reference. Reviewers work on a
copy and may edit only the three review fields. This command deliberately
reports observed counts, not a population exception or fraud estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.review_plan import ROUTES
from src.utils import REVIEW_PLAN_CSV, REVIEW_PLAN_SUMMARY_JSON, load_json, save_json

REVIEW_FIELDS = frozenset({"review_outcome", "evidence_reference", "reviewer_notes"})
OUTCOMES = frozenset({"exception", "no_exception", "inconclusive"})


def summarize_completed_review(
    completed_path: Path,
    selection_path: Path = REVIEW_PLAN_CSV,
    record_path: Path = REVIEW_PLAN_SUMMARY_JSON,
) -> dict[str, Any]:
    """Check the selection provenance, then count human outcomes by route.

    A blank outcome means pending. A concluded outcome requires an evidence
    reference. The selection order and every non-review field must match the
    original workpaper, including the random inclusion probability.
    """
    record = load_json(record_path)
    selected_bytes = selection_path.read_bytes()
    selection_sha = hashlib.sha256(selected_bytes).hexdigest()
    if selection_sha != record.get("queue_sha256"):
        raise ValueError("original workpaper does not match its selection record")

    original = pd.read_csv(
        selection_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    completed_bytes = completed_path.read_bytes()
    completed = pd.read_csv(
        completed_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    if (
        original.columns.tolist() != completed.columns.tolist()
        or not REVIEW_FIELDS.issubset(original)
    ):
        raise ValueError(
            "completed workpaper must have the original columns in the original order"
        )
    if len(original) != len(completed) or len(original) != record.get("budget"):
        raise ValueError("completed workpaper must preserve the selected voucher count")
    if not original["transaction_id"].is_unique:
        raise ValueError("original workpaper contains duplicate voucher IDs")
    immutable = [column for column in original if column not in REVIEW_FIELDS]
    for column in immutable:
        if not original[column].equals(completed[column]):
            raise ValueError(f"completed workpaper changed selection field {column!r}")

    outcomes = completed["review_outcome"].str.strip().str.lower()
    invalid = sorted(set(outcomes) - OUTCOMES - {""})
    if invalid:
        raise ValueError(f"unrecognized review outcome: {invalid}")
    evidence = completed["evidence_reference"].str.strip()
    if ((outcomes != "") & (evidence == "")).any():
        raise ValueError("each recorded outcome needs an evidence reference")

    by_route: dict[str, dict[str, int]] = {}
    for route in ROUTES:
        route_outcomes = outcomes[original["selection_route"] == route]
        by_route[route] = {
            "selected": len(route_outcomes),
            "exception": int((route_outcomes == "exception").sum()),
            "no_exception": int((route_outcomes == "no_exception").sum()),
            "inconclusive": int((route_outcomes == "inconclusive").sum()),
            "pending": int((route_outcomes == "").sum()),
        }
    expected_routes = record.get("selected_by_route")
    if {
        route: counts["selected"] for route, counts in by_route.items()
    } != expected_routes:
        raise ValueError("selection routes disagree with the selection record")

    return {
        "selection_sha256": selection_sha,
        "completed_workpaper_sha256": hashlib.sha256(completed_bytes).hexdigest(),
        "selected_count": len(completed),
        "by_route": by_route,
        "interpretation": (
            "Observed reviewed cases only. Targeted selections are judgmental; "
            "random controls cover only the non-targeted frame. No population "
            "exception or fraud rate is inferred from this report."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a filled audit workpaper."
    )
    parser.add_argument(
        "completed_csv", type=Path, help="edited copy of review_plan.csv"
    )
    parser.add_argument("--selection", type=Path, default=REVIEW_PLAN_CSV)
    parser.add_argument("--record", type=Path, default=REVIEW_PLAN_SUMMARY_JSON)
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    args = parser.parse_args(argv)
    summary = summarize_completed_review(
        args.completed_csv, args.selection, args.record
    )
    if args.output is not None:
        save_json(summary, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
