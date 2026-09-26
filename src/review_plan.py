"""Reproducible, label-blind audit review plan.

The risk score ranks cases for targeted review, but an auditor also needs a
small probability sample from the *remaining* population. The two routes are
kept separate: only the random route has a known inclusion probability and can
support later estimation of exceptions in its sampling frame. No estimate of
misstatement or fraud is made before a human reviews the selected evidence.

Run ``python src/review_plan.py --budget 300 --random-share 0.2`` after the
pipeline, or let ``src/run_pipeline.py`` generate the default workpaper.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import (
    RANDOM_SEED,
    REVIEW_PLAN_BENCHMARK_JSON,
    REVIEW_PLAN_CSV,
    REVIEW_PLAN_SUMMARY_JSON,
    RULE_ALERTS_CSV,
    SCORED_TRANSACTIONS,
    as_flag_series,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_json,
)

LOGGER = get_logger(__name__)

DECISION_COLUMNS = (
    "transaction_id", "transaction_date", "vendor_id", "debit_amount",
    "audit_risk_score", "risk_level", "rule_alert_count", "risk_reason_text",
)
ROUTES = ("rule_coverage", "risk_priority", "random_control")


@dataclass(frozen=True)
class ReviewPolicy:
    budget: int = 300
    random_share: float = 0.2
    seed: int = RANDOM_SEED

    def validate(self, population: int) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, int):
            raise ValueError("review budget must be a positive integer")
        if not 1 <= self.budget <= population:
            raise ValueError(f"review budget must be between 1 and {population}")
        if not np.isfinite(self.random_share) or not 0 <= self.random_share <= 1:
            raise ValueError("random_share must be between 0 and 1")


@dataclass
class ReviewPlan:
    queue: pd.DataFrame
    summary: dict[str, Any]
    benchmark: dict[str, Any] | None = None


def _validate_inputs(transactions: pd.DataFrame, alerts: pd.DataFrame) -> None:
    missing = set(DECISION_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(f"scored ledger is missing review fields: {sorted(missing)}")
    if not {"transaction_id", "rule_key"}.issubset(alerts.columns):
        raise ValueError("alerts must contain transaction_id and rule_key")

    ids = transactions["transaction_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("scored ledger needs unique, nonblank transaction IDs")
    if not np.isfinite(transactions["audit_risk_score"].to_numpy(dtype=float)).all():
        raise ValueError("review scores must be finite")
    if not transactions["audit_risk_score"].between(0, 100).all():
        raise ValueError("review scores must be between 0 and 100")
    amounts = transactions["debit_amount"].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("review amounts must be finite and nonnegative")

    pairs = alerts[["transaction_id", "rule_key"]]
    if pairs.isna().any().any() or pairs.duplicated().any():
        raise ValueError("alerts need unique, complete voucher/rule pairs")
    if not pairs["transaction_id"].isin(ids).all():
        raise ValueError("alerts contain a voucher outside the review population")
    counts = pairs.groupby("transaction_id").size()
    expected = ids.map(counts).fillna(0).to_numpy(dtype=float)
    actual = transactions["rule_alert_count"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
        raise ValueError("rule_alert_count disagrees with the rule evidence")


def selection_fingerprint(transactions: pd.DataFrame, alerts: pd.DataFrame) -> str:
    """Fingerprint selection inputs; synthetic answer-key columns are excluded."""
    ledger = transactions[list(DECISION_COLUMNS)].sort_values("transaction_id")
    rules = alerts[["transaction_id", "rule_key"]].sort_values(["transaction_id", "rule_key"])
    digest = hashlib.sha256()
    for frame in (ledger, rules):
        digest.update(frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode())
    return digest.hexdigest()


def build_review_plan(
    transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    policy: ReviewPolicy = ReviewPolicy(),
) -> ReviewPlan:
    """Select rule coverage, highest-risk cases and random controls.

    All tie breaks use score, value and voucher ID in that order. Random draws
    use a sorted remaining frame, so input row order cannot change the plan.
    Ground-truth labels are deliberately absent from every decision input.
    """
    _validate_inputs(transactions, alerts)
    policy.validate(len(transactions))

    ledger = transactions[list(DECISION_COLUMNS)].copy()
    ranked = ledger.sort_values(
        ["audit_risk_score", "debit_amount", "transaction_id"],
        ascending=[False, False, True], kind="stable",
    )
    by_id = ledger.set_index("transaction_id")
    rule_sets = {
        tx_id: frozenset(keys)
        for tx_id, keys in alerts.groupby("transaction_id")["rule_key"]
        .agg(lambda values: sorted(set(values))).items()
    }
    rule_labels = (
        alerts.drop_duplicates("rule_key").set_index("rule_key")["rule_label"].to_dict()
        if "rule_label" in alerts.columns else {}
    )

    random_slots = int(np.ceil(policy.budget * policy.random_share))
    targeted_slots = policy.budget - random_slots
    selected: list[tuple[Any, str, str]] = []
    selected_ids: set[Any] = set()
    uncovered = set(alerts["rule_key"].unique())

    # Greedy set cover reserves one review for each available procedure where
    # possible. Among equal gains, the highest-risk voucher wins.
    while uncovered and len(selected) < targeted_slots:
        best_id = None
        best_new: set[str] = set()
        for tx_id in ranked["transaction_id"]:
            if tx_id in selected_ids:
                continue
            new = rule_sets.get(tx_id, frozenset()) & uncovered
            if len(new) > len(best_new):
                best_id, best_new = tx_id, set(new)
        if best_id is None:
            break
        labels = [str(rule_labels.get(key, key)) for key in sorted(best_new)]
        selected.append((best_id, "rule_coverage", "Covers procedures: " + ", ".join(labels)))
        selected_ids.add(best_id)
        uncovered -= best_new

    for tx_id in ranked["transaction_id"]:
        if len(selected) == targeted_slots:
            break
        if tx_id not in selected_ids:
            score = float(by_id.loc[tx_id, "audit_risk_score"])
            selected.append((tx_id, "risk_priority", f"Risk score {score:.2f}/100"))
            selected_ids.add(tx_id)

    random_frame = sorted(set(by_id.index) - selected_ids)
    if random_slots:
        rng = np.random.default_rng(policy.seed)
        draws = rng.choice(len(random_frame), size=random_slots, replace=False)
        for tx_id in sorted(random_frame[int(index)] for index in draws):
            selected.append((tx_id, "random_control", "Random control from non-targeted frame"))
            selected_ids.add(tx_id)

    inclusion_probability = random_slots / len(random_frame) if random_frame else 0.0
    rows: list[dict[str, Any]] = []
    for review_order, (tx_id, route, reason) in enumerate(selected, start=1):
        tx = by_id.loc[tx_id]
        rows.append({
            "review_order": review_order,
            "transaction_id": tx_id,
            "selection_route": route,
            "selection_reason": reason,
            "random_inclusion_probability": inclusion_probability if route == "random_control" else "",
            "random_sampling_weight": 1 / inclusion_probability if route == "random_control" else "",
            "transaction_date": str(pd.Timestamp(tx["transaction_date"]).date()),
            "vendor_id": "" if pd.isna(tx["vendor_id"]) else str(tx["vendor_id"]),
            "debit_amount": float(tx["debit_amount"]),
            "audit_risk_score": float(tx["audit_risk_score"]),
            "risk_level": str(tx["risk_level"]),
            "rule_keys": "; ".join(sorted(rule_sets.get(tx_id, frozenset()))),
            "risk_reason_text": "" if pd.isna(tx["risk_reason_text"]) else str(tx["risk_reason_text"]),
            "review_outcome": "",
            "evidence_reference": "",
            "reviewer_notes": "",
        })

    queue = pd.DataFrame(rows)
    population_rules = set(alerts["rule_key"].unique())
    covered = sorted(set().union(*(rule_sets.get(tx_id, frozenset()) for tx_id in selected_ids)))
    summary = {
        "method": "targeted rule coverage and risk ranking, then simple random sample of the remainder",
        "population_count": len(ledger),
        "budget": policy.budget,
        "seed": policy.seed,
        "random_share_requested": policy.random_share,
        "selected_by_route": {route: int((queue["selection_route"] == route).sum()) for route in ROUTES},
        "random_frame_count": len(random_frame),
        "random_inclusion_probability": inclusion_probability,
        "population_rule_keys": sorted(population_rules),
        "selected_rule_keys": covered,
        "uncovered_rule_keys": sorted(population_rules - set(covered)),
        "selected_amount_cny": round(float(queue["debit_amount"].sum()), 2),
        "selected_high_or_critical": int(queue["risk_level"].isin(["High", "Critical"]).sum()),
        "source_sha256": selection_fingerprint(transactions, alerts),
        "interpretation": (
            "Targeted cases are judgmental selections. The random probability and weight "
            "apply only to the non-targeted frame; no exception rate is inferred before review."
        ),
    }
    return ReviewPlan(queue=queue, summary=summary)


def write_review_plan(
    plan: ReviewPlan,
    queue_path: Path = REVIEW_PLAN_CSV,
    summary_path: Path = REVIEW_PLAN_SUMMARY_JSON,
) -> dict[str, Any]:
    """Write the auditor queue and a checksum-bound selection record."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    contents = plan.queue.to_csv(index=False, lineterminator="\n", float_format="%.10g")
    summary = {**plan.summary, "queue_sha256": hashlib.sha256(contents.encode()).hexdigest()}
    queue_path.write_bytes(contents.encode("utf-8"))
    save_json(summary, summary_path)
    LOGGER.info("Review plan: %s vouchers, %s random controls", len(plan.queue),
                summary["selected_by_route"]["random_control"])
    return summary


def benchmark_review_plan(plan: ReviewPlan, transactions: pd.DataFrame) -> dict[str, Any]:
    """Grade a *completed selection* on synthetic labels, never select with them."""
    if "anomaly_label" not in transactions.columns:
        raise ValueError("synthetic benchmark requires an anomaly_label column")
    labels = pd.Series(
        as_flag_series(transactions["anomaly_label"]).to_numpy(dtype=int),
        index=transactions["transaction_id"],
    )
    by_route = {
        route: {
            "reviewed": int((plan.queue["selection_route"] == route).sum()),
            "injected_anomalies_found": int(labels.reindex(
                plan.queue.loc[plan.queue["selection_route"] == route, "transaction_id"]
            ).sum()),
        }
        for route in ROUTES
    }
    risk_only = transactions.sort_values(
        ["audit_risk_score", "debit_amount", "transaction_id"],
        ascending=[False, False, True], kind="stable",
    ).head(len(plan.queue))
    return {
        "scope": "synthetic injected labels only; unavailable on a live audit",
        "budget": len(plan.queue),
        "injected_anomalies_in_population": int(labels.sum()),
        "plan_injected_anomalies_found": sum(item["injected_anomalies_found"] for item in by_route.values()),
        "risk_only_same_budget_found": int(labels.reindex(risk_only["transaction_id"]).sum()),
        "by_route": by_route,
        "interpretation": (
            "The control sample consumes review capacity and may lower benchmark hit yield. "
            "It is reserved to assess the non-targeted population after human review, "
            "not to maximize this synthetic score."
        ),
    }


def run_review_plan(
    transactions: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
    policy: ReviewPolicy = ReviewPolicy(),
) -> ReviewPlan:
    ensure_directories()
    if transactions is None:
        transactions = load_dataframe(SCORED_TRANSACTIONS)
    if alerts is None:
        alerts = pd.read_csv(RULE_ALERTS_CSV)
    plan = build_review_plan(transactions, alerts, policy)
    plan.summary = write_review_plan(plan)
    if "anomaly_label" in transactions.columns:
        plan.benchmark = benchmark_review_plan(plan, transactions)
        save_json(plan.benchmark, REVIEW_PLAN_BENCHMARK_JSON)
    else:
        # A stale synthetic benchmark must not accompany an unlabeled client run.
        REVIEW_PLAN_BENCHMARK_JSON.unlink(missing_ok=True)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible audit review workpaper.")
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--random-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    run_review_plan(policy=ReviewPolicy(args.budget, args.random_share, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
