"""End-to-end pipeline for AuditLens.

Runs every stage in order and writes all artefacts. This is the single command a
reviewer needs to reproduce the whole project::

    python src/run_pipeline.py

Stages
------
==============  ============================================================
Stage           Module
==============  ============================================================
1. Generate     :mod:`src.data_generator`
2. Clean        :mod:`src.data_cleaning`
3. Features     :mod:`src.feature_engineering`
4. Rules        :mod:`src.audit_rules`
5. Benford      :mod:`src.benford`
6. Machine      :mod:`src.anomaly_detection`
   learning
7. Risk score   :mod:`src.risk_scoring`
8. Warehouse    :mod:`src.database`
9. Reporting    :mod:`src.reporting`
10. Review plan :mod:`src.review_plan`
==============  ============================================================

Use ``--skip-generation`` to reuse an existing synthetic extract, and
``--transactions N`` to change the ledger size.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/run_pipeline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import (
    BENFORD_RESULTS_JSON,
    CHART_DIR,
    DATA_QUALITY_REPORT,
    DB_PATH,
    HIGH_RISK_CSV,
    MODEL_METRICS_JSON,
    RANDOM_SEED,
    REPORT_DIR,
    REVIEW_PLAN_CSV,
    SCORED_TRANSACTIONS,
    TRANSACTIONS_FEATURES,
    VENDOR_RISK_TABLE,
    ensure_directories,
    format_cny,
    get_logger,
)

LOGGER = get_logger("auditlens.pipeline")


def _banner(step: int, total: int, title: str) -> None:
    """Print a stage banner."""
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info("  STEP %s/%s  %s", step, total, title)
    LOGGER.info("=" * 78)


def run_pipeline(
    generate: bool = True,
    n_transactions: int = 30_000,
    seed: int = RANDOM_SEED,
    review_budget: int = 300,
    review_random_share: float = 0.2,
) -> dict[str, Any]:
    """Execute the full AuditLens pipeline.

    Args:
        generate: Whether to regenerate the synthetic dataset. When ``False`` the
            existing extract in ``data/raw`` is reused.
        n_transactions: Number of vouchers to generate.
        seed: Random seed for the generator.
        review_budget: Number of vouchers in the auditor workpaper.
        review_random_share: Share reserved for random controls from the remainder.

    Returns:
        Mapping with the key artefacts of every stage.
    """
    ensure_directories()
    started = time.perf_counter()
    total_steps = 10
    results: dict[str, Any] = {}

    # Imported here so that `--help` does not pay the import cost.
    from src import audit_rules, anomaly_detection, benford, data_cleaning, database
    from src import data_generator, feature_engineering, reporting, review_plan, risk_scoring

    # ---------------------------------------------------------------- 1. data
    if generate:
        _banner(1, total_steps, "Generate synthetic ledger")
        generator_config = data_generator.GeneratorConfig(
            n_transactions=n_transactions, seed=seed
        )
        results["raw"] = data_generator.generate_dataset(generator_config)
    else:
        LOGGER.info("Skipping generation; reusing the existing extract in data/raw.")

    # ------------------------------------------------------------- 2. cleaning
    _banner(2, total_steps, "Data quality pipeline")
    cleaning = data_cleaning.run_cleaning()
    results["cleaning"] = cleaning
    quality = cleaning["report"]
    LOGGER.info(
        "Data quality: %s raw rows -> %s clean rows, %s issues (%.3f%%)",
        f"{quality.raw_rows:,}",
        f"{quality.clean_rows:,}",
        f"{quality.total_issues:,}",
        quality.to_dict()["issue_rate_pct"],
    )

    # ------------------------------------------------------------- 3. features
    _banner(3, total_steps, "Feature engineering")
    features = feature_engineering.build_features(cleaning["transactions"], cleaning["vendors"])
    transactions, vendor_features = features
    feature_engineering.save_dataframe(transactions, TRANSACTIONS_FEATURES)
    feature_engineering.save_dataframe(vendor_features, VENDOR_RISK_TABLE)
    LOGGER.info("Feature table: %s rows x %s columns", f"{len(transactions):,}", transactions.shape[1])

    # ---------------------------------------------------------------- 4. rules
    _banner(4, total_steps, "Audit rule engine")
    rule_result = audit_rules.run_rule_engine(transactions)
    scored = rule_result["scored"]
    evaluation = rule_result["evaluation"]
    results["rules"] = rule_result
    LOGGER.info("Rule alerts: %s across %s vouchers", f"{len(rule_result['alerts']):,}", f"{len(scored):,}")

    # -------------------------------------------------------------- 5. Benford
    _banner(5, total_steps, "Benford's Law analysis")
    benford_result = benford.run_benford_analysis(scored)
    results["benford"] = benford_result
    LOGGER.info(
        "First-digit test: MAD %.4f, chi-square %.0f, %s",
        benford_result["first_digit"].mad,
        benford_result["first_digit"].chi_square,
        benford_result["first_digit"].conformity,
    )

    # ------------------------------------------------------------ 6. ML model
    _banner(6, total_steps, "Isolation Forest anomaly detection")
    ml_result = anomaly_detection.run_anomaly_detection(scored)
    scored = ml_result["transactions"]
    results["ml"] = ml_result

    # ------------------------------------------------------------ 7. risk score
    _banner(7, total_steps, "Audit Risk Score engine")
    risk_result = risk_scoring.run_risk_scoring(scored, vendor_features)
    scored = risk_result["transactions"]
    vendor_risk = risk_result["vendor_risk"]
    summary = risk_result["summary"]
    results["risk"] = risk_result
    LOGGER.info("Average Audit Risk Score: %.2f / 100", summary.average_risk_score)
    LOGGER.info(
        "High/Critical vouchers: %s of %s",
        f"{summary.high_risk_transactions:,}",
        f"{summary.total_transactions:,}",
    )

    # ------------------------------------------------------------- 8. warehouse
    _banner(8, total_steps, "SQLite warehouse")
    counts = database.load_warehouse(
        transactions=scored,
        vendor_risk=vendor_risk,
        employees=cleaning["employees"],
        alerts=rule_result["alerts"],
        summary=summary.to_dict(),
    )
    sql_results = database.run_sql_file()
    results["database"] = {"counts": counts, "queries": sql_results}

    # ------------------------------------------------------------- 9. reporting
    _banner(9, total_steps, "Charts and reports")
    report_result = reporting.run_reporting(scored, vendor_risk, rule_result["alerts"])
    results["reporting"] = report_result

    # ---------------------------------------------------------- 10. review plan
    _banner(10, total_steps, "Audit review workpaper")
    results["review_plan"] = review_plan.run_review_plan(
        scored,
        rule_result["alerts"],
        review_plan.ReviewPolicy(review_budget, review_random_share, seed),
    )

    elapsed = time.perf_counter() - started
    _print_summary(summary, evaluation, ml_result["evaluation"], sql_results, elapsed)
    return results


def _print_summary(
    summary: Any,
    rule_evaluation: pd.DataFrame,
    ml_evaluation: Any,
    sql_results: dict[str, pd.DataFrame],
    elapsed: float,
) -> None:
    """Print the closing summary an auditor would want to see."""
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info("  PIPELINE COMPLETE in %.1fs", elapsed)
    LOGGER.info("=" * 78)
    LOGGER.info("Population")
    LOGGER.info("  vouchers            %14s", f"{summary.total_transactions:,}")
    LOGGER.info("  total value         %14s", format_cny(summary.total_amount))
    LOGGER.info("  unique vendors      %14s", f"{summary.unique_vendors:,}")
    LOGGER.info("")
    LOGGER.info("Triage")
    LOGGER.info("  flagged by rules    %14s", f"{summary.flagged_transactions:,}")
    LOGGER.info("  high or critical    %14s", f"{summary.high_risk_transactions:,}")
    LOGGER.info("  average risk score  %14.2f", summary.average_risk_score)
    for level, count in summary.risk_level_counts.items():
        LOGGER.info("    %-8s          %14s", level, f"{count:,}")
    LOGGER.info("")
    LOGGER.info("Rule performance (against injected ground truth)")
    for row in rule_evaluation.itertuples(index=False):
        LOGGER.info(
            "  %-26s flagged=%6s  precision=%.3f  recall=%.3f  f1=%.3f",
            row.rule_label, f"{row.flagged:,}", row.precision, row.recall, row.f1,
        )
    if ml_evaluation is not None:
        LOGGER.info("")
        LOGGER.info("Isolation Forest")
        LOGGER.info("  precision           %14.3f", ml_evaluation.precision)
        LOGGER.info("  recall              %14.3f", ml_evaluation.recall)
        LOGGER.info("  f1                  %14.3f", ml_evaluation.f1)
        LOGGER.info("  roc_auc             %14.3f", ml_evaluation.roc_auc)
    LOGGER.info("")
    LOGGER.info("Artefacts")
    LOGGER.info("  processed data      %s", SCORED_TRANSACTIONS)
    LOGGER.info("  warehouse           %s", DB_PATH)
    LOGGER.info("  charts              %s", CHART_DIR)
    LOGGER.info("  reports             %s", REPORT_DIR)
    LOGGER.info("  review extract      %s", HIGH_RISK_CSV)
    LOGGER.info("  review workpaper    %s", REVIEW_PLAN_CSV)
    LOGGER.info("")
    LOGGER.info("SQL analytics: %s queries executed", len(sql_results))
    LOGGER.info("")
    LOGGER.info("Next step:  streamlit run dashboard/app.py")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete AuditLens pipeline.")
    parser.add_argument("--transactions", type=int, default=30_000, help="number of vouchers to generate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument("--review-budget", type=int, default=300, help="vouchers in the review workpaper")
    parser.add_argument("--review-random-share", type=float, default=0.2,
                        help="share reserved for random controls from the remainder")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="reuse the existing raw extract instead of regenerating it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    run_pipeline(
        generate=not args.skip_generation,
        n_transactions=args.transactions,
        seed=args.seed,
        review_budget=args.review_budget,
        review_random_share=args.review_random_share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
