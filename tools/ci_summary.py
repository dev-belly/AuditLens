"""Write a Markdown summary of a pipeline run, for the CI job summary.

GitHub renders whatever is appended to ``$GITHUB_STEP_SUMMARY``. Putting the headline
numbers there means a reviewer can see what the pipeline produced without downloading
anything or reading the log - and, more usefully, a run whose numbers have moved is
visible at a glance rather than buried in a diff.

This lives in ``tools/`` rather than inline in the workflow so it can be run and tested
locally::

    python src/run_pipeline.py
    python tools/ci_summary.py

It prints the summary and, when ``--output`` is given, also writes it to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import AUDIT_SUMMARY_REPORT  # noqa: E402  (needs sys.path above)


def _get(mapping: dict[str, Any], path: str, default: Any = "-") -> Any:
    """Read a dotted path out of the summary, tolerating a missing key."""
    node: Any = mapping
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def build_summary(summary: dict[str, Any]) -> str:
    """Render the summary table. Pure, so it can be tested without a pipeline run."""
    rows: list[tuple[str, str]] = [
        ("Vouchers", f"{_get(summary, 'population.total_transactions', 0):,}"),
        (
            "Flagged by rules",
            f"{_get(summary, 'triage.flagged_by_rules', 0):,} "
            f"({_get(summary, 'triage.flagged_by_rules_pct', 0)}%)",
        ),
        ("High or Critical", f"{_get(summary, 'triage.high_or_critical', 0):,}"),
        ("Rule alerts raised", f"{_get(summary, 'rules.total_alerts', 0):,}"),
        (
            "Anomalies caught by the rules",
            f"{_get(summary, 'machine_learning.anomaly_overlap.caught_by_rule', 0)} "
            f"of {_get(summary, 'machine_learning.anomaly_overlap.injected_anomalies', 0)}",
        ),
        (
            "Anomalies caught by the model only",
            f"{_get(summary, 'machine_learning.anomaly_overlap.caught_by_model_only', 0)}",
        ),
        ("ROC-AUC", f"{_get(summary, 'machine_learning.roc_auc', '-')}"),
        ("Benford first-digit MAD", f"{_get(summary, 'benford.first_digit.mad', '-')}"),
    ]

    lines = ["## AuditLens pipeline", "", "| Measure | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines += [
        "",
        "_The nine rules recovering almost all injected anomalies is a property of how "
        "the benchmark was generated, not evidence that the model is unnecessary. "
        "See the README._",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--summary",
        type=Path,
        default=AUDIT_SUMMARY_REPORT,
        help="Path to audit_summary.json",
    )
    parser.add_argument("--output", type=Path, default=None, help="Also write the summary here")
    args = parser.parse_args(argv)

    if not args.summary.exists():
        print(f"No summary at {args.summary}. Run `python src/run_pipeline.py` first.")
        return 1

    text = build_summary(json.loads(args.summary.read_text()))
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
