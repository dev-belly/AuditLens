"""The README's numbers, checked against the reports they claim to come from.

For a portfolio repository the numbers *are* the credibility. A reviewer who checks one
headline figure and finds it stale stops trusting the rest, and the whole "every claim is
verified" pitch collapses. Nothing else in the suite reads the README's prose, so before
this file the numbers were correct only for as long as nobody re-ran the pipeline with a
different seed or threshold.

The counts are deliberately rounded in the README (30,128; 0.94%; CNY 267.0M), so the
comparisons allow the rounding the author intended rather than demanding exact equality:

* counts and degrees of freedom - exact
* percentages - within 0.01 of a point
* money - within 1% relative, because a figure quoted to two decimals in billions is
  only good to about that
* probabilities and scores - within 0.001

A failure here means the README and the pipeline disagree. One of them is wrong, and
which one is a judgement call the author has to make - but it must not go unnoticed.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
SUMMARY = PROJECT_ROOT / "outputs" / "reports" / "audit_summary.json"
RULE_EVALUATION = PROJECT_ROOT / "outputs" / "reports" / "rule_evaluation.csv"
CHARTS = PROJECT_ROOT / "outputs" / "charts"
WAREHOUSE = PROJECT_ROOT / "data" / "auditlens.db"

pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(),
    reason="Run `python src/run_pipeline.py` before the README claim tests.",
)

#: Tolerances. See the module docstring for why these are not zero.
PCT_TOLERANCE = 0.01
MONEY_RELATIVE_TOLERANCE = 0.01
SCORE_TOLERANCE = 0.001


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _cell(label: str) -> str:
    """Return the value cell of a two-column README row, e.g. ``| Population | 30,128 |``.

    Returns an empty string when the row is absent, so a deleted row surfaces as a
    clear assertion failure rather than a regex error.
    """
    match = re.search(rf"^\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|", _readme(), re.MULTILINE)
    return match.group(1) if match else ""


def _number(text: str) -> float | None:
    """Pull the first number out of a markdown cell, ignoring ``**`` and ``,``."""
    match = re.search(r"-?[\d,]+(?:\.\d+)?", text.replace("**", ""))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _money(text: str) -> float | None:
    """Parse the currency figure out of a markdown cell.

    Anchored on the currency token rather than on the first number, because several
    cells lead with a count: ``**30,128** clean vouchers, CNY **4,443,451,722**``.
    Returns ``None`` when the cell quotes no amount at all.
    """
    match = re.search(r"(?:CNY|¥)\s*\**\s*([\d,]+(?:\.\d+)?)\s*([BMK]?)", text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value * {"B": 1e9, "M": 1e6, "K": 1e3, "": 1.0}[match.group(2)]


def _close(actual: float | None, expected: float, *, relative: float = 0.0, absolute: float = 0.0) -> bool:
    if actual is None:
        return False
    if relative:
        return abs(actual - expected) <= abs(expected) * relative
    return abs(actual - expected) <= absolute


class TestHeadlineTable:
    """The table a reviewer reads first."""

    def test_the_headline_rows_are_present(self) -> None:
        for label in ("Population", "Raw extract", "Injected anomalies", "Flagged by rules"):
            assert _cell(label), f"the README no longer has a '{label}' row"

    def test_population(self) -> None:
        summary = _summary()
        assert _close(_number(_cell("Population")), summary["population"]["total_transactions"], absolute=0.5)

    def test_population_value(self) -> None:
        summary = _summary()
        assert _close(_money(_cell("Population")), summary["population"]["total_amount"], relative=MONEY_RELATIVE_TOLERANCE)

    def test_raw_extract_and_issue_count(self) -> None:
        quality = _summary()["data_quality"]
        cell = _cell("Raw extract")
        assert _close(_number(cell), quality["raw_rows"], absolute=0.5), f"raw rows: {cell}"
        # the issue count is the second number in the cell
        numbers = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+(?:\.\d+)?", cell.replace("**", ""))]
        assert quality["total_issues"] in numbers, f"issue count missing from {cell!r}"

    def test_injected_anomalies(self) -> None:
        overlap = _summary()["machine_learning"]["anomaly_overlap"]
        assert _close(_number(_cell("Injected anomalies")), overlap["injected_anomalies"], absolute=0.5)

    def test_flagged_by_rules(self) -> None:
        triage = _summary()["triage"]
        assert _close(_number(_cell("Flagged by rules")), triage["flagged_by_rules"], absolute=0.5)

    def test_high_or_critical(self) -> None:
        triage = _summary()["triage"]
        assert _close(_number(_cell("High or Critical risk")), triage["high_or_critical"], absolute=0.5)

    def test_anomalies_caught_by_the_rules(self) -> None:
        overlap = _summary()["machine_learning"]["anomaly_overlap"]
        cell = _cell("Anomalies caught by the nine rules")
        assert _close(_number(cell), overlap["caught_by_rule"], absolute=0.5), cell

    def test_anomalies_caught_by_the_model_only(self) -> None:
        """The number that was once overstated by 100x by counting flags as findings."""
        overlap = _summary()["machine_learning"]["anomaly_overlap"]
        assert _close(_number(_cell("Anomalies caught by the model and no rule")), overlap["caught_by_model_only"], absolute=0.5)

    def test_anomalies_caught_by_neither(self) -> None:
        overlap = _summary()["machine_learning"]["anomaly_overlap"]
        assert _close(_number(_cell("Anomalies caught by neither")), overlap["caught_by_neither"], absolute=0.5)

    def test_isolation_forest_scores(self) -> None:
        ml = _summary()["machine_learning"]
        cell = _cell("Isolation Forest")
        scores = [float(n) for n in re.findall(r"0\.\d+", cell)]
        assert scores, cell
        assert any(abs(s - ml["roc_auc"]) < SCORE_TOLERANCE for s in scores), f"ROC-AUC missing from {cell!r}"

    def test_benford_first_digit_mad(self) -> None:
        mad = _summary()["benford"]["first_digit"]["mad"]
        assert _close(_number(_cell("Benford first-digit MAD")), mad, absolute=1e-4)


class TestRuleTable:
    """The nine procedures, against ``rule_evaluation.csv``."""

    @staticmethod
    def _rows() -> dict[str, tuple[int, float, float]]:
        """The evaluated rules, keyed by the label the README uses."""
        with RULE_EVALUATION.open(encoding="utf-8-sig") as handle:
            return {
                row["rule_label"]: (int(row["flagged"]), float(row["precision"]), float(row["recall"]))
                for row in csv.DictReader(handle)
            }

    def test_the_evaluation_file_has_every_rule(self) -> None:
        assert len(self._rows()) == 9

    def test_every_rule_appears_in_the_readme(self) -> None:
        readme = _readme()
        missing = [label for label in self._rows() if f"| {label} |" not in readme]
        assert not missing, f"rules absent from the README table: {missing}"

    def test_flagged_counts_and_metrics_match(self) -> None:
        """Each row is ``| Rule | definition | weight | flagged | precision | recall |``."""
        readme = _readme()
        mismatches: list[str] = []
        for label, (flagged, precision, recall) in self._rows().items():
            match = re.search(rf"^\|\s*{re.escape(label)}\s*\|(.+?)\|\s*$", readme, re.MULTILINE)
            if not match:
                mismatches.append(f"{label}: row not found")
                continue
            cells = [c.strip().replace("**", "") for c in match.group(1).split("|")]
            numbers = [float(c) for c in cells if re.fullmatch(r"[\d,]+(?:\.\d+)?", c.replace(",", ""))]
            if not any(abs(n - flagged) < 0.5 for n in numbers):
                mismatches.append(f"{label}: flagged {flagged} not in row")
            if not any(abs(n - precision) < SCORE_TOLERANCE for n in numbers):
                mismatches.append(f"{label}: precision {precision} not in row")
            if not any(abs(n - recall) < SCORE_TOLERANCE for n in numbers):
                mismatches.append(f"{label}: recall {recall} not in row")
        assert not mismatches, "; ".join(mismatches)


class TestRiskScore:
    """The weights are a design decision, so they must sum to one."""

    def test_the_component_weights_sum_to_one(self) -> None:
        weights = [
            float(w)
            for w in re.findall(
                r"^\|\s*(?:Rule indicators|Isolation Forest|Vendor risk|Amount context|Statistical \(Benford\))\s*\|"
                r"\s*\*\*([\d.]+)\*\*",
                _readme(),
                re.MULTILINE,
            )
        ]
        assert len(weights) == 5, f"expected five components, parsed {weights}"
        assert abs(sum(weights) - 1.0) < 1e-9, f"weights sum to {sum(weights)}, not 1.0"

    def test_the_band_table_matches_the_reports(self) -> None:
        triage = _summary()["triage"]
        readme = _readme()
        mismatches: list[str] = []
        for band in ("Low", "Medium", "High", "Critical"):
            match = re.search(
                rf"^\|\s*{band}\s*\|[^|]*\|[^|]*\|\s*([\d,]+)\s*\|\s*(CNY|¥)\s*([\d.]+)\s*([BMK]?)\s*\|",
                readme,
                re.MULTILINE,
            )
            if not match:
                mismatches.append(f"{band}: row not parsed")
                continue
            count = int(match.group(1).replace(",", ""))
            value = float(match.group(3)) * {"B": 1e9, "M": 1e6, "K": 1e3, "": 1.0}[match.group(4)]
            if count != triage["risk_level_counts"][band]:
                mismatches.append(f"{band}: README {count} vs report {triage['risk_level_counts'][band]}")
            if not _close(value, triage["risk_level_amounts"][band], relative=MONEY_RELATIVE_TOLERANCE):
                mismatches.append(f"{band}: README value {value:,.0f} vs report {triage['risk_level_amounts'][band]:,.0f}")
        assert not mismatches, "; ".join(mismatches)


class TestArtefactCounts:
    """Counts that describe committed artefacts."""

    def test_the_stated_chart_count_is_right(self) -> None:
        match = re.search(r"`outputs/charts/`\s*\|\s*(\d+)\s+charts", _readme())
        assert match, "the README no longer states a chart count"
        assert int(match.group(1)) == len(list(CHARTS.glob("*.png")))

    def test_review_workpaper_claims_match_the_committed_selection(self) -> None:
        section = _readme().split("## Audit review workpaper", 1)[1].split("## Dashboard", 1)[0]
        report_dir = PROJECT_ROOT / "outputs" / "reports"
        plan = json.loads((report_dir / "review_plan_summary.json").read_text())
        benchmark = json.loads((report_dir / "review_plan_benchmark.json").read_text())

        assert f"[{plan['budget']}-voucher workpaper]" in section
        for route in ("rule_coverage", "risk_priority", "random_control"):
            assert re.search(rf"\b{plan['selected_by_route'][route]}\b", section)
        assert f"{plan['random_frame_count']:,}" in section
        assert f"{benchmark['plan_injected_anomalies_found']} of the " in section
        assert f"{benchmark['injected_anomalies_in_population']} injected" in section
        assert f"**{benchmark['risk_only_same_budget_found']}**" in section

    def test_the_two_employee_figures_are_stated_and_distinct(self) -> None:
        """140 people are on the roster; 104 of them raise a voucher.

        Both numbers are correct and they describe different populations. The summary
        JSON once labelled the smaller one ``unique_employees``, which read as a
        contradiction of the 140-row ``employees`` table in the same set of reports.
        The README now spells the distinction out, so this pins both figures and the
        relationship between them.
        """
        roster_match = re.search(r"`employees`\s*\((\d[\d,]*)\)", _readme())
        creators_match = re.search(r"only\s+\*\*(\d[\d,]*)\*\*\s+of those people", _readme())
        assert roster_match, "the README no longer states the employees table size"
        assert creators_match, "the README no longer states the voucher-creator count"

        roster_size = int(roster_match.group(1).replace(",", ""))
        creator_count = int(creators_match.group(1).replace(",", ""))

        population = _summary()["population"]
        assert population["unique_voucher_creators"] == creator_count
        assert "unique_employees" not in population, "the ambiguous field name came back"
        assert creator_count < roster_size, "voucher creators are a subset of the roster"

    def test_the_roster_size_matches_the_warehouse(self) -> None:
        """The 140 is a real table count, not a decorative figure."""
        if not WAREHOUSE.exists():
            pytest.skip("Run `python src/run_pipeline.py` to build the warehouse.")

        roster_match = re.search(r"`employees`\s*\((\d[\d,]*)\)", _readme())
        assert roster_match

        with sqlite3.connect(WAREHOUSE) as connection:
            actual = connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0]

        assert actual == int(roster_match.group(1).replace(",", ""))
