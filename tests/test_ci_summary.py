"""Tests for the CI job-summary helper.

``tools/ci_summary.py`` reads ``outputs/reports/audit_summary.json``, which is written
by the pipeline. The failure mode worth guarding is not a wrong number - the numbers
come straight from the report - but a *schema drift*: if the summary's key layout
changes, the job summary should degrade to a dash rather than crash the build. A CI
step that dies while reporting success is worse than one that reports nothing.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ci_summary import build_summary, main  # noqa: E402  (needs sys.path above)

#: A miniature of the real report, trimmed to the keys the summary reads.
SAMPLE: dict = {
    "population": {"total_transactions": 30_128},
    "triage": {"flagged_by_rules": 2_208, "flagged_by_rules_pct": 7.329, "high_or_critical": 255},
    "rules": {"total_alerts": 2_475},
    "machine_learning": {
        "roc_auc": 0.8162,
        "anomaly_overlap": {
            "injected_anomalies": 919,
            "caught_by_rule": 902,
            "caught_by_model_only": 2,
        },
    },
    "benford": {"first_digit": {"mad": 0.002182}},
}


@pytest.fixture
def workdir() -> Iterator[Path]:
    """A scratch directory that works in this environment.

    Deliberately **not** pytest's ``tmp_path``. The sandboxed shell this project is
    developed in intercepts pytest's own temp-directory creation and raises
    ``PermissionError: EEXIST``, which makes ``tmp_path`` unusable here - the same
    constraint that shaped the dashboard render tests. ``tempfile.mkdtemp`` goes
    through ``os.mkdir`` and is not intercepted.
    """
    path = Path(tempfile.mkdtemp(prefix="auditlens-ci-summary-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestBuildSummary:
    """The renderer is pure, so it can be checked without running the pipeline."""

    def test_it_renders_every_headline_measure(self) -> None:
        text = build_summary(SAMPLE)

        for expected in (
            "30,128",
            "2,208 (7.329%)",
            "255",
            "2,475",
            "902 of 919",
            "0.8162",
            "0.002182",
        ):
            assert expected in text, f"{expected!r} missing from the summary"

    def test_it_is_valid_markdown_with_a_header_row(self) -> None:
        lines = build_summary(SAMPLE).splitlines()

        assert lines[0] == "## AuditLens pipeline"
        header = lines.index("| Measure | Value |")
        assert lines[header + 1] == "|---|---|"
        # Every row after the separator must be a two-cell table row.
        for row in lines[header + 2 :]:
            if not row.startswith("|"):
                break
            assert row.count("|") == 3, row

    def test_it_carries_the_benchmark_caveat(self) -> None:
        """The summary travels without the README, so the caveat has to travel with it.

        A bare table saying "the rules catch 902 of 919" invites exactly the wrong
        conclusion, which is the misreading this project spends most of its prose
        correcting.
        """
        text = build_summary(SAMPLE)

        assert "benchmark was generated" in text
        assert "not evidence that the model is unnecessary" in text

    def test_a_missing_key_degrades_to_a_dash_rather_than_raising(self) -> None:
        """Schema drift must not break the build while reporting on it."""
        text = build_summary({"population": {"total_transactions": 10}})

        assert "10" in text
        assert text.count("-") > 1
        assert "ROC-AUC" in text

    def test_an_empty_summary_still_renders(self) -> None:
        text = build_summary({})

        assert text.startswith("## AuditLens pipeline")
        assert "| Measure | Value |" in text

    def test_missing_counters_read_as_zero_not_a_dash(self) -> None:
        """Counts are formatted with ``:,``, so a missing one must not raise."""
        text = build_summary({})

        assert "| Vouchers | 0 |" in text


class TestMain:
    """The CLI is what the workflow actually calls."""

    def test_it_fails_clearly_when_the_pipeline_has_not_run(self, workdir: Path) -> None:
        missing = workdir / "audit_summary.json"

        assert main(["--summary", str(missing)]) == 1

    def test_it_writes_the_summary_when_asked(self, workdir: Path) -> None:
        source = workdir / "audit_summary.json"
        source.write_text(json.dumps(SAMPLE))
        destination = workdir / "summary.md"

        assert main(["--summary", str(source), "--output", str(destination)]) == 0
        assert destination.read_text().startswith("## AuditLens pipeline")

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "outputs" / "reports" / "audit_summary.json").exists(),
        reason="Run `python src/run_pipeline.py` first.",
    )
    def test_it_reads_the_real_pipeline_output(self) -> None:
        """The end-to-end check: the real report must render without special-casing."""
        text = build_summary(
            json.loads((PROJECT_ROOT / "outputs" / "reports" / "audit_summary.json").read_text())
        )

        assert "30,128" in text
        assert "902 of 919" in text
