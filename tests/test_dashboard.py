"""Dashboard render tests.

A Streamlit app fails in a way that unit tests never catch: it imports cleanly,
the functions are all defined, and then a page raises inside a callback because a
column is missing or a filter returns an empty frame. These tests execute each
page for real, in-process, against the pipeline's own output, and assert that
nothing raised and that the page actually produced content.

They also pin the project's hard constraint: **the dashboard must never show the
ground truth.** The injected ``anomaly_label`` / ``anomaly_type`` columns exist so
the model evaluation can be reproduced, not so an auditor can be handed the
answers. If a page ever starts rendering them, the guard test below fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components import transaction_table_frame
from src.database import GROUND_TRUTH_COLUMNS, GROUND_TRUTH_LABEL
from src.utils import SCORED_TRANSACTIONS

#: Page modules, keyed by the label shown in the sidebar.
VIEW_MODULES: dict[str, str] = {
    "Executive Overview": "executive_overview",
    "Transaction Explorer": "transaction_explorer",
    "Audit Rules": "audit_rules",
    "Benford Analysis": "benford_analysis",
    "Machine Learning": "machine_learning",
    "Vendor Risk": "vendor_risk",
}

#: The pages need the pipeline's artefacts. Skipping is the honest response when
#: they are absent - a render test that invents its own data would not be testing
#: the thing that breaks in practice.
pytestmark = pytest.mark.skipif(
    not SCORED_TRANSACTIONS.exists(),
    reason="Run `python src/run_pipeline.py` before the dashboard render tests.",
)

#: A page takes a few seconds to run the real ledger through the filters.
PAGE_TIMEOUT = 240


def _runner_source(module_name: str) -> str:
    """Return a minimal Streamlit script that renders one page.

    ``AppTest.from_function`` re-executes only the function's own source, so a page
    that relies on module-level imports cannot be tested that way. A generated
    three-line runner is the smallest thing that exercises the real module.

    The runner is handed to ``AppTest.from_string`` rather than written to disk:
    the test then needs no writable temporary directory, which keeps it runnable
    under a read-only sandbox.
    """
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        f"from dashboard.views import {module_name}\n"
        f"{module_name}.render()\n"
    )


def _run_page(module_name: str) -> AppTest:
    """Execute one dashboard page and return the test harness."""
    app = AppTest.from_string(_runner_source(module_name), default_timeout=PAGE_TIMEOUT)
    app.run()
    return app


def _rendered_text(app: AppTest) -> str:
    """Flatten the page's visible prose into one searchable string.

    Everything a reader can see is folded in, not just markdown: headings, captions,
    the sidebar, **tab labels** and **expander labels**. Those last two matter because
    a page can put real content in them — the vendor page's three scheme views exist
    only as tab labels, and the machine-learning page's list of excluded features
    exists only as an expander label. A helper that collected only markdown would
    report those pages as missing content they plainly display.

    Whitespace is collapsed because the assertions are about wording, not about where
    an f-string happened to wrap. Without this, a phrase that straddles two source
    lines fails to match even though a reader sees it as one sentence.
    """
    parts = [block.value for block in app.markdown]
    parts += [block.value for block in app.caption]
    parts += [block.value for block in app.subheader]
    parts += [block.value for block in app.title]
    parts += [block.value for block in app.sidebar.markdown]
    parts += [tab.label for tab in app.tabs]
    parts += [expander.label for expander in app.expander]
    return re.sub(r"\s+", " ", "\n".join(str(part) for part in parts))


@pytest.fixture(scope="module")
def rendered_pages() -> dict[str, AppTest]:
    """Render every page once and share the result across the assertions.

    Rendering is expensive - each page pulls 30,000 vouchers through the filters
    and builds its charts - so the results are shared rather than recomputed per
    test.
    """
    return {label: _run_page(module) for label, module in VIEW_MODULES.items()}


@pytest.fixture(scope="module")
def rendered_app() -> AppTest:
    """Render the navigation entry point once."""
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=PAGE_TIMEOUT)
    app.run()
    return app


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #
class TestAppRouter:
    """The entry point itself, including the navigation shell."""

    def test_app_runs_without_raising(self, rendered_app: AppTest) -> None:
        assert not rendered_app.exception, [
            f"{type(exception.value).__name__}: {exception.value}"
            for exception in rendered_app.exception
        ]

    def test_default_page_is_the_executive_overview(self, rendered_app: AppTest) -> None:
        assert "Executive Overview" in _rendered_text(rendered_app)

    def test_sidebar_carries_the_branding_and_data_provenance(self, rendered_app: AppTest) -> None:
        sidebar = "\n".join(str(block.value) for block in rendered_app.sidebar.markdown)

        assert "AuditLens" in sidebar
        assert "synthetic" in sidebar.lower()
        assert "auditlens.db" in sidebar

    def test_no_page_level_error_is_shown(self, rendered_app: AppTest) -> None:
        assert not rendered_app.error, [str(error.value) for error in rendered_app.error]


# --------------------------------------------------------------------------- #
# Every page
# --------------------------------------------------------------------------- #
class TestEveryPageRenders:
    """The core guarantee: no page raises on the pipeline's own output."""

    @pytest.mark.parametrize("label", list(VIEW_MODULES))
    def test_page_raises_nothing(self, label: str, rendered_pages: dict[str, AppTest]) -> None:
        app = rendered_pages[label]

        assert not app.exception, [
            f"{type(exception.value).__name__}: {exception.value}" for exception in app.exception
        ]

    @pytest.mark.parametrize("label", list(VIEW_MODULES))
    def test_page_shows_no_error_banner(self, label: str, rendered_pages: dict[str, AppTest]) -> None:
        app = rendered_pages[label]

        assert not app.error, [str(error.value) for error in app.error]

    @pytest.mark.parametrize("label", list(VIEW_MODULES))
    def test_page_produces_content(self, label: str, rendered_pages: dict[str, AppTest]) -> None:
        """A page that renders nothing is as broken as one that raises."""
        app = rendered_pages[label]
        text = _rendered_text(app)

        assert len(text) > 500, f"{label} produced almost no text"

    @pytest.mark.parametrize("label", list(VIEW_MODULES))
    def test_page_has_a_heading(self, label: str, rendered_pages: dict[str, AppTest]) -> None:
        app = rendered_pages[label]
        headings = [block.value for block in app.markdown if str(block.value).startswith("## ")]

        assert headings, f"{label} has no page heading"


# --------------------------------------------------------------------------- #
# Page-specific content
# --------------------------------------------------------------------------- #
class TestPageContent:
    """Each page has to actually deliver what it claims to."""

    def test_executive_overview_reports_the_population_and_the_triage(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Executive Overview"])

        assert "Vouchers tested" in text
        assert "Flagged by rules" in text
        assert "High or critical" in text

    def test_executive_overview_states_the_benford_limitation(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        """The limitation is a deliverable, not a footnote."""
        text = _rendered_text(rendered_pages["Executive Overview"])

        assert "does <b>not</b> prove the absence of fraud" in text
        assert "synthetic" in text.lower()

    def test_executive_overview_states_that_metrics_are_an_upper_bound(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Executive Overview"])

        assert "upper bound" in text

    def test_executive_overview_reports_the_overlap_at_the_anomaly_level(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        """The detector overlap must be reported on the honest basis.

        An earlier version of this page showed the flag-level overlap and labelled it
        "Would be missed by rules alone", which reads as "anomalies the rules missed"
        and overstates the model's contribution by two orders of magnitude. The page
        must now report anomalies, not flags.
        """
        text = _rendered_text(rendered_pages["Executive Overview"])

        assert "Anomalies the rules catch" in text
        assert "Missed by both" in text
        assert "Would be missed by rules alone" not in text

    def test_transaction_explorer_offers_the_risk_explanation_drilldown(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Transaction Explorer"])

        assert "Why this voucher is flagged" in text
        assert "Score composition" in text
        assert "Procedure-by-procedure result" in text

    def test_transaction_explorer_shows_a_peer_comparison(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        """A flag is meaningless without the population it sits in."""
        text = _rendered_text(rendered_pages["Transaction Explorer"])

        assert "How this voucher compares" in text

    def test_audit_rules_documents_every_procedure(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        from src.audit_rules import RULE_DEFINITIONS

        text = _rendered_text(rendered_pages["Audit Rules"])

        for definition in RULE_DEFINITIONS:
            assert definition.label in text, definition.label

    def test_audit_rules_reports_thresholds_and_rationale(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Audit Rules"])

        assert "Threshold:" in text
        assert "Why it matters:" in text

    def test_audit_rules_warns_that_high_recall_is_a_benchmark_artefact(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Audit Rules"])

        assert "consequence of the benchmark" in text

    def test_benford_page_states_that_it_is_not_a_fraud_test(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Benford Analysis"])

        assert "not a test for fraud" in text
        assert "cannot clear anyone" in text

    def test_benford_page_shows_the_disaggregation(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Benford Analysis"])

        assert "Disaggregation by account" in text
        assert "Disaggregation by business process" in text

    def test_machine_learning_page_discusses_the_review_budget(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Machine Learning"])

        assert "Vouchers reviewed" in text
        assert "Recall matters more than" in text

    def test_machine_learning_page_states_that_metrics_are_an_upper_bound(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Machine Learning"])

        assert "upper bound" in text
        # Asserted on a phrase that does not straddle an f-string line break.
        assert "leak from the answer key" in text

    def test_machine_learning_page_explains_the_excluded_features(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Machine Learning"])

        assert "Features that were deliberately excluded" in text

    def test_machine_learning_page_relates_the_model_to_the_rules(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        """The model's incremental value must not be oversold on its own page."""
        text = _rendered_text(rendered_pages["Machine Learning"])

        assert "Read these metrics against the rules" in text

    def test_vendor_risk_page_covers_the_shell_company_indicators(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Vendor Risk"])

        assert "Shared bank accounts" in text
        assert "Dormant reactivation" in text
        assert "Spend concentration" in text

    def test_vendor_risk_page_shows_the_weighted_breakdown(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        text = _rendered_text(rendered_pages["Vendor Risk"])

        assert "Why this vendor scores what it does" in text


# --------------------------------------------------------------------------- #
# Hard constraints
# --------------------------------------------------------------------------- #
class TestNoGroundTruthLeak:
    """The dashboard must never reveal the injected answer key."""

    #: Column names that only exist because the anomalies were injected. Taken from
    #: the warehouse module rather than restated, so adding a new ground-truth
    #: column there is automatically covered by the guard below.
    FORBIDDEN: tuple[str, ...] = GROUND_TRUTH_COLUMNS

    @pytest.mark.parametrize("label", list(VIEW_MODULES))
    def test_ground_truth_columns_are_never_rendered(
        self, label: str, rendered_pages: dict[str, AppTest]
    ) -> None:
        app = rendered_pages[label]

        rendered_columns: set[str] = set()
        for frame in app.dataframe:
            value = frame.value
            if hasattr(value, "columns"):
                rendered_columns.update(str(column) for column in value.columns)
            elif hasattr(value, "data") and isinstance(value.data, dict):
                rendered_columns.update(str(column) for column in value.data)

        leaked = rendered_columns & set(self.FORBIDDEN)
        assert not leaked, f"{label} renders ground-truth column(s): {sorted(leaked)}"

    @pytest.mark.parametrize("label", list(VIEW_MODULES))
    def test_no_transaction_level_label_is_rendered_in_text(
        self, label: str, rendered_pages: dict[str, AppTest]
    ) -> None:
        """A voucher-by-voucher answer key would defeat the whole exercise."""
        text = _rendered_text(rendered_pages[label]).lower()

        # The model page may *name* the columns in its methodology note - that is
        # disclosure of method, not disclosure of the answer - so only the raw
        # injection vocabulary is forbidden.
        for token in ("duplicate_payment,", "weekend_posting,", "large_round_amount,"):
            assert token not in text, f"{label} leaks injection vocabulary: {token}"

    def test_the_model_page_discloses_that_it_uses_labels_for_grading(
        self, rendered_pages: dict[str, AppTest]
    ) -> None:
        """Using the labels to grade the model is legitimate - hiding that is not."""
        text = _rendered_text(rendered_pages["Machine Learning"])

        assert "anomaly_label" in text
        assert "only after the fact" in text


class TestTransactionGridColumns:
    """The voucher grid is the one an auditor reads row by row.

    The render tests above prove that no page *currently* shows the answer key. These
    tests are the cheaper, more durable half: they pin the grid builder itself, so a
    column added to the pipeline cannot reach the grid by default. ``transaction_table``
    lists its columns explicitly rather than slicing ``frame.columns``, and that
    property is what makes the guarantee hold for columns nobody has thought of yet.
    """

    @staticmethod
    def _ledger(**extra: object) -> pd.DataFrame:
        """A minimal scored ledger, plus whatever extra columns a test wants."""
        base = {
            "transaction_id": ["T1", "T2"],
            "transaction_date": pd.to_datetime(["2025-03-01", "2025-03-02"]),
            "account_code": ["6601", "6602"],
            "account_name": ["Expense", "Expense"],
            "vendor_name": ["Acme", None],
            "department": ["Finance", None],
            "debit_amount": [1000.0, 2500.0],
            "rule_alert_count": [2, 0],
            "anomaly_score": [0.71, 0.12],
            "audit_risk_score": [88.5, 12.0],
            "risk_level": ["Critical", "Low"],
        }
        return pd.DataFrame({**base, **extra})

    def test_the_grid_excludes_every_ground_truth_column(self) -> None:
        frame = self._ledger(anomaly_label=["1", "0"], anomaly_type=["self_approval", None])

        columns = set(transaction_table_frame(frame).columns)

        assert not columns & set(GROUND_TRUTH_COLUMNS), (
            "the voucher grid rendered a ground-truth column"
        )

    def test_the_grid_ignores_columns_it_does_not_know_about(self) -> None:
        """An explicit column list, not ``frame.columns``.

        This is the property that makes the guard durable: a future pipeline column
        cannot appear in the grid without someone deliberately adding it.
        """
        frame = self._ledger(anomaly_label=["1", "0"], injected_by="seed_42", leak_me="oops")

        columns = set(transaction_table_frame(frame).columns)

        assert "injected_by" not in columns
        assert "leak_me" not in columns

    def test_the_grid_renders_the_expected_audit_columns(self) -> None:
        """Guard against the whitelist silently shrinking."""
        columns = list(transaction_table_frame(self._ledger()).columns)

        assert columns == [
            "Transaction",
            "Date",
            "Account",
            "Vendor",
            "Department",
            "Amount (CNY)",
            "Alerts",
            "ML score",
            "Risk score",
            "Risk",
        ]

    def test_the_ground_truth_constant_has_a_single_definition(self) -> None:
        """The dashboard must import the tuple, not restate it.

        Three copies of the same list is how one of them goes stale: a new
        ground-truth column would be added to the warehouse module and silently not
        guarded anywhere else.
        """
        from dashboard import common

        assert common.GROUND_TRUTH_COLUMNS is GROUND_TRUTH_COLUMNS
        assert GROUND_TRUTH_LABEL in GROUND_TRUTH_COLUMNS
