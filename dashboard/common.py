"""Shared data access, formatting and styling for the AuditLens dashboard.

The dashboard is deliberately a *thin* layer. Every number it shows was produced
by the pipeline and persisted to the SQLite warehouse; the app's job is to let an
auditor slice that warehouse without writing SQL. Keeping the app thin means the
figures on screen and the figures in ``outputs/reports`` cannot drift apart.

Data sources, in order of preference:

1. ``data/auditlens.db`` - the SQLite warehouse. Preferred, because it is the
   artefact an audit team would actually query.
2. ``data/processed/*.parquet`` - used when the warehouse has not been built yet,
   so the dashboard still works after a partial pipeline run.

Ground truth (``anomaly_label`` / ``anomaly_type``) exists in the warehouse so the
model evaluation can be reproduced. It is not available to an auditor: no page
renders it per voucher, and the one place it is read - the benchmark diagnostic on
the Machine Learning page - is collapsed inside an expander that states in the
surrounding text that the labels would not exist on a live engagement. Everything
else the dashboard shows is conditioned only on the pipeline's own output.

The list of protected columns lives in ``src.database.GROUND_TRUTH_COLUMNS`` and is
imported here rather than restated, so there is one definition to keep correct.
``tests/test_dashboard.py`` asserts that the grid builders cannot render them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The dashboard is launched with `streamlit run dashboard/app.py`, which puts
# `dashboard/` on sys.path rather than the project root. Add the root so that
# `src` imports resolve the same way they do inside the pipeline.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.database import GROUND_TRUTH_COLUMNS, GROUND_TRUTH_LABEL, query
from src.utils import (
    AUDIT_SUMMARY_REPORT,
    BENFORD_RESULTS_JSON,
    DATA_QUALITY_REPORT,
    DB_PATH,
    MODEL_METRICS_JSON,
    RISK_COLORS,
    RISK_LEVEL_ORDER,
    RULE_EVALUATION_CSV,
    SCORED_TRANSACTIONS,
    VENDOR_RISK_TABLE,
    as_reason_list,
    ensure_directories,
    format_cny,
    load_dataframe,
    load_json,
)

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
#: Audit-report palette. Muted navy reads as "professional services" rather than
#: "startup dashboard", which matters when the audience is an audit partner.
THEME = {
    "primary": "#1F4E79",
    "accent": "#3C8DBC",
    "muted": "#8A9BA8",
    "surface": "#FFFFFF",
    "canvas": "#F7F9FB",
    "border": "#DCE3E8",
    "text": "#1B2A38",
}

#: Colours used consistently for the four risk bands, everywhere in the app.
RISK_BAND_COLORS: dict[str, str] = {
    "Low": "#2E8B57",
    "Medium": "#D9A441",
    "High": "#D2691E",
    "Critical": "#B22222",
}

#: Ground-truth columns, imported from the warehouse module so there is exactly one
#: definition. See ``src.database.GROUND_TRUTH_COLUMNS`` for why they are kept at
#: all. Anything that builds a table for an auditor must exclude these; the grid
#: builders in ``components.py`` are tested against this tuple.


def inject_css() -> None:
    """Apply a small, consistent stylesheet to the app.

    Streamlit's defaults are quite loud for a finance audience, so this trims the
    chrome: tighter headings, quieter captions, and KPI cards that look like a
    printed audit summary rather than a marketing page.
    """
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
        h1 {{ font-size: 1.85rem !important; color: {THEME['text']}; letter-spacing: -0.01em; }}
        h2 {{ font-size: 1.30rem !important; color: {THEME['primary']}; margin-top: 1.6rem !important; }}
        h3 {{ font-size: 1.05rem !important; color: {THEME['text']}; margin-top: 1.2rem !important; }}
        .al-caption {{ color: {THEME['muted']}; font-size: 0.86rem; line-height: 1.55; }}
        .al-kpi {{
            background: {THEME['surface']}; border: 1px solid {THEME['border']};
            border-left: 4px solid {THEME['primary']}; border-radius: 6px;
            padding: 0.85rem 1rem 0.75rem 1rem; height: 100%;
        }}
        .al-kpi-label {{
            color: {THEME['muted']}; font-size: 0.72rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.30rem;
        }}
        .al-kpi-value {{ color: {THEME['text']}; font-size: 1.55rem; font-weight: 700; line-height: 1.15; }}
        .al-kpi-delta {{ color: {THEME['muted']}; font-size: 0.78rem; margin-top: 0.25rem; }}
        .al-badge {{
            display: inline-block; padding: 0.12rem 0.55rem; border-radius: 10px;
            font-size: 0.74rem; font-weight: 600; color: #FFFFFF; letter-spacing: 0.02em;
        }}
        .al-note {{
            background: {THEME['canvas']}; border: 1px solid {THEME['border']};
            border-radius: 6px; padding: 0.75rem 0.95rem; font-size: 0.85rem;
            color: {THEME['text']}; line-height: 1.6;
        }}
        .al-note-warn {{ border-left: 4px solid {RISK_BAND_COLORS['High']}; }}
        .al-note-info {{ border-left: 4px solid {THEME['accent']}; }}
        .al-rule {{ background: {THEME['surface']}; border: 1px solid {THEME['border']};
            border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }}
        .al-rule-code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.78rem; color: {THEME['accent']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def warehouse_ready() -> bool:
    """Return ``True`` only for a usable, populated SQLite warehouse.

    A failed or interrupted build can leave a ``transactions`` table with zero
    rows. In that case the dashboard should use the scored parquet file instead
    of presenting an empty engagement as a successful one.
    """
    if not DB_PATH.exists():
        return False
    try:
        tables = query("SELECT name FROM sqlite_master WHERE type='table'")
        if not {"transactions", "vendors", "employees", "audit_alerts"}.issubset(
            set(tables["name"])
        ):
            return False
        return int(query("SELECT COUNT(*) AS n FROM transactions")["n"].iloc[0]) > 0
    except Exception:
        return False


def _load_transactions() -> pd.DataFrame:
    """Load the scored ledger, preferring the warehouse over the parquet file."""
    if warehouse_ready():
        frame = query(
            """
            SELECT *
            FROM transactions
            ORDER BY audit_risk_score DESC, transaction_date
            """
        )
    else:
        frame = load_dataframe(SCORED_TRANSACTIONS)

    # Normalise the columns the dashboard filters on. The warehouse stores dates
    # as ISO text and booleans as integers, so both paths converge here.
    for column in ("transaction_date", "posting_date", "approval_time", "invoice_time", "payment_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    for column in ("rule_alert_count", "audit_risk_score", "anomaly_score", "debit_amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Rule flags arrive as 0/1 integers from SQLite and as int8 from parquet.
    for column in frame.columns:
        if column.endswith("_flag") or column.startswith("is_"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(bool)

    # `risk_reason_text` is a "; "-joined string; split it back into a list so the
    # explorer can render one bullet per reason.
    if "risk_reason_text" in frame.columns:
        frame["risk_reasons"] = frame["risk_reason_text"].map(as_reason_list)
    elif "risk_reasons" not in frame.columns:
        frame["risk_reasons"] = [[] for _ in range(len(frame))]

    return frame


@st.cache_data(show_spinner="Loading scored ledger from the warehouse...")
def load_transactions() -> pd.DataFrame:
    """Cached scored transaction table."""
    return _load_transactions()


@st.cache_data(show_spinner=False)
def load_vendors() -> pd.DataFrame:
    """Cached vendor master data joined to the aggregated vendor risk score."""
    if warehouse_ready():
        try:
            frame = query("SELECT * FROM vendors")
        except Exception:
            frame = load_dataframe(VENDOR_RISK_TABLE)
    else:
        frame = load_dataframe(VENDOR_RISK_TABLE)

    for column in ("registration_date",):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("vendor_risk_score", "alert_count", "vendor_total_amount", "vendor_transaction_count"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "shared_bank_account" in frame.columns:
        frame["shared_bank_account"] = pd.to_numeric(
            frame["shared_bank_account"], errors="coerce"
        ).fillna(0).astype(bool)
    return frame


@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    """Cached long-format rule alerts (one row per voucher/rule pair)."""
    if warehouse_ready():
        try:
            return query("SELECT * FROM audit_alerts")
        except Exception:
            pass
    from src.utils import RULE_ALERTS_CSV

    if RULE_ALERTS_CSV.exists():
        frame = pd.read_csv(RULE_ALERTS_CSV)
        frame.insert(0, "alert_id", range(1, len(frame) + 1))
        return frame
    return pd.DataFrame(
        columns=["alert_id", "transaction_id", "rule_key", "rule_label", "rule_score", "risk_reason"]
    )


@st.cache_data(show_spinner=False)
def load_employees() -> pd.DataFrame:
    """Cached employee master data."""
    if warehouse_ready():
        try:
            return query("SELECT * FROM employees")
        except Exception:
            pass
    from src.utils import EMPLOYEES_CLEAN

    return load_dataframe(EMPLOYEES_CLEAN)


@st.cache_data(show_spinner=False)
def load_rule_evaluation() -> pd.DataFrame:
    """Cached per-rule precision/recall against the labelled benchmark.

    This is the one place the dashboard reads a ground-truth-derived artefact, and
    it does so only to report how well each procedure performed - never to filter
    or rank what an auditor sees.
    """
    if RULE_EVALUATION_CSV.exists():
        return pd.read_csv(RULE_EVALUATION_CSV)
    return pd.DataFrame(
        columns=[
            "rule_key", "rule_label", "flagged", "true_positives",
            "false_positives", "precision", "recall", "f1",
        ]
    )


@st.cache_data(show_spinner=False)
def load_summary() -> dict[str, Any]:
    """Cached engagement summary produced by the reporting stage."""
    return load_json(AUDIT_SUMMARY_REPORT) if AUDIT_SUMMARY_REPORT.exists() else {}


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    """Cached Isolation Forest evaluation metrics."""
    return load_json(MODEL_METRICS_JSON) if MODEL_METRICS_JSON.exists() else {}


@st.cache_data(show_spinner=False)
def load_benford() -> dict[str, Any]:
    """Cached Benford's Law test results."""
    return load_json(BENFORD_RESULTS_JSON) if BENFORD_RESULTS_JSON.exists() else {}


@st.cache_data(show_spinner=False)
def load_quality() -> dict[str, Any]:
    """Cached data quality report."""
    return load_json(DATA_QUALITY_REPORT) if DATA_QUALITY_REPORT.exists() else {}


def pipeline_available() -> bool:
    """Return ``True`` when at least one pipeline artefact exists."""
    return warehouse_ready() or SCORED_TRANSACTIONS.exists()


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
def sidebar_filters(frame: pd.DataFrame, *, show_dates: bool = True) -> dict[str, Any]:
    """Render the shared filter panel and return the selected values.

    Every page that lists transactions uses this, so a filter set on one page
    behaves identically on the next. That consistency is the difference between a
    dashboard an auditor trusts and one they fight with.

    Args:
        frame: The scored transaction table (used to populate the option lists).
        show_dates: Whether to offer the date range control.

    Returns:
        A dictionary of filter selections.
    """
    st.sidebar.markdown("### Filters")
    selections: dict[str, Any] = {}

    if show_dates and "transaction_date" in frame.columns and frame["transaction_date"].notna().any():
        low = frame["transaction_date"].min().date()
        high = frame["transaction_date"].max().date()
        picked = st.sidebar.date_input(
            "Transaction date range",
            value=(low, high),
            min_value=low,
            max_value=high,
            help="Posting period covered by the engagement.",
        )
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            selections["date_range"] = (pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + pd.Timedelta(days=1))
        else:
            selections["date_range"] = (pd.Timestamp(low), pd.Timestamp(high) + pd.Timedelta(days=1))

    levels = [level for level in RISK_LEVEL_ORDER if level in set(frame.get("risk_level", pd.Series(dtype=str)))]
    selections["risk_levels"] = st.sidebar.multiselect(
        "Risk level",
        options=levels,
        default=levels,
        help="Audit Risk Score bands: Low 0-30, Medium 30-60, High 60-80, Critical 80-100.",
    )

    if "process" in frame.columns:
        processes = sorted(frame["process"].dropna().unique().tolist())
        selections["processes"] = st.sidebar.multiselect("Business process", options=processes, default=processes)

    if "department" in frame.columns:
        departments = sorted(frame["department"].dropna().unique().tolist())
        selections["departments"] = st.sidebar.multiselect(
            "Department", options=departments, default=departments
        )

    if "currency" in frame.columns:
        currencies = sorted(frame["currency"].dropna().unique().tolist())
        selections["currencies"] = st.sidebar.multiselect(
            "Currency", options=currencies, default=currencies
        )

    selections["min_score"] = st.sidebar.slider(
        "Minimum Audit Risk Score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Raise this to focus the review budget on the highest-scoring vouchers.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Filters apply to the transaction, rule and vendor views. "
        "The Benford and model pages use the full population, because digit "
        "tests and model metrics are only meaningful on a complete dataset."
    )
    return selections


def apply_filters(frame: pd.DataFrame, selections: dict[str, Any]) -> pd.DataFrame:
    """Apply a :func:`sidebar_filters` selection to a transaction frame."""
    out = frame

    if (date_range := selections.get("date_range")) is not None:
        start, end = date_range
        out = out[(out["transaction_date"] >= start) & (out["transaction_date"] < end)]

    if (levels := selections.get("risk_levels")) is not None and len(levels):
        out = out[out["risk_level"].isin(levels)]

    if (processes := selections.get("processes")) is not None and len(processes):
        out = out[out["process"].isin(processes)]

    if (departments := selections.get("departments")) is not None and len(departments):
        out = out[out["department"].isin(departments)]

    if (currencies := selections.get("currencies")) is not None and len(currencies):
        out = out[out["currency"].isin(currencies)]

    if (min_score := selections.get("min_score")) is not None:
        out = out[out["audit_risk_score"] >= min_score]

    return out


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def money(value: float) -> str:
    """Format a CNY amount for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return format_cny(float(value))


def money_exact(value: float) -> str:
    """Format a CNY amount with full precision and thousands separators."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"¥{float(value):,.2f}"


def count(value: float) -> str:
    """Format an integer count with thousands separators."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{int(value):,}"


def percent(value: float, digits: int = 2) -> str:
    """Format a 0-1 ratio as a percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def score(value: float) -> str:
    """Format a 0-100 score."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{float(value):.1f}"


def risk_color(level: str) -> str:
    """Return the hex colour for a risk band."""
    return RISK_BAND_COLORS.get(str(level), THEME["muted"])


def ensure_ready() -> bool:
    """Guard every page: explain how to build the data instead of crashing."""
    ensure_directories()
    if pipeline_available():
        return True

    st.error("No pipeline output found.")
    st.markdown(
        """
        The dashboard reads the artefacts produced by the analytics pipeline. Build
        them first:

        ```bash
        python src/run_pipeline.py
        ```

        That command generates the synthetic ledger, runs the rule engine, the
        Benford tests, the Isolation Forest and the risk scoring, then writes the
        SQLite warehouse this app queries.
        """
    )
    return False
