"""AuditLens - Financial Anomaly Detection & Audit Analytics Platform.

Streamlit entry point and navigation.

Run with::

    streamlit run dashboard/app.py

The app is a thin presentation layer over the artefacts written by
``src/run_pipeline.py``. It performs no analytics of its own, which is deliberate:
the numbers on screen are the same numbers in ``outputs/reports``, and the same
numbers in ``data/auditlens.db``. A dashboard that recomputes its own metrics
eventually disagrees with the pipeline, and then nobody trusts either.

Pages
-----
1. **Executive Overview** - population, triage and data reliability at a glance.
2. **Transaction Explorer** - filter, sort, and open any voucher to see why it scored.
3. **Audit Rules** - the nine procedures, their thresholds, and how each performed.
4. **Benford Analysis** - digit tests, conformity statistics and disaggregation.
5. **Machine Learning** - Isolation Forest, its evaluation, and the review-budget trade-off.
6. **Vendor Risk** - counterparty scoring and the shell-company indicators.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` and `dashboard` importable regardless of how Streamlit is invoked.
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
for _path in (str(PROJECT_ROOT), str(DASHBOARD_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import streamlit as st

st.set_page_config(
    page_title="AuditLens - Audit Analytics",
    page_icon=":material/policy:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**AuditLens** - financial anomaly detection and audit analytics over a "
            "synthetic journal ledger. Synthetic data only; no client information is "
            "used anywhere in this project."
        )
    },
)

from dashboard.common import ensure_ready, inject_css, load_summary
from dashboard.views import (
    audit_rules,
    benford_analysis,
    executive_overview,
    machine_learning,
    transaction_explorer,
    vendor_risk,
)


def _sidebar_branding(summary: dict) -> None:
    """Render the sidebar header and the data provenance note.

    Provenance sits directly under the title rather than at the bottom of the
    sidebar, because everything below it is navigation: a reader should know what
    they are looking at before they choose where to go.
    """
    population = summary.get("population", {}) if summary else {}
    date_range = population.get("date_range", ["-", "-"])

    st.sidebar.markdown(
        f"""
        <div style='padding-bottom:0.5rem;'>
            <div style='font-size:1.22rem;font-weight:800;color:#1F4E79;letter-spacing:-0.01em;'>
                AuditLens
            </div>
            <div style='font-size:0.76rem;color:#8A9BA8;line-height:1.4;'>
                Financial Anomaly Detection<br/>&amp; Audit Analytics
            </div>
        </div>
        <div style='font-size:0.72rem;color:#8A9BA8;line-height:1.65;
                    border-top:1px solid #DCE3E8;padding-top:0.5rem;'>
            <b>{population.get('total_transactions', 0):,}</b> synthetic vouchers<br/>
            {date_range[0]} to {date_range[1]}<br/>
            Warehouse: data/auditlens.db
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")


def main() -> None:
    """Configure the app and dispatch to the selected page."""
    inject_css()

    if not ensure_ready():
        return

    pages = [
        st.Page(
            executive_overview.render,
            title="Executive Overview",
            icon=":material/dashboard:",
            url_path="overview",
            default=True,
        ),
        st.Page(
            transaction_explorer.render,
            title="Transaction Explorer",
            icon=":material/manage_search:",
            url_path="explorer",
        ),
        st.Page(
            audit_rules.render,
            title="Audit Rules",
            icon=":material/rule:",
            url_path="rules",
        ),
        st.Page(
            benford_analysis.render,
            title="Benford Analysis",
            icon=":material/ssid_chart:",
            url_path="benford",
        ),
        st.Page(
            machine_learning.render,
            title="Machine Learning",
            icon=":material/scatter_plot:",
            url_path="model",
        ),
        st.Page(
            vendor_risk.render,
            title="Vendor Risk",
            icon=":material/storefront:",
            url_path="vendors",
        ),
    ]

    _sidebar_branding(load_summary())
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


main()
