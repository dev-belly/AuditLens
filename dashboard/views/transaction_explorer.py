"""Page 2 - Transaction Explorer with risk explanation drill-down.

This is the page that decides whether the whole project is credible. A ranked list
of voucher IDs is worth nothing to an auditor; a ranked list where every entry can
be opened, explained line by line, and compared against the population is a working
tool. So the drill-down is the point of this page, not an appendix to it.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    RISK_BAND_COLORS,
    THEME,
    apply_filters,
    count,
    load_alerts,
    load_transactions,
    load_vendors,
    money_exact,
    percent,
    sidebar_filters,
)
from dashboard.components import (
    component_breakdown,
    kpi_row,
    note,
    page_header,
    reason_list,
    risk_badge,
    section,
    transaction_table,
)
from src.utils import RISK_WEIGHTS, format_cny

#: Components of the composite score, in the order they are presented.
COMPONENT_ORDER: tuple[tuple[str, str], ...] = (
    ("risk_component_rule", "Rule indicators"),
    ("risk_component_ml", "Isolation Forest"),
    ("risk_component_vendor", "Vendor risk"),
    ("risk_component_amount", "Amount context"),
    ("risk_component_statistical", "Statistical (Benford)"),
)

#: Rule flag columns and the label used in the drill-down.
RULE_FLAGS: tuple[tuple[str, str], ...] = (
    ("duplicate_payment_flag", "Duplicate payment"),
    ("split_transaction_flag", "Split transaction"),
    ("self_approval_flag", "Self approval"),
    ("unusual_vendor_flag", "Unusual vendor"),
    ("large_round_amount_flag", "Large round amount"),
    ("rapid_payment_flag", "Rapid payment"),
    ("suspicious_description_flag", "Suspicious description"),
    ("rare_account_usage_flag", "Rare account usage"),
    ("weekend_posting_flag", "Weekend / holiday posting"),
)


def _timeline_figure(row: pd.Series) -> go.Figure | None:
    """Build a control-timeline figure for a single voucher.

    The order and spacing of invoice, approval, payment and posting dates is what
    most control findings actually rest on - a payment made four hours after
    approval tells a different story from one made three weeks later. Plotting the
    timeline makes that visible instead of asking the reader to subtract dates.
    """
    events: list[tuple[str, pd.Timestamp]] = []
    for column, label in (
        ("invoice_time", "Invoice received"),
        ("approval_time", "Approved"),
        ("payment_time", "Paid"),
        ("posting_date", "Posted to ledger"),
    ):
        value = row.get(column)
        if value is not None and not pd.isna(value):
            events.append((label, pd.Timestamp(value)))

    if len(events) < 2:
        return None

    events.sort(key=lambda item: item[1])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[timestamp for _, timestamp in events],
            y=[label for label, _ in events],
            mode="markers+lines",
            marker=dict(size=13, color=THEME["accent"]),
            line=dict(color=THEME["muted"], width=2),
            text=[timestamp.strftime("%Y-%m-%d %H:%M") for _, timestamp in events],
            textposition="top center",
        )
    )
    figure.update_layout(
        height=250,
        margin=dict(l=10, r=20, t=30, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
        xaxis=dict(gridcolor="#EDF1F4"),
    )
    return figure


def _render_drilldown(row: pd.Series, frame: pd.DataFrame, alerts: pd.DataFrame, vendors: pd.DataFrame) -> None:
    """Render the full explanation for one voucher."""
    level = str(row["risk_level"])

    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem;'>
            <span style='font-size:1.15rem;font-weight:700;'>{row['transaction_id']}</span>
            {risk_badge(level)}
            <span class='al-caption'>Audit Risk Score
                <b>{row['audit_risk_score']:.1f}</b> / 100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------- #
    # Why this voucher is flagged - the most important block
    # ------------------------------------------------------------- #
    section(
        "Why this voucher is flagged",
        "Each line names the indicator, the value that breached it and the "
        "threshold it breached. Ordered from strongest evidence to weakest.",
    )
    reason_list(row.get("risk_reasons") or [])

    st.write("")
    # ------------------------------------------------------------- #
    # Voucher detail
    # ------------------------------------------------------------- #
    left, right = st.columns([1.1, 1])

    with left:
        section("Voucher detail")
        detail = pd.DataFrame(
            {
                "Field": [
                    "Journal ID",
                    "Invoice ID",
                    "Transaction date",
                    "Posting date",
                    "Account",
                    "Vendor",
                    "Department",
                    "Business process",
                    "Payment method",
                    "Currency",
                    "Amount (original)",
                    "Amount (CNY)",
                    "Created by",
                    "Approved by",
                    "Approval turnaround",
                    "Payment delay",
                    "Description",
                ],
                "Value": [
                    str(row.get("journal_id", "-")),
                    str(row.get("invoice_id") or "-"),
                    pd.to_datetime(row["transaction_date"]).strftime("%Y-%m-%d"),
                    pd.to_datetime(row["posting_date"]).strftime("%Y-%m-%d") if pd.notna(row.get("posting_date")) else "-",
                    f"{row.get('account_code', '-')} {row.get('account_name', '')}",
                    str(row.get("vendor_name") or "-"),
                    str(row.get("department") or "-"),
                    str(row.get("process") or "-"),
                    str(row.get("payment_method") or "-"),
                    str(row.get("currency", "CNY")),
                    f"{float(row.get('amount_original') or 0):,.2f}" if pd.notna(row.get("amount_original")) else "-",
                    money_exact(row["debit_amount"]),
                    str(row.get("created_by", "-")),
                    str(row.get("approved_by") or "-"),
                    f"{float(row['approval_time_hours']):.1f} h" if pd.notna(row.get("approval_time_hours")) else "-",
                    f"{float(row['payment_delay_hours']):.1f} h" if pd.notna(row.get("payment_delay_hours")) else "-",
                    str(row.get("description") or "-"),
                ],
            }
        )
        st.dataframe(detail, width="stretch", hide_index=True, height=640)

    with right:
        section("Control timeline")
        timeline = _timeline_figure(row)
        if timeline is not None:
            st.plotly_chart(timeline, width="stretch")
        else:
            st.markdown(
                "<div class='al-caption'>No usable timeline - this voucher has fewer "
                "than two recorded control events.</div>",
                unsafe_allow_html=True,
            )

        section("Score composition")
        components = {
            label: float(row.get(column) or 0.0) for column, label in COMPONENT_ORDER if column in row.index
        }
        weights = {
            "Rule indicators": RISK_WEIGHTS.get("rule_risk", 0.0),
            "Isolation Forest": RISK_WEIGHTS.get("ml_risk", 0.0),
            "Vendor risk": RISK_WEIGHTS.get("vendor_risk", 0.0),
            "Amount context": RISK_WEIGHTS.get("amount_risk", 0.0),
            "Statistical (Benford)": RISK_WEIGHTS.get("statistical_risk", 0.0),
        }
        component_breakdown(components, weights=weights)

        st.markdown(
            f"<div class='al-caption'>The components are 0-100 sub-scores; the "
            f"composite applies the weights shown. "
            f"Rule <b>{float(row.get('risk_component_rule') or 0):.0f}</b> x "
            f"{weights['Rule indicators']:.0%} + ML "
            f"<b>{float(row.get('risk_component_ml') or 0):.0f}</b> x "
            f"{weights['Isolation Forest']:.0%} + ... = "
            f"<b>{row['audit_risk_score']:.1f}</b>.</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------- #
    # Rule-by-rule result
    # ------------------------------------------------------------- #
    section(
        "Procedure-by-procedure result",
        "Every rule is evaluated for every voucher. Showing the rules that did "
        "<i>not</i> fire is as useful as showing the ones that did - it tells the "
        "reviewer what has already been cleared.",
    )

    triggered = []
    for column, label in RULE_FLAGS:
        if column in row.index and bool(row[column]):
            score_column = column.replace("_flag", "_score")
            rule_score = float(row.get(score_column) or 0.0)
            triggered.append({"Procedure": label, "Rule score": round(rule_score, 3), "Result": "Triggered"})

    if triggered:
        st.dataframe(pd.DataFrame(triggered), width="stretch", hide_index=True)
    else:
        st.markdown(
            "<div class='al-caption'>No audit rule triggered. The voucher is scored "
            "on its amount profile and statistical context only.</div>",
            unsafe_allow_html=True,
        )

    if not alerts.empty:
        own = alerts[alerts["transaction_id"] == row["transaction_id"]]
        if not own.empty:
            with st.expander(f"Rule alert detail ({len(own)} alerts)"):
                st.dataframe(
                    own[["rule_label", "rule_score", "risk_reason"]].rename(
                        columns={"rule_label": "Procedure", "rule_score": "Score", "risk_reason": "Reason"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

    # ------------------------------------------------------------- #
    # Peer comparison
    # ------------------------------------------------------------- #
    section(
        "How this voucher compares",
        "A flag is only meaningful relative to the population. These are the same "
        "measures for every voucher sharing this account.",
    )

    account_peers = frame[frame["account_code"] == row["account_code"]]
    vendor_peers = frame[frame["vendor_id"] == row["vendor_id"]] if pd.notna(row.get("vendor_id")) else pd.DataFrame()

    columns = st.columns(3)
    with columns[0]:
        st.markdown(
            f"""
            <div class='al-rule'>
                <div class='al-kpi-label'>Amount percentile</div>
                <div style='font-size:1.4rem;font-weight:700;'>
                    {percent(float(row.get('amount_percentile') or 0), 1)}
                </div>
                <div class='al-caption'>of all {count(len(frame))} vouchers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class='al-rule'>
                <div class='al-kpi-label'>Account peers</div>
                <div style='font-size:1.4rem;font-weight:700;'>{count(len(account_peers))}</div>
                <div class='al-caption'>
                    account mean {money_exact(account_peers['debit_amount'].mean())}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[2]:
        if not vendor_peers.empty:
            vendor_row = vendors[vendors["vendor_id"] == row["vendor_id"]]
            vendor_risk = float(vendor_row["vendor_risk_score"].iloc[0]) if not vendor_row.empty else 0.0
            st.markdown(
                f"""
                <div class='al-rule'>
                    <div class='al-kpi-label'>Vendor</div>
                    <div style='font-size:1.4rem;font-weight:700;'>{vendor_risk:.1f}</div>
                    <div class='al-caption'>
                        risk score across {count(len(vendor_peers))} payments
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='al-rule'>
                    <div class='al-kpi-label'>Vendor</div>
                    <div style='font-size:1.4rem;font-weight:700;'>n/a</div>
                    <div class='al-caption'>Employee reimbursement - no vendor</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    note(
        "A flag is a question, not a conclusion. This voucher is worth asking about "
        "because of the combination of indicators above; the answer may be entirely "
        "innocent. The rule thresholds are planning assumptions and should be "
        "re-set for the client's own materiality and control environment."
    )


def render() -> None:
    """Render the transaction explorer."""
    page_header(
        "Transaction Explorer",
        "Filter the population, sort by risk, then open any voucher to see exactly "
        "why it scored what it did.",
        eyebrow="AuditLens - Voucher level",
    )

    frame = load_transactions()
    vendors = load_vendors()
    alerts = load_alerts()

    selections = sidebar_filters(frame)
    filtered = apply_filters(frame, selections)

    if filtered.empty:
        st.warning("No vouchers match the current filters. Widen the selection in the sidebar.")
        return

    # ----------------------------------------------------------------- #
    # Filtered population summary
    # ----------------------------------------------------------------- #
    kpi_row(
        [
            {"label": "Vouchers in view", "value": count(len(filtered)),
             "delta": f"of {count(len(frame))} in the population"},
            {"label": "Value in view", "value": format_cny(filtered["debit_amount"].sum()),
             "delta": "Sum of debit amounts"},
            {"label": "Flagged", "value": count(int((filtered["rule_alert_count"] > 0).sum())),
             "delta": percent((filtered["rule_alert_count"] > 0).mean()),
             "accent": THEME["accent"]},
            {"label": "Mean risk score", "value": f"{filtered['audit_risk_score'].mean():.1f}",
             "delta": f"max {filtered['audit_risk_score'].max():.0f}",
             "accent": THEME["muted"]},
        ]
    )

    st.write("")
    section("Voucher population")

    control, _ = st.columns([1, 2])
    with control:
        sort_option = st.selectbox(
            "Sort by",
            options=(
                "Audit Risk Score (high to low)",
                "Audit Risk Score (low to high)",
                "Amount (high to low)",
                "Amount (low to high)",
                "Date (newest first)",
                "Date (oldest first)",
                "Rule alerts (most first)",
            ),
            index=0,
        )

    sort_map = {
        "Audit Risk Score (high to low)": ("audit_risk_score", False),
        "Audit Risk Score (low to high)": ("audit_risk_score", True),
        "Amount (high to low)": ("debit_amount", False),
        "Amount (low to high)": ("debit_amount", True),
        "Date (newest first)": ("transaction_date", False),
        "Date (oldest first)": ("transaction_date", True),
        "Rule alerts (most first)": ("rule_alert_count", False),
    }
    sort_column, ascending = sort_map[sort_option]
    ordered = filtered.sort_values(sort_column, ascending=ascending)

    page_size = st.select_slider(
        "Rows to display", options=(25, 50, 100, 250, 500), value=100
    )
    transaction_table(ordered.head(page_size), height=460, key="explorer_table")
    st.markdown(
        f"<div class='al-caption'>Showing {count(min(page_size, len(ordered)))} of "
        f"{count(len(ordered))} vouchers matching the filters.</div>",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------- #
    # Drill-down
    # ----------------------------------------------------------------- #
    section(
        "Risk explanation drill-down",
        "Select any voucher to see the evidence behind its score, the control "
        "timeline, and how it compares with its peers.",
    )

    options = ordered.head(1000)["transaction_id"].tolist()
    default_index = 0
    selected = st.selectbox(
        "Voucher to inspect",
        options=options,
        index=default_index,
        format_func=lambda value: (
            f"{value}  |  score {float(ordered.loc[ordered['transaction_id'] == value, 'audit_risk_score'].iloc[0]):.1f}"
            f"  |  {ordered.loc[ordered['transaction_id'] == value, 'risk_level'].iloc[0]}"
        ),
        key="explorer_drilldown",
    )

    if selected is None:
        return

    row = frame.loc[frame["transaction_id"] == selected].iloc[0]
    st.write("")
    _render_drilldown(row, frame, alerts, vendors)
