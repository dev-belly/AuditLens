"""Page 6 - Vendor Risk.

Vendor-level analysis is where transaction testing turns into a finding. A single
odd voucher is a question; a vendor whose whole payment history is odd is a
conclusion. This page aggregates the voucher-level work up to the counterparty and
scores it on seven weighted features, then surfaces the three patterns that most
often indicate a shell company or a related party: shared bank accounts, dormancy
followed by reactivation, and spend concentration.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    RISK_BAND_COLORS,
    THEME,
    count,
    load_transactions,
    load_vendors,
    money,
    money_exact,
    percent,
    risk_color,
)
from dashboard.components import (
    bar_chart,
    kpi_row,
    note,
    page_header,
    risk_badge,
    section,
)
from src.risk_scoring import VENDOR_RISK_WEIGHTS
from src.utils import NEW_VENDOR_DAYS, format_cny

#: Sub-score columns written by the risk scoring stage, in presentation order.
VENDOR_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("vendor_risk_alert_rate", "Alert rate"),
    ("vendor_risk_master_risk", "Master data risk"),
    ("vendor_risk_shared_bank_account", "Shared bank account"),
    ("vendor_risk_new_vendor", "New vendor"),
    ("vendor_risk_self_approval_share", "Self-approval share"),
    ("vendor_risk_weekend_share", "Weekend posting share"),
    ("vendor_risk_amount_concentration", "Amount concentration"),
)


def _component_weights() -> dict[str, float]:
    """Map the display labels back onto the configured component weights."""
    mapping = {
        "Alert rate": VENDOR_RISK_WEIGHTS.get("alert_rate", 0.0),
        "Master data risk": VENDOR_RISK_WEIGHTS.get("master_risk", 0.0),
        "Shared bank account": VENDOR_RISK_WEIGHTS.get("shared_bank_account", 0.0),
        "New vendor": VENDOR_RISK_WEIGHTS.get("new_vendor", 0.0),
        "Self-approval share": VENDOR_RISK_WEIGHTS.get("self_approval_share", 0.0),
        "Weekend posting share": VENDOR_RISK_WEIGHTS.get("weekend_share", 0.0),
        "Amount concentration": VENDOR_RISK_WEIGHTS.get("amount_concentration", 0.0),
    }
    return mapping


def _scatter(vendors: pd.DataFrame) -> go.Figure:
    """Plot vendor risk against cumulative spend.

    The upper-right quadrant is the uncomfortable one: high risk and high value.
    Those vendors are where the review hours should go first.
    """
    figure = go.Figure()
    for level in ("Low", "Medium", "High", "Critical"):
        subset = vendors[vendors["vendor_risk_level"] == level]
        if subset.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=subset["vendor_total_amount"],
                y=subset["vendor_risk_score"],
                mode="markers",
                name=level,
                marker=dict(
                    size=(subset["alert_count"].clip(lower=1) * 2.2 + 6),
                    color=risk_color(level),
                    opacity=0.72,
                    line=dict(width=0.5, color="#FFFFFF"),
                ),
                customdata=subset[["vendor_name", "alert_count"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Total spend ¥%{x:,.0f}<br>"
                    "Risk score %{y:.1f}<br>"
                    "Alerts %{customdata[1]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        height=430,
        margin=dict(l=10, r=20, t=30, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title="Vendor risk band"),
        xaxis=dict(title="Cumulative spend (CNY)", gridcolor="#EDF1F4", type="log"),
        yaxis=dict(title="Vendor risk score (0-100)", gridcolor="#EDF1F4", range=[0, 100]),
    )
    return figure


def _vendor_detail(vendor_row: pd.Series, frame: pd.DataFrame) -> None:
    """Render the drill-down for one vendor."""
    level = str(vendor_row["vendor_risk_level"])

    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;'>
            <span style='font-size:1.15rem;font-weight:700;'>{vendor_row['vendor_name']}</span>
            {risk_badge(level)}
            <span class='al-caption'>Vendor risk score
                <b>{vendor_row['vendor_risk_score']:.1f}</b> / 100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(5)
    metrics = (
        ("Payments", count(vendor_row["vendor_transaction_count"])),
        ("Total spend", format_cny(vendor_row["vendor_total_amount"])),
        ("Average payment", money_exact(vendor_row["vendor_average_amount"])),
        ("Largest payment", money_exact(vendor_row["vendor_max_amount"])),
        ("Rule alerts", count(vendor_row["alert_count"])),
    )
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class='al-rule'>
                    <div class='al-kpi-label'>{label}</div>
                    <div style='font-size:1.15rem;font-weight:700;'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("**Vendor master data**")
        registration = vendor_row.get("registration_date")
        st.dataframe(
            pd.DataFrame(
                {
                    "Field": [
                        "Vendor ID", "Category", "Registration date", "Country",
                        "Bank account", "Shared bank account", "Master risk rating",
                        "Alert rate", "Multi-alert vouchers", "First payment", "Last payment",
                    ],
                    "Value": [
                        str(vendor_row["vendor_id"]),
                        str(vendor_row.get("vendor_category") or "-"),
                        pd.to_datetime(registration).strftime("%Y-%m-%d") if pd.notna(registration) else "-",
                        str(vendor_row.get("country") or "-"),
                        str(vendor_row.get("bank_account") or "-"),
                        "Yes - shared with another vendor" if bool(vendor_row.get("shared_bank_account")) else "No",
                        str(vendor_row.get("risk_level") or "-"),
                        percent(float(vendor_row.get("alert_rate") or 0), 1),
                        count(vendor_row.get("multi_alert_transaction_count", 0)),
                        str(vendor_row.get("vendor_first_transaction") or "-")[:10],
                        str(vendor_row.get("vendor_last_transaction") or "-")[:10],
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with right:
        st.markdown("**Why this vendor scores what it does**")
        components = {
            label: float(vendor_row.get(column) or 0.0)
            for column, label in VENDOR_COMPONENTS
            if column in vendor_row.index
        }
        if components:
            weights = _component_weights()
            ordered = {key: components[key] for key in components}
            figure = go.Figure(
                go.Bar(
                    x=[ordered[key] * weights.get(key, 0.0) for key in ordered],
                    y=list(ordered.keys()),
                    orientation="h",
                    marker_color=THEME["accent"],
                    text=[
                        f"{ordered[key]:.0f} x {weights.get(key, 0):.0%} = {ordered[key] * weights.get(key, 0):.1f}"
                        for key in ordered
                    ],
                    textposition="outside",
                )
            )
            figure.update_layout(
                height=330,
                margin=dict(l=10, r=120, t=10, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=10),
                showlegend=False,
                xaxis=dict(title="Weighted contribution to the vendor score", gridcolor="#EDF1F4"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(figure, width="stretch")

    # ----------------------------------------------------------------- #
    # This vendor's vouchers
    # ----------------------------------------------------------------- #
    st.markdown("**Vouchers raised against this vendor**")
    own = frame[frame["vendor_id"] == vendor_row["vendor_id"]].sort_values(
        "audit_risk_score", ascending=False
    )
    if own.empty:
        st.caption("No vouchers in the current population for this vendor.")
        return

    flagged = own[own["rule_alert_count"] > 0]
    st.markdown(
        f"<div class='al-caption'>{count(len(flagged))} of {count(len(own))} payments "
        f"carry at least one rule alert.</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Transaction": own["transaction_id"],
                "Date": pd.to_datetime(own["transaction_date"]).dt.strftime("%Y-%m-%d"),
                "Account": own["account_code"].astype(str) + " " + own["account_name"].astype(str),
                "Amount (CNY)": own["debit_amount"].map(money),
                "Alerts": own["rule_alert_count"].astype(int),
                "Risk score": own["audit_risk_score"].round(1),
                "Risk": own["risk_level"].map(risk_badge),
                "Why": own["risk_reason_text"].fillna("").astype(str).str.slice(0, 120),
            }
        ),
        width="stretch",
        hide_index=True,
        height=340,
    )


def render() -> None:
    """Render the vendor risk page."""
    page_header(
        "Vendor Risk",
        "Voucher-level findings aggregated to the counterparty, scored on seven "
        "weighted features and ranked by how much of the review budget each deserves.",
        eyebrow="AuditLens - Counterparty analysis",
    )

    vendors = load_vendors()
    frame = load_transactions()

    if vendors.empty:
        st.error("Vendor risk table not found. Run `python src/risk_scoring.py` first.")
        return

    total_spend = float(vendors["vendor_total_amount"].sum())
    high_risk = vendors[vendors["vendor_risk_level"].isin(("High", "Critical"))]
    shared = vendors[vendors["shared_bank_account"].astype(bool)]
    new_vendors = vendors[vendors.get("is_new_vendor", pd.Series(False, index=vendors.index)).astype(bool)]

    kpi_row(
        [
            {"label": "Vendors in scope", "value": count(len(vendors)),
             "delta": f"{format_cny(total_spend)} total spend"},
            {"label": "High or critical", "value": count(len(high_risk)),
             "delta": f"{percent(len(high_risk) / max(len(vendors), 1))} of vendors",
             "accent": RISK_BAND_COLORS["High"]},
            {"label": "Shared bank accounts", "value": count(len(shared)),
             "delta": "Possible shell-company indicator",
             "accent": RISK_BAND_COLORS["Critical"]},
            {"label": f"Registered < {NEW_VENDOR_DAYS} days", "value": count(len(new_vendors)),
             "delta": "Paid before establishing a history", "accent": THEME["muted"]},
            {"label": "Alerts per vendor",
             "value": f"{vendors['alert_count'].mean():.2f}",
             "delta": f"max {int(vendors['alert_count'].max())} on a single vendor",
             "accent": THEME["muted"]},
        ]
    )

    # ----------------------------------------------------------------- #
    # Risk versus value
    # ----------------------------------------------------------------- #
    st.write("")
    section(
        "Risk against value",
        "Bubble size is the number of rule alerts. The vendors worth reviewing first "
        "are high on the y-axis and far to the right - high risk on material spend. "
        "High risk on immaterial spend can wait.",
    )
    st.plotly_chart(_scatter(vendors), width="stretch")

    # ----------------------------------------------------------------- #
    # Ranking
    # ----------------------------------------------------------------- #
    section(
        "Vendor risk ranking",
        "Seven weighted features, each scored 0-100 and combined. The weights are "
        "documented in <code>src/risk_scoring.py</code> and are a planning "
        "judgement, not an actuarial result.",
    )

    weights = _component_weights()
    st.markdown(
        " ".join(
            f"<span class='al-badge' style='background:{THEME['primary']};margin-right:0.35rem;'>"
            f"{label} {weight:.0%}</span>"
            for label, weight in weights.items()
            if weight > 0
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    top_n = st.slider("Vendors to display", min_value=10, max_value=100, value=25, step=5)
    ranking = vendors.head(top_n).copy()
    st.dataframe(
        pd.DataFrame(
            {
                "Rank": range(1, len(ranking) + 1),
                "Vendor": ranking["vendor_name"],
                "Category": ranking["vendor_category"].fillna("-"),
                "Payments": ranking["vendor_transaction_count"].map(count),
                "Total spend": ranking["vendor_total_amount"].map(money),
                "Alerts": ranking["alert_count"].astype(int),
                "Alert rate": ranking["alert_rate"].map(lambda value: percent(float(value or 0), 1)),
                "Shared bank": ranking["shared_bank_account"].map(lambda flag: "Yes" if bool(flag) else ""),
                "Risk score": ranking["vendor_risk_score"].round(1),
                "Risk": ranking["vendor_risk_level"].map(risk_badge),
            }
        ),
        width="stretch",
        hide_index=True,
        height=460,
    )

    # ----------------------------------------------------------------- #
    # Drill-down
    # ----------------------------------------------------------------- #
    section(
        "Vendor drill-down",
        "Open any vendor to see the component breakdown and every voucher raised "
        "against them.",
    )
    options = vendors["vendor_id"].tolist()
    selected = st.selectbox(
        "Vendor",
        options=options,
        format_func=lambda value: (
            f"{vendors.loc[vendors['vendor_id'] == value, 'vendor_name'].iloc[0]}"
            f"  |  risk {float(vendors.loc[vendors['vendor_id'] == value, 'vendor_risk_score'].iloc[0]):.1f}"
            f"  |  {vendors.loc[vendors['vendor_id'] == value, 'vendor_risk_level'].iloc[0]}"
        ),
        key="vendor_drilldown",
    )
    st.write("")
    _vendor_detail(vendors.loc[vendors["vendor_id"] == selected].iloc[0], frame)

    # ----------------------------------------------------------------- #
    # Specific schemes
    # ----------------------------------------------------------------- #
    st.write("")
    section(
        "Pattern-specific views",
        "The three vendor schemes that audit procedures target most often, each "
        "isolated from the rest of the population.",
    )

    tab_shared, tab_dormant, tab_concentration = st.tabs(
        ["Shared bank accounts", "Dormant reactivation", "Spend concentration"]
    )

    with tab_shared:
        if shared.empty:
            st.markdown(
                "<div class='al-caption'>No bank account is shared between two "
                "vendors in this population.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='al-caption'>Two vendor records pointing at the same bank "
                "account is the classic shell-company signature: one beneficiary "
                "behind multiple supplier names. It is also frequently innocent - "
                "group companies legitimately share a treasury account - so it is a "
                "question to ask, not a conclusion.</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                shared[
                    ["vendor_name", "bank_account", "vendor_category", "registration_date",
                     "vendor_transaction_count", "vendor_total_amount", "vendor_risk_score"]
                ].rename(
                    columns={
                        "vendor_name": "Vendor",
                        "bank_account": "Bank account",
                        "vendor_category": "Category",
                        "registration_date": "Registered",
                        "vendor_transaction_count": "Payments",
                        "vendor_total_amount": "Total spend",
                        "vendor_risk_score": "Risk score",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_dormant:
        # Dormancy is measured from the vendor master record: a counterparty that
        # was registered long before it was first paid is the footprint a
        # dormant-then-reactivated vendor scheme leaves behind - the entity exists
        # and is approved, then sits unused until someone needs it.
        #
        # The earlier version keyed off ``days_since_last_transaction``, which only
        # exists on the transaction table, so the condition could never be met and
        # the tab silently fell through to its fallback on every run.
        registration = pd.to_datetime(vendors["registration_date"], errors="coerce")
        first_payment = pd.to_datetime(vendors["vendor_first_transaction"], errors="coerce")
        dormancy_gap = (first_payment - registration).dt.days

        dormant = vendors[(vendors["alert_count"] > 0) & (dormancy_gap > NEW_VENDOR_DAYS)].copy()

        if dormant.empty:
            st.markdown(
                "<div class='al-caption'>No vendor in this population was paid more "
                "than "
                f"{NEW_VENDOR_DAYS} days after it was registered. Showing vendors "
                "carrying two or more alerts instead - the same question, "
                "approximated.</div>",
                unsafe_allow_html=True,
            )
            dormant = vendors[vendors["alert_count"] >= 2].copy()
        else:
            st.markdown(
                "<div class='al-caption'>Each vendor below was registered more than "
                f"{NEW_VENDOR_DAYS} days before its first payment and carries at "
                "least one rule alert. A dormant entity that is suddenly used is "
                "worth a question about who approved it onto the vendor master and "
                "who benefited.</div>",
                unsafe_allow_html=True,
            )

        dormant["Dormancy gap (days)"] = dormancy_gap.reindex(dormant.index)
        st.dataframe(
            dormant[
                ["vendor_name", "vendor_transaction_count", "vendor_total_amount",
                 "alert_count", "Dormancy gap (days)", "vendor_last_transaction",
                 "vendor_risk_score", "vendor_risk_level"]
            ].rename(
                columns={
                    "vendor_name": "Vendor",
                    "vendor_transaction_count": "Payments",
                    "vendor_total_amount": "Total spend",
                    "alert_count": "Alerts",
                    "vendor_last_transaction": "Last payment",
                    "vendor_risk_score": "Risk score",
                    "vendor_risk_level": "Risk",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_concentration:
        concentration = vendors.sort_values("vendor_total_amount", ascending=False).head(15)
        share = concentration["vendor_total_amount"] / max(total_spend, 1)
        chart = pd.DataFrame(
            {"Vendor": concentration["vendor_name"], "share": share.values}
        )
        bar_chart(
            chart,
            x="Vendor",
            y="share",
            horizontal=True,
            colors=[THEME["primary"]] * len(chart),
            height=420,
            x_title="Share of total vendor spend",
        )
        top_10_share = float(vendors.nlargest(10, "vendor_total_amount")["vendor_total_amount"].sum() / max(total_spend, 1))
        note(
            f"The ten largest vendors account for <b>{percent(top_10_share, 1)}</b> of "
            f"vendor spend. Concentration is not misconduct, but it changes the audit "
            f"approach: the smaller the supplier base, the more each relationship "
            f"matters and the less a sample of transactions tells you about the "
            f"population."
        )
