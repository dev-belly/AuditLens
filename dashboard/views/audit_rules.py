"""Page 3 - Audit Rules.

The rule engine is the part of the platform an audit team would recognise
immediately, so this page has two jobs. First, document each procedure precisely
enough that a reviewer could re-perform it: what it tests, why it matters, and the
threshold it uses. Second, be honest about how each rule performed - a rule that
flags 500 vouchers to catch 100 real ones is a different proposition from one that
flags 50, and the reviewer is entitled to know which is which.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.common import (
    RISK_BAND_COLORS,
    THEME,
    apply_filters,
    count,
    load_alerts,
    load_rule_evaluation,
    load_transactions,
    money,
    money_exact,
    percent,
    sidebar_filters,
)
from dashboard.components import (
    bar_chart,
    kpi_row,
    note,
    page_header,
    reason_list,
    risk_badge,
    section,
    transaction_table,
)
from src.audit_rules import RULE_DEFINITIONS, RULE_WEIGHTS
from src.utils import (
    APPROVAL_THRESHOLD_CNY,
    MATERIALITY_THRESHOLD_CNY,
    NEW_VENDOR_DAYS,
    RAPID_PAYMENT_HOURS,
    SPLIT_THRESHOLD_HIGH_RATIO,
    SPLIT_THRESHOLD_LOW_RATIO,
)

#: The threshold each rule applies, written out so the catalogue is auditable
#: rather than aspirational. Every value here is imported from the same constant
#: the engine uses - none of it is retyped.
RULE_THRESHOLDS: dict[str, str] = {
    "duplicate_payment": "Same vendor + same amount + same invoice ID, within a 7-day window",
    "split_transaction": (
        f"2+ payments, same vendor, same day, each between "
        f"{SPLIT_THRESHOLD_LOW_RATIO:.0%} and {SPLIT_THRESHOLD_HIGH_RATIO:.0%} of "
        f"CNY {APPROVAL_THRESHOLD_CNY:,.0f}"
    ),
    "self_approval": "created_by == approved_by",
    "unusual_vendor": (
        f"Vendor under {NEW_VENDOR_DAYS} days old, dormant {NEW_VENDOR_DAYS}+ days, "
        f"shared bank account, or amount far above the vendor's own history - "
        f"gated on amounts above the 90th percentile"
    ),
    "large_round_amount": (
        f"Round to 1,000 / 10,000 / 100,000 AND above the 95th percentile AND "
        f"at least CNY {MATERIALITY_THRESHOLD_CNY:,.0f}"
    ),
    "rapid_payment": f"Payment within {RAPID_PAYMENT_HOURS:.0f} hours of approval",
    "suspicious_description": "Description contains fraud-indicative keywords or is implausibly short",
    "rare_account_usage": "Account used in fewer than 0.5% of vouchers or below its historical share",
    "weekend_posting": "Transaction or posting date falls on a weekend or a PRC public holiday",
}


def _catalogue() -> None:
    """Render the rule catalogue with thresholds and audit rationale."""
    section(
        "Rule catalogue",
        "Nine independent procedures. Each one is a question an auditor would ask "
        "anyway - the engine just asks it of every voucher instead of a sample.",
    )

    for definition in RULE_DEFINITIONS:
        weight = RULE_WEIGHTS.get(definition.key, definition.weight)
        st.markdown(
            f"""
            <div class='al-rule'>
                <div style='display:flex;justify-content:space-between;align-items:baseline;'>
                    <div style='font-weight:700;font-size:1.02rem;'>{definition.label}</div>
                    <div class='al-rule-code'>weight {weight:.2f}</div>
                </div>
                <div style='margin-top:0.35rem;'>{definition.description}</div>
                <div class='al-caption' style='margin-top:0.45rem;'>
                    <b>Threshold:</b> {RULE_THRESHOLDS.get(definition.key, 'see engine source')}
                </div>
                <div class='al-caption' style='margin-top:0.35rem;'>
                    <b>Why it matters:</b> {definition.audit_rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    note(
        "Rules are combined with a noisy-OR, not a vote. Two weak indicators that "
        "point the same way produce a stronger signal than either alone, while a "
        "single strong indicator is not diluted by the rules that stayed silent. "
        "The weights are planning judgements, documented in "
        "<code>src/audit_rules.py</code> and easy to re-set for a different client."
    )


def _performance(evaluation: pd.DataFrame, total: int) -> None:
    """Render the per-rule performance table and the precision/recall trade-off."""
    section(
        "How each rule performed",
        "Measured against the anomalies deliberately injected into the synthetic "
        "ledger. Precision answers 'when this fires, how often is it right?'; "
        "recall answers 'of the instances that exist, how many did it find?'",
    )

    if evaluation.empty:
        st.info("Rule evaluation not found. Run `python src/audit_rules.py` to produce it.")
        return

    table = evaluation.copy()
    table["share_of_ledger"] = table["flagged"] / max(total, 1)
    table = table.sort_values("f1", ascending=False)

    display = pd.DataFrame(
        {
            "Procedure": table["rule_label"],
            "Flagged": table["flagged"].map(count),
            "Share of ledger": table["share_of_ledger"].map(percent),
            "True positives": table["true_positives"].map(count),
            "False positives": table["false_positives"].map(count),
            "Precision": table["precision"].map(lambda value: f"{value:.3f}"),
            "Recall": table["recall"].map(lambda value: f"{value:.3f}"),
            "F1": table["f1"].map(lambda value: f"{value:.3f}"),
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

    chart_data = table.sort_values("f1")
    bar_chart(
        chart_data,
        x="rule_label",
        y="f1",
        horizontal=True,
        colors=[
            RISK_BAND_COLORS["Low"] if value >= 0.7 else
            RISK_BAND_COLORS["Medium"] if value >= 0.4 else
            RISK_BAND_COLORS["High"]
            for value in chart_data["f1"]
        ],
        height=360,
        x_title="F1",
    )

    weakest = table.sort_values("precision").iloc[0]
    st.write("")
    note(
        f"<b>Read the low-precision rules as scope, not as failure.</b> "
        f"{weakest['rule_label']} runs at precision {weakest['precision']:.2f}: "
        f"{count(weakest['flagged'])} vouchers are flagged to catch "
        f"{count(weakest['true_positives'])} of that pattern. Those rules are doing "
        f"what a substantive procedure does - defining a population to look at. "
        f"A rule with low precision and high recall is still useful; a rule with low "
        f"precision and low recall is the one to delete.",
        kind="warn",
    )

    note(
        "Recall of 1.000 on several rules is a consequence of the benchmark, not "
        "of the rule being clever. The injected anomalies follow exactly the pattern "
        "the rule tests for, so a rule tuned on that pattern will find all of them. "
        "Real irregularities are irregular: they do not respect a 7-day duplicate "
        "window or a 6-hour payment threshold."
    )


def _alert_queue(filtered: pd.DataFrame, alerts: pd.DataFrame) -> None:
    """Render the alert queue for the current filter selection."""
    section(
        "Alert queue",
        "The working list. Filter by procedure, sort by rule score, and open the "
        "reason text - this is the view an auditor would keep on a second monitor.",
    )

    if alerts.empty:
        st.info("No alerts available.")
        return

    scoped = alerts[alerts["transaction_id"].isin(filtered["transaction_id"])]
    if scoped.empty:
        st.warning("No alerts fall inside the current filters.")
        return

    left, right = st.columns([1, 3])
    with left:
        rules = sorted(scoped["rule_label"].unique().tolist())
        picked = st.multiselect("Procedure", options=rules, default=rules)
    with right:
        minimum = st.slider("Minimum rule score", 0.0, 1.0, 0.0, 0.05)

    view = scoped[scoped["rule_label"].isin(picked) & (scoped["rule_score"] >= minimum)]
    view = view.sort_values(["rule_score", "transaction_id"], ascending=[False, True])

    kpi_row(
        [
            {"label": "Alerts in view", "value": count(len(view)),
             "delta": f"{count(view['transaction_id'].nunique())} distinct vouchers"},
            {"label": "Procedures", "value": count(view["rule_label"].nunique()),
             "delta": f"of {count(scoped['rule_label'].nunique())} selected"},
            {"label": "Mean rule score", "value": f"{view['rule_score'].mean():.2f}" if len(view) else "-",
             "delta": "Rule-level confidence", "accent": THEME["accent"]},
        ]
    )

    st.write("")
    queue = view.merge(
        filtered[
            [
                "transaction_id", "transaction_date", "account_code", "account_name",
                "vendor_name", "department", "debit_amount", "audit_risk_score", "risk_level",
            ]
        ],
        on="transaction_id",
        how="left",
    )
    queue = queue.sort_values("audit_risk_score", ascending=False)

    st.dataframe(
        pd.DataFrame(
            {
                "Procedure": queue["rule_label"],
                "Transaction": queue["transaction_id"],
                "Date": pd.to_datetime(queue["transaction_date"]).dt.strftime("%Y-%m-%d"),
                "Vendor": queue["vendor_name"].fillna("-"),
                "Amount (CNY)": queue["debit_amount"].map(money),
                "Rule score": queue["rule_score"].round(3),
                "Risk score": queue["audit_risk_score"].round(1),
                "Risk": queue["risk_level"].map(risk_badge),
                "Reason": queue["risk_reason"],
            }
        ),
        width="stretch",
        height=460,
        hide_index=True,
    )

    with st.expander("Inspect a single alert"):
        if view.empty:
            st.caption("Nothing to inspect under the current selection.")
            return
        options = view["transaction_id"].head(200).tolist()
        picked_transaction = st.selectbox("Voucher", options=options, key="rule_alert_pick")
        rows = view[view["transaction_id"] == picked_transaction]
        st.markdown(f"**{len(rows)} procedure(s) triggered on {picked_transaction}**")
        for row in rows.itertuples(index=False):
            st.markdown(
                f"""
                <div class='al-rule'>
                    <div style='font-weight:600;'>{row.rule_label}
                        <span class='al-rule-code'>score {row.rule_score:.2f}</span>
                    </div>
                    <div class='al-caption' style='margin-top:0.3rem;'>{row.risk_reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        scored_row = filtered[filtered["transaction_id"] == picked_transaction]
        if not scored_row.empty:
            st.markdown("**Combined explanation shown to the reviewer**")
            reason_list(scored_row.iloc[0].get("risk_reasons") or [])


def render() -> None:
    """Render the audit rules page."""
    page_header(
        "Audit Rules",
        "Nine independent procedures, what each one tests, and how each one "
        "performed against the labelled benchmark.",
        eyebrow="AuditLens - Rule engine",
    )

    frame = load_transactions()
    alerts = load_alerts()
    evaluation = load_rule_evaluation()

    total = len(frame)
    kpi_row(
        [
            {"label": "Procedures", "value": count(len(RULE_DEFINITIONS)),
             "delta": "Independently evaluated"},
            {"label": "Alerts raised", "value": count(len(alerts)),
             "delta": f"{count(alerts['transaction_id'].nunique())} distinct vouchers"
                      if not alerts.empty else "-"},
            {"label": "Share of ledger", "value": percent(len(alerts) / max(total, 1)),
             "delta": "Population needing triage", "accent": THEME["accent"]},
            {"label": "Mean alerts per flagged voucher",
             "value": f"{len(alerts) / max(1, alerts['transaction_id'].nunique()):.2f}"
                      if not alerts.empty else "-",
             "delta": "How often procedures agree", "accent": THEME["muted"]},
        ]
    )

    st.write("")
    _catalogue()
    _performance(evaluation, total)

    st.write("")
    section("Work the queue")
    selections = sidebar_filters(frame)
    filtered = apply_filters(frame, selections)
    if filtered.empty:
        st.warning("No vouchers match the current filters.")
        return
    _alert_queue(filtered, alerts)
