"""Page 1 - Executive Overview.

The one page a partner reads. It answers four questions in order: how big is the
population, how much of it did we flag, which procedures produced the findings,
and can the underlying data be relied upon at all. Detail lives on the other
pages; this one exists to support a conversation, not an investigation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.common import (
    RISK_BAND_COLORS,
    THEME,
    count,
    load_alerts,
    load_benford,
    load_metrics,
    load_quality,
    load_summary,
    load_transactions,
    money,
    percent,
    score,
)
from dashboard.components import (
    coverage_bar,
    donut_chart,
    kpi_row,
    line_chart,
    note,
    page_header,
    risk_legend,
    section,
)


def render() -> None:
    """Render the executive overview."""
    page_header(
        "Executive Overview",
        "Journal entry testing across the full voucher population. Every voucher is "
        "scored; the numbers below describe the triage, not a sample.",
        eyebrow="AuditLens - Financial Anomaly Detection",
    )

    frame = load_transactions()
    summary = load_summary()
    quality = load_quality()
    metrics = load_metrics()
    benford = load_benford()

    population = summary.get("population", {})
    triage = summary.get("triage", {})
    rules = summary.get("rules", {})
    ml = summary.get("machine_learning", {})

    total = int(population.get("total_transactions", len(frame)))
    total_amount = float(population.get("total_amount", frame["debit_amount"].sum()))
    flagged = int(triage.get("flagged_by_rules", int((frame["rule_alert_count"] > 0).sum())))
    high_critical = int(triage.get("high_or_critical", 0))
    high_critical_amount = float(triage.get("high_or_critical_amount", 0.0))
    average_score = float(triage.get("average_risk_score", frame["audit_risk_score"].mean()))

    # ----------------------------------------------------------------- #
    # Headline numbers
    # ----------------------------------------------------------------- #
    kpi_row(
        [
            {
                "label": "Vouchers tested",
                "value": count(total),
                "delta": "100% of the journal population",
            },
            {
                "label": "Total value",
                "value": money(total_amount),
                "delta": f"{count(population.get('unique_vendors', 0))} vendors, "
                         f"{count(population.get('unique_accounts', 0))} accounts",
            },
            {
                "label": "Flagged by rules",
                "value": percent(flagged / max(total, 1)),
                "delta": f"{count(flagged)} vouchers require triage",
                "accent": THEME["accent"],
            },
            {
                "label": "High or critical",
                "value": count(high_critical),
                "delta": f"{money(high_critical_amount)} of value",
                "accent": "#D2691E",
            },
            {
                "label": "Average risk score",
                "value": score(average_score),
                "delta": "Weighted 0-100 composite",
                "accent": THEME["muted"],
            },
        ]
    )

    st.write("")
    date_range = population.get("date_range", [])
    if len(date_range) == 2:
        st.markdown(
            f"<div class='al-caption'>Engagement period <b>{date_range[0]}</b> to "
            f"<b>{date_range[1]}</b>. Generated population - no client data.</div>",
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------- #
    # Risk distribution and monthly profile
    # ----------------------------------------------------------------- #
    section(
        "How risk is distributed",
        "A well-calibrated triage pushes almost everything into Low. If half the "
        "ledger is High, the scoring is not discriminating - it is just relabelling "
        "the population.",
    )

    left, right = st.columns([1, 1.6])
    with left:
        level_counts = triage.get("risk_level_counts") or frame["risk_level"].value_counts().to_dict()
        labels = [level for level in ("Low", "Medium", "High", "Critical") if level in level_counts]
        donut_chart(labels, [level_counts[level] for level in labels])
        coverage_bar(flagged, total, "Rule coverage")

    with right:
        monthly = (
            frame.assign(month=frame["transaction_date"].dt.to_period("M").astype(str))
            .groupby("month")
            .agg(
                vouchers=("transaction_id", "count"),
                flagged=("rule_alert_count", lambda series: int((series > 0).sum())),
            )
            .reset_index()
        )
        line_chart(
            monthly,
            x="month",
            series={"vouchers": "Vouchers posted", "flagged": "Flagged by rules"},
            height=330,
            y_title="Vouchers",
        )

    with st.expander("Risk band definitions"):
        risk_legend()

    # ----------------------------------------------------------------- #
    # Which procedures produced the findings
    # ----------------------------------------------------------------- #
    section(
        "Where the findings came from",
        "Nine independent procedures plus an unsupervised model. Each column is a "
        "different question asked of the same population.",
    )

    alerts = load_alerts()
    if not alerts.empty:
        by_rule = (
            alerts.groupby("rule_label")
            .agg(alerts=("transaction_id", "count"), mean_score=("rule_score", "mean"))
            .reset_index()
            .sort_values("alerts", ascending=False)
        )
        by_rule["share_of_ledger"] = (by_rule["alerts"] / max(total, 1)).map(percent)
        by_rule["alerts"] = by_rule["alerts"].map(count)
        by_rule["mean_score"] = by_rule["mean_score"].round(2)
        st.dataframe(
            by_rule.rename(
                columns={
                    "rule_label": "Procedure",
                    "alerts": "Alerts",
                    "mean_score": "Mean rule score",
                    "share_of_ledger": "Share of ledger",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No rule alerts were produced. Run `python src/run_pipeline.py` first.")

    # ----------------------------------------------------------------- #
    # Detector overlap
    # ----------------------------------------------------------------- #
    overlap = ml.get("anomaly_overlap") or {}
    rule_coverage = float(overlap.get("rule_coverage_pct") or 0.0)
    model_only_anomalies = int(overlap.get("caught_by_model_only") or 0)

    section(
        "How much does the model add on top of the rules?",
        "The honest answer on this benchmark is: very little. The injected anomalies "
        "were generated to match the nine procedures, so the rules already catch almost "
        "all of them. The model's case rests on patterns nobody wrote a rule for, which "
        "this benchmark cannot demonstrate - so the numbers below are reported at the "
        "anomaly level, not the flag level, and they say what they say.",
    )

    kpi_row(
        [
            {
                "label": "Anomalies the rules catch",
                "value": percent(rule_coverage / 100),
                "delta": f"{count(overlap.get('caught_by_rule', 0))} of "
                         f"{count(overlap.get('injected_anomalies', 0))} injected",
                "accent": THEME["primary"],
            },
            {
                "label": "Caught by both",
                "value": count(overlap.get("caught_by_both", 0)),
                "delta": "Rule and model agree",
            },
            {
                "label": "Model only",
                "value": count(model_only_anomalies),
                "delta": "Genuinely invisible to every rule",
                "accent": RISK_BAND_COLORS["High"],
            },
            {
                "label": "Missed by both",
                "value": count(overlap.get("caught_by_neither", 0)),
                "delta": "The residual no detector reaches",
                "accent": RISK_BAND_COLORS["Critical"],
            },
            {
                "label": "ROC-AUC",
                "value": f"{float(ml.get('roc_auc') or 0):.3f}",
                "delta": "Ranking quality vs injected anomalies",
                "accent": THEME["muted"],
            },
        ]
    )

    if overlap:
        note(
            f"The model raised <b>{count(ml.get('flagged_by_model_only', 0))}</b> vouchers "
            f"that no rule raised, but only <b>{count(model_only_anomalies)}</b> of those "
            f"were actually injected anomalies - the rest are false positives. That is the "
            f"same benchmark artefact that makes several rules show recall of 1.000: when "
            f"the answer key is generated from the rule definitions, the rules look "
            f"exhaustive. The model's <b>{float(ml.get('roc_auc') or 0):.3f}</b> ROC-AUC "
            f"and its "
            f"{percent((ml.get('precision_at_k') or {}).get('top_0.5%') or 0)} precision on "
            f"the top 0.5% show that it <i>ranks</i> well; whether it would find an "
            f"irregularity nobody wrote a rule for is a question this benchmark cannot "
            f"answer."
        )

    # ----------------------------------------------------------------- #
    # Benford and data quality
    # ----------------------------------------------------------------- #
    section("Analytical procedures and data reliability")

    first_digit = benford.get("first_digit_test", {}) or {}
    first_two = benford.get("first_two_digits_test", {}) or {}
    columns = st.columns(3)

    with columns[0]:
        st.markdown(
            f"""
            <div class='al-rule'>
                <div class='al-kpi-label'>Benford first digit</div>
                <div style='font-size:1.3rem;font-weight:700;color:{THEME['text']};'>
                    {first_digit.get('conformity', 'n/a')}
                </div>
                <div class='al-caption' style='margin-top:0.3rem;'>
                    MAD {float(first_digit.get('mad') or 0):.5f} - Chi-square
                    p = {float(first_digit.get('p_value') or 0):.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class='al-rule'>
                <div class='al-kpi-label'>Benford first two digits</div>
                <div style='font-size:1.3rem;font-weight:700;color:{THEME['text']};'>
                    {first_two.get('conformity', 'n/a')}
                </div>
                <div class='al-caption' style='margin-top:0.3rem;'>
                    MAD {float(first_two.get('mad') or 0):.5f} across
                    {count(first_two.get('n_observations', 0))} values
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[2]:
        issue_rate = float(quality.get("issue_rate_pct", 0.0))
        st.markdown(
            f"""
            <div class='al-rule'>
                <div class='al-kpi-label'>Data quality</div>
                <div style='font-size:1.3rem;font-weight:700;color:{THEME['text']};'>
                    {issue_rate:.2f}% of rows
                </div>
                <div class='al-caption' style='margin-top:0.3rem;'>
                    {count(quality.get('total_issues', 0))} issues across
                    {count(quality.get('raw_rows', 0))} raw rows
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    note(
        "<b>What Benford's Law does and does not tell you.</b> The digit distribution "
        "is consistent with expectation, which supports the completeness of the "
        "population as recorded. It does <b>not</b> prove the absence of fraud: a "
        "conforming digit profile is entirely compatible with fabricated entries, "
        "and a nonconforming one is often just a legitimately skewed account. "
        "Benford's Law is a disaggregation tool, not a verdict.",
        kind="warn",
    )

    note(
        "<b>Every figure on this page is measured against deliberately injected "
        "anomalies in a synthetic ledger.</b> The rules were tuned knowing the "
        "anomaly types; real irregularities do not announce themselves. Treat the "
        "detection rates as an upper bound and a demonstration of method, not as a "
        "claim about performance on a live engagement.",
        kind="warn",
    )

    # ----------------------------------------------------------------- #
    # Where to go next
    # ----------------------------------------------------------------- #
    section("Suggested next steps for the reviewer")
    top_vendor = summary.get("top_risk_vendor") or {}
    st.markdown(
        f"""
1. **Work the exception list first.** {count(high_critical)} vouchers sit in the
   High or Critical band. That is a reviewable population, not a haystack.
2. **Start with the vendor.** {top_vendor.get('vendor_name', 'n/a')} carries the
   highest vendor risk score ({top_vendor.get('vendor_risk_score', 0):.1f}) on
   {count(top_vendor.get('alert_count', 0))} alerts.
3. **Check the control gaps.** Self-approval and rapid-payment findings are
   segregation-of-duties issues - they are remediated by changing who can approve,
   not by recovering money.
4. **Disaggregate anything nonconforming.** The Benford page breaks the digit test
   down by account, which is where a real enquiry starts.
        """
    )
