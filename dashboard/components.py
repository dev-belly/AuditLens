"""Reusable UI components for the AuditLens dashboard.

Keeping the presentation pieces here means the six views read as analysis rather
than as HTML. It also keeps the risk-band colours and wording identical on every
page - an auditor should never have to wonder whether "High" means the same thing
on the vendor page as it does on the transaction page.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import RISK_BAND_COLORS, THEME, count, money, percent, risk_color

#: Plain-English definitions, shown next to the score so the band is never a
#: black box. An audit finding that cannot be explained is not a finding.
RISK_BAND_DEFINITIONS: dict[str, str] = {
    "Low": "No rule triggered and the model saw nothing unusual. Reviewed in aggregate only.",
    "Medium": "One indicator fired, or a moderate model score. Cleared by analytical review.",
    "High": "Multiple indicators, or a single indicator on a material amount. Warranting substantive testing.",
    "Critical": "Several independent indicators on a material amount. Escalate for immediate enquiry.",
}


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str, *, eyebrow: str | None = None) -> None:
    """Render a consistent page heading."""
    if eyebrow:
        st.markdown(
            f"<div class='al-kpi-label' style='margin-bottom:0.15rem;'>{eyebrow}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"## {title}")
    st.markdown(f"<div class='al-caption'>{subtitle}</div>", unsafe_allow_html=True)
    st.write("")


def section(title: str, description: str | None = None) -> None:
    """Render a section heading with optional explanatory copy."""
    st.markdown(f"### {title}")
    if description:
        st.markdown(f"<div class='al-caption'>{description}</div>", unsafe_allow_html=True)


def note(text: str, *, kind: str = "info") -> None:
    """Render a callout box.

    Args:
        text: Body text. Markdown is supported.
        kind: ``info`` for methodology notes, ``warn`` for limitations and
            caveats. The distinction matters: the project is explicit about what
            it cannot conclude, and that should be visually obvious.
    """
    css = "al-note al-note-warn" if kind == "warn" else "al-note al-note-info"
    st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
def kpi_card(label: str, value: str, delta: str | None = None, *, accent: str | None = None) -> None:
    """Render a single KPI card."""
    border = accent or THEME["primary"]
    delta_html = f"<div class='al-kpi-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class='al-kpi' style='border-left-color:{border};'>
            <div class='al-kpi-label'>{label}</div>
            <div class='al-kpi-value'>{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: Sequence[dict[str, Any]]) -> None:
    """Render a row of KPI cards.

    Args:
        cards: One dict per card with keys ``label``, ``value`` and optionally
            ``delta`` and ``accent``.
    """
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            kpi_card(
                card["label"],
                card["value"],
                card.get("delta"),
                accent=card.get("accent"),
            )


# --------------------------------------------------------------------------- #
# Risk badges
# --------------------------------------------------------------------------- #
def risk_badge(level: str) -> str:
    """Return an HTML badge for a risk band."""
    return (
        f"<span class='al-badge' style='background:{risk_color(level)};'>"
        f"{str(level).upper()}</span>"
    )


def risk_badge_column(frame: pd.DataFrame, column: str = "risk_level") -> pd.Series:
    """Return a Series of HTML badges, for use with ``st.dataframe``."""
    return frame[column].map(risk_badge)


def risk_legend() -> None:
    """Render the four risk bands with their definitions."""
    columns = st.columns(4)
    for column, level in zip(columns, ("Low", "Medium", "High", "Critical")):
        with column:
            st.markdown(
                f"""
                <div class='al-rule' style='border-left:4px solid {RISK_BAND_COLORS[level]};'>
                    <div style='font-weight:700;color:{RISK_BAND_COLORS[level]};'>{level}</div>
                    <div class='al-caption' style='margin-top:0.25rem;'>{RISK_BAND_DEFINITIONS[level]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Explanation rendering
# --------------------------------------------------------------------------- #
def reason_list(reasons: Iterable[str], *, limit: int | None = None) -> None:
    """Render the reasons behind a risk score as a bulleted list.

    This is the answer to the only question that matters in audit analytics:
    *why* is this voucher flagged? A ranked list with no explanation gets ignored
    by the audit team, however good the ranking is.
    """
    items = [str(item) for item in reasons if str(item).strip()]
    if not items:
        st.markdown(
            "<div class='al-caption'>No indicator fired. The voucher was scored on "
            "amount and statistical context only.</div>",
            unsafe_allow_html=True,
        )
        return

    shown = items if limit is None else items[:limit]
    for item in shown:
        st.markdown(f"- {item}")
    if limit is not None and len(items) > limit:
        st.markdown(f"<div class='al-caption'>... and {len(items) - limit} more.</div>", unsafe_allow_html=True)


def component_breakdown(components: dict[str, float], *, weights: dict[str, float] | None = None) -> None:
    """Render the weighted components of a risk score as a small bar chart.

    Args:
        components: Component name to its 0-100 contribution.
        weights: Optional component weights, shown in the labels so the reader can
            see how a 0-100 component becomes a smaller contribution.
    """
    labels = list(components.keys())
    values = [float(components[key]) for key in labels]
    if weights:
        labels = [f"{key} (weight {weights.get(key, 0):.0%})" for key in components]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=THEME["accent"],
            text=[f"{value:.0f}" for value in values],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=230,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, 105], title="Component score (0-100)"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    orientation: str = "v",
    color: str | None = None,
    colors: Sequence[str] | None = None,
    text: str | None = None,
    height: int = 340,
    x_title: str | None = None,
    y_title: str | None = None,
    horizontal: bool = False,
) -> None:
    """Render a Plotly bar chart with the project's styling."""
    marker: dict[str, Any] = {}
    if colors is not None:
        marker["color"] = list(colors)
    else:
        marker["color"] = color or THEME["primary"]

    figure = go.Figure(
        go.Bar(
            x=frame[y] if horizontal else frame[x],
            y=frame[x] if horizontal else frame[y],
            orientation="h" if horizontal else "v",
            marker=marker,
            text=frame[text] if text else None,
            textposition="outside" if horizontal else None,
        )
    )
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=30, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(gridcolor="#EDF1F4"),
        yaxis=dict(gridcolor="#EDF1F4"),
    )
    st.plotly_chart(figure, width="stretch")


def line_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    series: dict[str, str],
    height: int = 340,
    y_title: str | None = None,
) -> None:
    """Render a multi-series Plotly line chart.

    Args:
        frame: Source data.
        x: Column for the x axis.
        series: Mapping of column name to legend label.
        height: Figure height in pixels.
        y_title: Optional y axis title.
    """
    figure = go.Figure()
    palette = [THEME["primary"], THEME["accent"], "#D9A441", "#B22222", "#2E8B57"]
    for index, (column, label) in enumerate(series.items()):
        figure.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[column],
                mode="lines+markers",
                name=label,
                line=dict(color=palette[index % len(palette)], width=2),
                marker=dict(size=5),
            )
        )
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title=y_title,
        xaxis=dict(gridcolor="#EDF1F4"),
        yaxis=dict(gridcolor="#EDF1F4"),
    )
    st.plotly_chart(figure, width="stretch")


def donut_chart(labels: Sequence[str], values: Sequence[float], *, height: int = 320) -> None:
    """Render a donut chart coloured by risk band."""
    colors = [RISK_BAND_COLORS.get(str(label), THEME["muted"]) for label in labels]
    figure = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=11),
        )
    )
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=11),
    )
    st.plotly_chart(figure, width="stretch")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def transaction_table(
    frame: pd.DataFrame,
    *,
    height: int = 520,
    key: str | None = None,
) -> None:
    """Render the standard transaction grid used by the explorer and rule views."""
    display = pd.DataFrame(
        {
            "Transaction": frame["transaction_id"].astype(str),
            "Date": pd.to_datetime(frame["transaction_date"]).dt.strftime("%Y-%m-%d"),
            "Account": frame["account_code"].astype(str) + " " + frame["account_name"].astype(str),
            "Vendor": frame["vendor_name"].fillna("-").astype(str),
            "Department": frame["department"].fillna("-").astype(str),
            "Amount (CNY)": frame["debit_amount"].map(money),
            "Alerts": frame["rule_alert_count"].astype(int),
            "ML score": frame["anomaly_score"].round(3),
            "Risk score": frame["audit_risk_score"].round(1),
            "Risk": frame["risk_level"].map(risk_badge),
        }
    )
    st.dataframe(
        display,
        width="stretch",
        height=height,
        hide_index=True,
        key=key,
        column_config={
            "Risk": st.column_config.Column("Risk", help="Audit Risk Score band", width="small"),
            "Risk score": st.column_config.NumberColumn("Risk score", help="0-100 composite score", format="%.1f"),
            "ML score": st.column_config.NumberColumn(
                "ML score", help="Isolation Forest anomaly score, normalised to 0-1", format="%.3f"
            ),
            "Alerts": st.column_config.NumberColumn("Alerts", help="Number of audit rules triggered"),
        },
    )


def quality_metric_row(report: dict[str, Any]) -> None:
    """Render the data quality headline metrics."""
    total_issues = report.get("total_issues", 0)
    issue_rate = report.get("issue_rate_pct", 0.0)
    kpi_row(
        [
            {"label": "Raw rows", "value": count(report.get("raw_rows", 0))},
            {
                "label": "Clean rows",
                "value": count(report.get("clean_rows", 0)),
                "delta": f"{count(report.get('duplicates_removed', 0))} duplicate rows removed",
            },
            {
                "label": "Issues detected",
                "value": count(total_issues),
                "delta": f"{issue_rate:.2f}% of the raw population",
                "accent": RISK_BAND_COLORS["Medium"],
            },
            {
                "label": "Duplicates removed",
                "value": count(report.get("duplicates_removed", 0)),
                "delta": "Exact duplicate vouchers",
            },
        ]
    )


def coverage_bar(flagged: int, total: int, label: str = "Population flagged") -> None:
    """Render a thin progress bar showing how much of the ledger was flagged."""
    ratio = flagged / max(total, 1)
    st.markdown(
        f"""
        <div class='al-caption' style='margin-bottom:0.25rem;'>
            {label}: <b>{count(flagged)}</b> of {count(total)} vouchers ({percent(ratio)})
        </div>
        <div style='background:#EDF1F4;border-radius:4px;height:10px;width:100%;'>
            <div style='background:{THEME["accent"]};width:{min(ratio * 100, 100):.2f}%;
                        height:10px;border-radius:4px;'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
