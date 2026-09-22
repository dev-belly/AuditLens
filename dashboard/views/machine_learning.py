"""Page 5 - Machine Learning.

An unsupervised model is only defensible if three things are true, and this page is
organised around them. First, the features must be explainable in audit language -
if you cannot describe a feature to a partner, it should not be in the model.
Second, the evaluation must be honest about the fact that it is measured against
injected anomalies, which is an upper bound. Third, the operating point must be
discussed in terms of the review budget, because precision and recall are not
interesting on their own - "how many hours of work does this create?" is.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    RISK_BAND_COLORS,
    THEME,
    count,
    load_metrics,
    load_transactions,
    percent,
)
from dashboard.components import (
    bar_chart,
    kpi_row,
    note,
    page_header,
    section,
)
from src.feature_engineering import FEATURE_DESCRIPTIONS
from src.utils import as_flag_series


def _confusion_matrix_figure(matrix: list[list[int]], threshold: float | None) -> go.Figure:
    """Render the confusion matrix as an annotated heatmap."""
    frame = pd.DataFrame(
        matrix,
        index=["Actually normal", "Actually anomalous"],
        columns=["Predicted normal", "Predicted anomalous"],
    )
    figure = go.Figure(
        go.Heatmap(
            z=frame.values,
            x=frame.columns.tolist(),
            y=frame.index.tolist(),
            colorscale=[[0, "#F2F6F9"], [1, THEME["primary"]]],
            text=[[f"{value:,}" for value in row] for row in frame.values],
            texttemplate="%{text}",
            textfont=dict(size=15, color="#FFFFFF"),
            showscale=False,
            hovertemplate="%{y} / %{x}<br>%{z:,} vouchers<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=20, t=50, b=10),
        paper_bgcolor="white",
        font=dict(size=11),
        title=dict(
            text=f"Confusion matrix at contamination {threshold:.2f}" if threshold else "Confusion matrix",
            font=dict(size=13, color=THEME["primary"]),
        ),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return figure


def _budget_figure(precision_at_k: dict[str, float], recall_at_k: dict[str, float], total: int) -> go.Figure:
    """Render precision and recall against the number of vouchers reviewed."""
    budgets = list(precision_at_k.keys())
    sizes = [float(key.replace("top_", "").replace("%", "")) / 100 * total for key in budgets]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[f"{int(size):,}" for size in sizes],
            y=[precision_at_k[key] for key in budgets],
            name="Precision",
            marker_color=THEME["primary"],
            text=[f"{precision_at_k[key]:.1%}" for key in budgets],
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"{int(size):,}" for size in sizes],
            y=[recall_at_k[key] for key in budgets],
            name="Recall",
            mode="lines+markers",
            line=dict(color="#D2691E", width=2.5),
            marker=dict(size=9, symbol="diamond"),
            yaxis="y2",
        )
    )
    figure.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        title=dict(
            text="Detection performance against the review budget",
            font=dict(size=13, color=THEME["primary"]),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Vouchers reviewed (highest model score first)", gridcolor="#EDF1F4"),
        yaxis=dict(title="Precision", tickformat=".0%", gridcolor="#EDF1F4", range=[0, 1]),
        yaxis2=dict(title="Recall", tickformat=".0%", overlaying="y", side="right", range=[0, 1], showgrid=False),
        bargap=0.4,
    )
    return figure


def render() -> None:
    """Render the machine learning page."""
    page_header(
        "Machine Learning",
        "Isolation Forest over 20 explainable features. Unsupervised, because on a "
        "real engagement nobody hands you a column marked 'fraud'.",
        eyebrow="AuditLens - Anomaly detection",
    )

    metrics = load_metrics()
    if not metrics:
        st.error("Model metrics not found. Run `python src/anomaly_detection.py` first.")
        return

    evaluation = metrics.get("evaluation", {}) or {}
    hyperparameters = metrics.get("hyperparameters", {}) or {}
    frame = load_transactions()

    total = int(evaluation.get("n_samples", len(frame)))
    anomalies = int(evaluation.get("n_anomalies", 0))

    # ----------------------------------------------------------------- #
    # Headline metrics
    # ----------------------------------------------------------------- #
    kpi_row(
        [
            {"label": "Precision", "value": f"{float(evaluation.get('precision') or 0):.3f}",
             "delta": "Of the vouchers flagged, the share that were anomalies"},
            {"label": "Recall", "value": f"{float(evaluation.get('recall') or 0):.3f}",
             "delta": f"{count(evaluation.get('true_positives', 0))} of {count(anomalies)} anomalies found",
             "accent": "#D2691E"},
            {"label": "F1", "value": f"{float(evaluation.get('f1') or 0):.3f}",
             "delta": "Harmonic mean of the two"},
            {"label": "ROC-AUC", "value": f"{float(evaluation.get('roc_auc') or 0):.3f}",
             "delta": "Ranking quality - the metric that survives a budget change",
             "accent": THEME["accent"]},
            {"label": "Average precision", "value": f"{float(evaluation.get('average_precision') or 0):.3f}",
             "delta": f"vs {anomalies / max(total, 1):.3f} random baseline",
             "accent": THEME["muted"]},
        ]
    )

    st.write("")
    note(
        f"<b>These figures are measured against deliberately injected anomalies, so "
        f"they are an upper bound.</b> The injected patterns are clean and "
        f"self-consistent; real irregularities are not. The model is trained without "
        f"labels - <code>anomaly_label</code> is used only after the fact, to grade "
        f"the result. Contamination is set to "
        f"{hyperparameters.get('contamination', 0.03):.0%} as a <i>planning "
        f"assumption</i> about how much of the ledger is worth reviewing, not as a "
        f"leak from the answer key: a real engagement has no answer key, and the "
        f"model would be run exactly the same way.",
        kind="warn",
    )

    st.write("")
    note(
        "<b>Read these metrics against the rules, not in isolation.</b> The nine "
        "procedures already catch almost every injected anomaly, because the anomalies "
        "were generated to match the procedures - so a benchmark built from a rule set "
        "cannot measure what a model adds beyond that rule set. The overlap is reported "
        "on the Executive Overview at the <i>anomaly</i> level, which is the honest "
        "basis; the flag-level overlap flatters the model badly. What the numbers here "
        "do establish is that the model <b>ranks</b> well with no labels at all. "
        "Whether it would surface an irregularity nobody wrote a rule for is the "
        "question this benchmark cannot answer."
    )

    # ----------------------------------------------------------------- #
    # Confusion matrix and budget
    # ----------------------------------------------------------------- #
    section(
        "Where the model is right and where it is wrong",
        "The false negatives are the number that should worry you. The false "
        "positives are the number that will annoy your client.",
    )

    left, right = st.columns([1, 1.35])
    with left:
        matrix = evaluation.get("confusion_matrix")
        if matrix:
            st.plotly_chart(
                _confusion_matrix_figure(matrix, evaluation.get("threshold")),
                width="stretch",
            )
            true_negatives, false_positives = matrix[0]
            false_negatives, true_positives = matrix[1]
            # Compute the share from the raw counts: ``count`` returns a formatted
            # string, so dividing its result raises rather than producing a number.
            flagged_total = max(1, int(evaluation.get("n_flagged", 1)))
            false_positive_share = false_positives / flagged_total
            st.markdown(
                f"""
                <div class='al-caption'>
                    <b>{count(false_negatives)}</b> anomalies were not flagged - they
                    carry no model signal. <b>{count(false_positives)}</b> clean
                    vouchers were flagged, which is
                    {percent(false_positive_share)}
                    of the review list and the direct cost of the operating point.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        precision_at_k = evaluation.get("precision_at_k") or {}
        recall_at_k = evaluation.get("recall_at_k") or {}
        if precision_at_k:
            st.plotly_chart(
                _budget_figure(precision_at_k, recall_at_k, total),
                width="stretch",
            )

    # ----------------------------------------------------------------- #
    # The recall argument
    # ----------------------------------------------------------------- #
    section("Why the operating point is a resourcing decision, not a tuning knob")

    st.markdown(
        f"""
An audit team does not get to choose recall in the abstract. It chooses how many
vouchers it can work, and recall follows. Read the chart above from left to right:

| Vouchers reviewed | Precision | Recall | What it means in practice |
|---|---|---|---|
"""
        + "\n".join(
            f"| {int(float(key.replace('top_', '').replace('%', '')) / 100 * total):,} "
            f"| {precision_at_k[key]:.1%} | {recall_at_k[key]:.1%} "
            f"| {_budget_comment(float(key.replace('top_', '').replace('%', '')) / 100)} |"
            for key in precision_at_k
        )
        + f"""

A lower threshold lifts recall and buries the team in false positives; a higher one
cleans the list and quietly drops real findings. **Recall matters more than
precision in audit** - a missed misstatement is an audit failure, whereas a
false positive is a wasted hour. But recall is not free, and pretending otherwise
is how analytics projects lose the confidence of the people who have to work the
output. The defensible position is the one this page shows: state the budget,
state the recall it buys, and let the engagement partner decide.
        """
    )

    # ----------------------------------------------------------------- #
    # Features
    # ----------------------------------------------------------------- #
    section(
        "Features - and why each one is in the model",
        "Twenty features, all of them expressible as an audit question. Isolation "
        "Forest needs no feature scaling to work, but the scaler is applied anyway so "
        "the model matrix is reproducible and portable.",
    )

    feature_columns = metrics.get("feature_columns", []) or list(FEATURE_DESCRIPTIONS)
    feature_table = pd.DataFrame(
        {
            "Feature": feature_columns,
            "What it measures": [FEATURE_DESCRIPTIONS.get(name, "-") for name in feature_columns],
        }
    )
    st.dataframe(feature_table, width="stretch", hide_index=True, height=430)

    with st.expander("Features that were deliberately excluded"):
        st.markdown(
            """
- **Vendor master risk score.** It is a static attribute, so including it makes
  the model flag the same vendors on every run regardless of what they actually
  did. Behaviour belongs in a model; a risk rating belongs in a rule.
- **Any identifier.** Vendor ID, employee ID, account code and transaction ID are
  excluded: a model that memorises IDs does not generalise, and an auditor cannot
  act on "this vendor ID looks unusual".
- **Any ground-truth-derived column.** `anomaly_label` and `anomaly_type` are
  never inputs. Using them would make every metric on this page meaningless.
            """
        )

    # ----------------------------------------------------------------- #
    # Hyperparameters
    # ----------------------------------------------------------------- #
    section("Model configuration")
    columns = st.columns(len(hyperparameters) or 1)
    for column, (name, value) in zip(columns, hyperparameters.items()):
        with column:
            st.markdown(
                f"""
                <div class='al-rule'>
                    <div class='al-kpi-label'>{name.replace('_', ' ')}</div>
                    <div style='font-size:1.05rem;font-weight:700;'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    imputed = metrics.get("imputed_features") or {}
    if imputed:
        st.write("")
        st.markdown(
            "<div class='al-caption'><b>Imputation.</b> Four features are undefined for "
            "some vouchers - a vendor's first payment has no 'days since last "
            "transaction'. They are filled with the population median, recorded here "
            "so the model matrix can be reproduced exactly.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                {"Feature": list(imputed.keys()), "Median used": [f"{value:,.2f}" for value in imputed.values()]}
            ),
            width="stretch",
            hide_index=True,
        )

    # ----------------------------------------------------------------- #
    # Score distribution
    # ----------------------------------------------------------------- #
    section(
        "How the model scores the population",
        "Isolation Forest returns a continuous score; the flag is just a cut point on "
        "it. The distribution shows how much room there is above the cut.",
    )

    scores = frame["anomaly_score"].dropna()
    figure = go.Figure(
        go.Histogram(
            x=scores,
            nbinsx=60,
            marker_color=THEME["accent"],
        )
    )
    figure.update_layout(
        height=320,
        margin=dict(l=10, r=20, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        showlegend=False,
        xaxis=dict(title="Anomaly score (0 = typical, 1 = most isolated)", gridcolor="#EDF1F4"),
        yaxis=dict(title="Vouchers", gridcolor="#EDF1F4"),
    )
    st.plotly_chart(figure, width="stretch")

    note(
        "The score is a measure of isolation, not a probability of fraud. A voucher "
        "at 0.95 is unusual relative to this population; whether it is wrong is a "
        "question only a reviewer with the supporting documents can answer."
    )

    # ----------------------------------------------------------------- #
    # Benchmark diagnostic
    # ----------------------------------------------------------------- #
    with st.expander("Benchmark diagnostic - score separation by injected class"):
        st.markdown(
            "<div class='al-caption'>This view uses the injected labels, which would "
            "not exist on a live engagement. It is included only to show how much "
            "separation the model achieves; nothing an auditor sees elsewhere in the "
            "dashboard is conditioned on it.</div>",
            unsafe_allow_html=True,
        )
        if "anomaly_label" in frame.columns:
            # ``astype(bool)`` would read the text "0" as truthy and mark every
            # voucher as an anomaly; ``as_flag_series`` coerces through numeric.
            truth = as_flag_series(frame["anomaly_label"])
            summary = (
                frame.assign(injected=truth)
                .groupby("injected")["anomaly_score"]
                .agg(["count", "mean", "median"])
                .rename(index={False: "Normal (as injected)", True: "Anomaly (as injected)"})
            )
            summary["count"] = summary["count"].map(count)
            summary["mean"] = summary["mean"].round(3)
            summary["median"] = summary["median"].round(3)
            st.dataframe(
                summary.rename(columns={"count": "Vouchers", "mean": "Mean score", "median": "Median score"}),
                width="stretch",
            )
            separation = summary.loc["Anomaly (as injected)", "mean"] - summary.loc["Normal (as injected)", "mean"]
            st.markdown(
                f"<div class='al-caption'>Mean score separation: "
                f"<b>{separation:+.3f}</b>. A separation this modest is expected: the "
                f"model has to find a 3% minority using behaviour alone, with no "
                f"knowledge of what an anomaly looks like.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Ground-truth column not present in the loaded table.")


def _budget_comment(fraction: float) -> str:
    """Return a plain-English note for a review-budget fraction."""
    if fraction <= 0.005:
        return "Smallest defensible sample; high precision, misses most anomalies."
    if fraction <= 0.01:
        return "Realistic for a two-week fieldwork window."
    if fraction <= 0.02:
        return "Full-time analyst for the engagement - the usual compromise."
    if fraction <= 0.05:
        return "Only viable if the team is large or the client pays for it."
    return "Not a review list; this is the whole ledger with extra steps."
