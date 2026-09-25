"""Static reporting for AuditLens.

The Streamlit dashboard is interactive, but a portfolio repository also needs
images that render on GitHub and a machine-readable summary that other tools can
consume. This module produces both:

* **Charts** - PNG files in ``outputs/charts/``, rendered with matplotlib so they
  embed cleanly in the README.
* **Reports** - ``outputs/reports/audit_summary.json`` and
  ``outputs/reports/high_risk_transactions.csv``, the two artefacts an audit team
  would actually circulate.

Usage::

    python src/reporting.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/reporting.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless rendering - no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src.benford import BenfordResult
from src.utils import (
    AUDIT_SUMMARY_REPORT,
    BENFORD_RESULTS_JSON,
    CHART_DIR,
    DATA_QUALITY_REPORT,
    HIGH_RISK_CSV,
    MODEL_METRICS_JSON,
    RISK_COLORS,
    RISK_LEVEL_ORDER,
    SCORED_TRANSACTIONS,
    THEME_ACCENT,
    THEME_MUTED,
    THEME_PRIMARY,
    Timer,
    VENDOR_RISK_TABLE,
    ensure_directories,
    format_cny,
    get_logger,
    load_dataframe,
    load_json,
    save_json,
)

LOGGER = get_logger(__name__)

#: Risk bands below which a voucher is not worth circulating as a finding.
HIGH_RISK_LEVELS: tuple[str, ...] = ("High", "Critical")

#: Consistent matplotlib styling so every chart in the README looks like part of
#: the same deliverable.
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D5DBE1",
        "axes.labelcolor": "#2E3A46",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": THEME_PRIMARY,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#E8ECF0",
        "grid.linewidth": 0.8,
        "xtick.color": "#5A6B7B",
        "ytick.color": "#5A6B7B",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    }
)

#: CJK-capable fonts, in preference order. The chart of high-risk accounts is
#: labelled with the account names from the chart of accounts, which are Chinese
#: (the ledger follows 企业会计准则). Without a CJK font matplotlib renders those
#: labels as empty boxes, which would make the README images look broken.
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Hiragino Sans GB",      # macOS
    "PingFang SC",           # macOS 10.11+
    "Heiti SC",              # macOS
    "Songti SC",             # macOS
    "Arial Unicode MS",      # macOS / Windows (bundled with Office)
    "Microsoft YaHei",       # Windows
    "SimHei",                # Windows
    "Noto Sans CJK SC",      # Linux
    "Source Han Sans SC",    # Linux
    "WenQuanYi Zen Hei",     # Linux
)


def configure_fonts() -> str:
    """Select the first installed CJK-capable font and report the choice.

    Returns:
        The font family name matplotlib will use for text.
    """
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_FONT_CANDIDATES if name in available), None)

    if chosen is None:
        LOGGER.warning(
            "No CJK font found; Chinese account labels will render as boxes. "
            "Install one of: %s",
            ", ".join(_CJK_FONT_CANDIDATES[:4]),
        )
        return plt.rcParams["font.family"][0] if isinstance(plt.rcParams["font.family"], list) else "sans-serif"

    # Put the CJK font first and keep a Latin fallback for the digits and units.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # the CJK glyph set lacks U+2212
    LOGGER.info("Chart font: %s", chosen)
    return chosen


CJK_FONT: str = configure_fonts()


def _finish(fig: plt.Figure, path: Path) -> Path:
    """Tighten the layout, save the figure and close it."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path.name)
    return path


# --------------------------------------------------------------------------- #
# Individual charts
# --------------------------------------------------------------------------- #
def chart_risk_level_distribution(df: pd.DataFrame) -> Path:
    """Bar chart of voucher counts and value by risk band."""
    counts = df["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER, fill_value=0)
    amounts = df.groupby("risk_level")["debit_amount"].sum().reindex(RISK_LEVEL_ORDER, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = [RISK_COLORS[level] for level in RISK_LEVEL_ORDER]

    axes[0].bar(RISK_LEVEL_ORDER, counts.to_numpy(), color=colors)
    axes[0].set_title("Vouchers by risk level")
    axes[0].set_ylabel("Vouchers")
    for index, value in enumerate(counts.to_numpy()):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, counts.max() * 1.18)

    axes[1].bar(RISK_LEVEL_ORDER, amounts.to_numpy() / 1e6, color=colors)
    axes[1].set_title("Value at risk by band")
    axes[1].set_ylabel("CNY millions")
    for index, value in enumerate(amounts.to_numpy() / 1e6):
        axes[1].text(index, value, f"{value:,.0f}M", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, amounts.max() / 1e6 * 1.18)

    return _finish(fig, CHART_DIR / "risk_level_distribution.png")


def chart_risk_score_distribution(df: pd.DataFrame) -> Path:
    """Histogram of the Audit Risk Score with the band boundaries marked."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["audit_risk_score"], bins=80, color=THEME_ACCENT, edgecolor="white", linewidth=0.3)
    for boundary, label in ((30, "Medium"), (60, "High"), (80, "Critical")):
        ax.axvline(boundary, color=THEME_PRIMARY, linestyle="--", linewidth=1.0)
        ax.text(boundary, ax.get_ylim()[1] * 0.92, f" {label}", color=THEME_PRIMARY, fontsize=9)
    ax.set_title("Distribution of the Audit Risk Score")
    ax.set_xlabel("Audit Risk Score (0-100)")
    ax.set_ylabel("Vouchers")
    ax.set_xlim(0, 100)
    return _finish(fig, CHART_DIR / "risk_score_distribution.png")


def chart_monthly_trend(df: pd.DataFrame) -> Path:
    """Monthly value and flagged-voucher trend."""
    monthly = (
        df.groupby("year_month")
        .agg(total_amount=("debit_amount", "sum"), flagged=("rule_alert_count", lambda s: int((s > 0).sum())))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly["year_month"], monthly["total_amount"] / 1e6, color=THEME_ACCENT, label="Total value (CNY m)")
    ax.set_ylabel("CNY millions")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)

    ax2 = ax.twinx()
    ax2.plot(monthly["year_month"], monthly["flagged"], color=RISK_COLORS["Critical"], marker="o", linewidth=1.6, label="Flagged vouchers")
    ax2.set_ylabel("Flagged vouchers", color=RISK_COLORS["Critical"])
    ax2.grid(False)

    ax.set_title("Monthly transaction value and flagged vouchers")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    return _finish(fig, CHART_DIR / "monthly_transaction_trend.png")


def chart_benford(result: BenfordResult) -> Path:
    """Observed versus expected first-digit distribution."""
    table = result.table
    positions = np.arange(len(table))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(positions - width / 2, table["observed_frequency"], width, label="Observed", color=THEME_ACCENT)
    ax.bar(positions + width / 2, table["expected_frequency"], width, label="Benford expected", color=THEME_MUTED)

    ax.set_xticks(positions)
    ax.set_xticklabels(table["digit"].astype(int).astype(str))
    ax.set_xlabel("Leading digit")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Benford first-digit test - {result.conformity} "
        f"(MAD {result.mad:.4f}, chi-square {result.chi_square:,.0f}, n={result.n_observations:,})"
    )
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        -0.24,
        "A deviation identifies a population worth disaggregating. It is not evidence of fraud.",
        transform=ax.transAxes,
        fontsize=8,
        color="#6B7A88",
        style="italic",
    )
    return _finish(fig, CHART_DIR / "benford_observed_vs_expected.png")


def _truncate_label(text: str, limit: int = 36) -> str:
    """Shorten a chart label on a word boundary.

    A hard slice produces labels like ``Accumulated depreciat``, which reads as a
    typo rather than an abbreviation. Cutting at the last space and appending an
    ellipsis keeps the label honest.
    """
    text = str(text)
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + "\u2026"


def chart_top_risk_vendors(vendor_risk: pd.DataFrame, top_n: int = 12) -> Path:
    """Horizontal bar chart of the highest-risk vendors."""
    top = vendor_risk.head(top_n).iloc[::-1]
    labels = [_truncate_label(name) for name in top["vendor_name"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = [RISK_COLORS.get(level, THEME_MUTED) for level in top["vendor_risk_level"]]
    ax.barh(labels, top["vendor_risk_score"], color=colors)
    ax.set_xlabel("Vendor risk score (0-100)")
    ax.set_title(f"Top {len(top)} vendors by risk score")
    ax.set_xlim(0, 100)
    for index, (score, alerts) in enumerate(zip(top["vendor_risk_score"], top["alert_count"])):
        ax.text(score + 1, index, f"{score:.0f}  ({int(alerts)} alerts)", va="center", fontsize=8)
    return _finish(fig, CHART_DIR / "top_risk_vendors.png")


def chart_top_risk_accounts(df: pd.DataFrame, top_n: int = 12) -> Path:
    """Horizontal bar chart of the riskiest accounts."""
    grouped = (
        df.groupby(["account_code", "account_name"])
        .agg(mean_risk=("audit_risk_score", "mean"), total_amount=("debit_amount", "sum"), count=("transaction_id", "count"))
        .reset_index()
        .sort_values("mean_risk", ascending=False)
        .head(top_n)
    )
    grouped["label"] = (
        grouped["account_code"].astype(str)
        + "  "
        + grouped["account_name"].astype(str).map(_truncate_label)
    )
    grouped = grouped.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(grouped["label"], grouped["mean_risk"], color=THEME_PRIMARY)
    ax.set_xlabel("Average Audit Risk Score")
    ax.set_title(f"Top {len(grouped)} accounts by average risk score")
    ax.set_xlim(0, max(60.0, float(grouped["mean_risk"].max()) * 1.25))
    for index, (score, count) in enumerate(zip(grouped["mean_risk"], grouped["count"])):
        ax.text(score + 0.5, index, f"{score:.1f}  (n={int(count):,})", va="center", fontsize=8)
    return _finish(fig, CHART_DIR / "top_risk_accounts.png")


def chart_rule_alert_summary(alerts: pd.DataFrame, total_transactions: int) -> Path:
    """How much of the ledger each rule sweeps up."""
    grouped = (
        alerts.groupby("rule_label")
        .agg(alert_count=("transaction_id", "count"), average_score=("rule_score", "mean"))
        .reset_index()
        .sort_values("alert_count", ascending=True)
    )
    grouped["pct_of_ledger"] = grouped["alert_count"] / max(total_transactions, 1) * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(grouped["rule_label"], grouped["pct_of_ledger"], color=THEME_ACCENT)
    ax.set_xlabel("% of the ledger flagged")
    ax.set_title("Alert volume by audit rule")
    for index, (pct, count) in enumerate(zip(grouped["pct_of_ledger"], grouped["alert_count"])):
        ax.text(pct + 0.02, index, f"{pct:.2f}%  ({count:,})", va="center", fontsize=8)
    ax.set_xlim(0, max(grouped["pct_of_ledger"].max() * 1.35, 1.0))
    return _finish(fig, CHART_DIR / "rule_alert_summary.png")


def chart_confusion_matrix(metrics: dict[str, Any]) -> Path | None:
    """Confusion matrix heatmap for the Isolation Forest."""
    evaluation = (metrics or {}).get("evaluation")
    if not evaluation:
        return None

    matrix = np.array(evaluation["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.1)
    ax.set_xticks([0, 1], labels=["Predicted normal", "Predicted anomaly"])
    ax.set_yticks([0, 1], labels=["Actual normal", "Actual anomaly"])
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            ax.text(
                column, row, f"{int(value):,}",
                ha="center", va="center", fontsize=13,
                color="white" if value > matrix.max() * 0.55 else "#1F4E79",
            )
    ax.set_title(
        "Isolation Forest confusion matrix\n"
        f"precision {evaluation['precision']:.3f}  recall {evaluation['recall']:.3f}  F1 {evaluation['f1']:.3f}",
        fontsize=11,
    )
    ax.grid(False)
    return _finish(fig, CHART_DIR / "ml_confusion_matrix.png")


def chart_precision_at_k(metrics: dict[str, Any]) -> Path | None:
    """Precision and recall as a function of the review budget."""
    evaluation = (metrics or {}).get("evaluation")
    if not evaluation or not evaluation.get("precision_at_k"):
        return None

    budgets = list(evaluation["precision_at_k"].keys())
    precision = [evaluation["precision_at_k"][key] * 100 for key in budgets]
    recall = [evaluation["recall_at_k"][key] * 100 for key in budgets]
    x = np.arange(len(budgets))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(x, precision, marker="o", color=THEME_PRIMARY, linewidth=1.8, label="Precision")
    ax.plot(x, recall, marker="s", color=RISK_COLORS["High"], linewidth=1.8, label="Recall")

    base_rate = evaluation["n_anomalies"] / max(evaluation["n_samples"], 1) * 100
    ax.axhline(base_rate, color=THEME_MUTED, linestyle="--", linewidth=1.2)
    ax.text(len(budgets) - 1, base_rate + 1.2, f"base rate {base_rate:.2f}%", ha="right", fontsize=8, color="#6B7A88")

    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.set_xlabel("Review budget (share of ledger an auditor can examine)")
    ax.set_ylabel("%")
    ax.set_title("Precision and recall against the review budget")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(precision + recall) * 1.25)
    return _finish(fig, CHART_DIR / "ml_precision_at_k.png")


def chart_component_contributions(df: pd.DataFrame) -> Path:
    """Average contribution of each risk component, split by risk band."""
    components = {
        "Rule": "risk_component_rule",
        "Machine learning": "risk_component_ml",
        "Vendor": "risk_component_vendor",
        "Amount": "risk_component_amount",
        "Statistical": "risk_component_statistical",
    }
    available = {name: column for name, column in components.items() if column in df.columns}

    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.16
    positions = np.arange(len(RISK_LEVEL_ORDER))
    palette = [THEME_PRIMARY, THEME_ACCENT, RISK_COLORS["Medium"], RISK_COLORS["High"], THEME_MUTED]

    for offset, (name, column) in enumerate(available.items()):
        means = [
            float(df.loc[df["risk_level"] == level, column].mean() or 0.0)
            for level in RISK_LEVEL_ORDER
        ]
        ax.bar(positions + (offset - len(available) / 2 + 0.5) * width, means, width, label=name, color=palette[offset % len(palette)])

    ax.set_xticks(positions)
    ax.set_xticklabels(RISK_LEVEL_ORDER)
    ax.set_ylabel("Mean normalised component (0-1)")
    ax.set_title("What drives risk, by band")
    ax.legend(ncol=5, loc="upper left")
    return _finish(fig, CHART_DIR / "risk_component_contributions.png")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def write_high_risk_extract(df: pd.DataFrame) -> Path:
    """Write the review-ready extract of High and Critical vouchers."""
    columns = [
        column
        for column in (
            "transaction_id", "transaction_date", "posting_date", "vendor_id", "vendor_name",
            "account_code", "account_name", "debit_amount", "department", "description",
            "created_by", "approved_by", "audit_risk_score", "risk_level", "rule_alert_count",
            "anomaly_score", "ml_anomaly_flag", "risk_reason_text",
        )
        if column in df.columns
    ]
    extract = df.loc[df["risk_level"].isin(HIGH_RISK_LEVELS), columns].sort_values(
        "audit_risk_score", ascending=False
    )
    extract.to_csv(HIGH_RISK_CSV, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote %s (%s vouchers)", HIGH_RISK_CSV.name, len(extract))
    return HIGH_RISK_CSV


def detector_overlap(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, Any]]:
    """Measure how much the rule engine and the model overlap.

    The two detectors answer different questions, and the overlap has to be measured
    on **two different bases**, because conflating them is the easiest mistake to make
    with this project's output:

    * **Flag level** - how many *vouchers* each detector raised. This is what an
      auditor's workload is made of.
    * **Anomaly level** - how many *injected anomalies* each detector actually caught.
      This is the honest basis for asking whether the model adds anything.

    They are very different numbers. On this benchmark the rules catch 98% of the
    injected anomalies, so the vouchers the model raised that no rule raised are almost
    entirely false positives. An earlier version of the summary reported only the
    flag-level overlap under the name ``detections_not_caught_by_rules``, which reads as
    "anomalies the rules missed" - and is not what it measured.

    Args:
        df: Scored transaction table, with ``rule_alert_count`` and
            ``ml_anomaly_flag``. ``anomaly_type`` is used when present.

    Returns:
        ``(flag_overlap, anomaly_overlap)``. ``anomaly_overlap`` is empty when the
        ground-truth column is absent, since the comparison is undefined without it.
    """
    rule_flagged = set(df.loc[df["rule_alert_count"] > 0, "transaction_id"])
    model_flagged = set(df.loc[df["ml_anomaly_flag"].astype(bool), "transaction_id"])

    flag_overlap = {
        "flagged_by_rule": len(rule_flagged),
        "flagged_by_model": len(model_flagged),
        "flagged_by_both": len(rule_flagged & model_flagged),
        "flagged_by_model_only": len(model_flagged - rule_flagged),
        "flagged_by_rule_only": len(rule_flagged - model_flagged),
    }

    if "anomaly_type" not in df.columns:
        return flag_overlap, {}

    truth = df["anomaly_type"].astype("string").notna()
    total_anomalies = int(truth.sum())
    rule_anomalies = set(df.loc[truth & df["rule_alert_count"].gt(0), "transaction_id"])
    model_anomalies = set(df.loc[truth & df["ml_anomaly_flag"].astype(bool), "transaction_id"])

    anomaly_overlap: dict[str, Any] = {
        "injected_anomalies": total_anomalies,
        "caught_by_rule": len(rule_anomalies),
        "caught_by_model": len(model_anomalies),
        "caught_by_both": len(rule_anomalies & model_anomalies),
        "caught_by_model_only": len(model_anomalies - rule_anomalies),
        "caught_by_rule_only": len(rule_anomalies - model_anomalies),
        "caught_by_neither": total_anomalies - len(rule_anomalies | model_anomalies),
        "rule_coverage_pct": round(len(rule_anomalies) / max(total_anomalies, 1) * 100, 2),
    }
    return flag_overlap, anomaly_overlap


def build_audit_summary(
    df: pd.DataFrame,
    vendor_risk: pd.DataFrame,
    metrics: dict[str, Any],
    alerts: pd.DataFrame,
    quality_report: dict[str, Any],
    benford: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the headline engagement summary."""
    total = len(df)
    total_amount = float(df["debit_amount"].sum())
    flagged = int((df["rule_alert_count"] > 0).sum())
    high = int(df["risk_level"].isin(HIGH_RISK_LEVELS).sum())

    level_counts = df["risk_level"].value_counts().to_dict()
    level_amounts = df.groupby("risk_level")["debit_amount"].sum().to_dict()

    by_rule = (
        alerts.groupby("rule_label")
        .agg(alerts=("transaction_id", "count"), amount=("transaction_id", "size"))
        .sort_values("alerts", ascending=False)
    )
    rule_amounts = (
        alerts.merge(df[["transaction_id", "debit_amount"]], on="transaction_id", how="left")
        .groupby("rule_label")["debit_amount"]
        .sum()
    )

    evaluation = (metrics or {}).get("evaluation") or {}

    flag_overlap, anomaly_overlap = detector_overlap(df)

    top_vendor = vendor_risk.iloc[0] if not vendor_risk.empty else None

    summary = {
        "population": {
            "total_transactions": total,
            "total_amount": round(total_amount, 2),
            "date_range": [
                str(pd.to_datetime(df["transaction_date"]).min().date()),
                str(pd.to_datetime(df["transaction_date"]).max().date()),
            ],
            "unique_vendors": int(df["vendor_id"].nunique()),
            "unique_accounts": int(df["account_code"].nunique()),
            # Named for what it measures. ``created_by``, not the employee master
            # file, so this is smaller than the ``employees`` table row count and
            # is not a contradiction of it.
            "unique_voucher_creators": int(df["created_by"].nunique()),
        },
        "triage": {
            "flagged_by_rules": flagged,
            "flagged_by_rules_pct": round(flagged / max(total, 1) * 100, 3),
            "high_or_critical": high,
            "high_or_critical_pct": round(high / max(total, 1) * 100, 3),
            "high_or_critical_amount": round(
                float(df.loc[df["risk_level"].isin(HIGH_RISK_LEVELS), "debit_amount"].sum()), 2
            ),
            "average_risk_score": round(float(df["audit_risk_score"].mean()), 2),
            "risk_level_counts": {level: int(level_counts.get(level, 0)) for level in RISK_LEVEL_ORDER},
            "risk_level_amounts": {level: round(float(level_amounts.get(level, 0.0)), 2) for level in RISK_LEVEL_ORDER},
        },
        "rules": {
            "total_alerts": int(len(alerts)),
            "distinct_rules_triggered": int(alerts["rule_key"].nunique()) if not alerts.empty else 0,
            "alerts_per_rule": {label: int(count) for label, count in by_rule["alerts"].items()},
            "amount_per_rule": {label: round(float(value), 2) for label, value in rule_amounts.items()},
        },
        "machine_learning": {
            "model": (metrics or {}).get("model", "IsolationForest"),
            "flagged": int(df["ml_anomaly_flag"].astype(bool).sum()),
            "precision": evaluation.get("precision"),
            "recall": evaluation.get("recall"),
            "f1": evaluation.get("f1"),
            "roc_auc": evaluation.get("roc_auc"),
            "average_precision": evaluation.get("average_precision"),
            "precision_at_k": evaluation.get("precision_at_k"),
            "recall_at_k": evaluation.get("recall_at_k"),
            # Flag-level overlap: how many vouchers each detector raised.
            "flagged_by_model_only": flag_overlap["flagged_by_model_only"],
            "flagged_by_both": flag_overlap["flagged_by_both"],
            # Anomaly-level overlap: how many injected anomalies each detector
            # actually caught. This is the honest basis for comparing the two.
            "anomaly_overlap": anomaly_overlap,
        },
        "benford": {
            "first_digit": benford.get("first_digit_test"),
            "first_two_digits": benford.get("first_two_digits_test"),
        },
        "data_quality": quality_report,
        "top_risk_vendor": (
            {
                "vendor_id": str(top_vendor["vendor_id"]),
                "vendor_name": str(top_vendor["vendor_name"]),
                "vendor_risk_score": float(top_vendor["vendor_risk_score"]),
                "alert_count": int(top_vendor["alert_count"]),
            }
            if top_vendor is not None
            else None
        ),
    }
    save_json(summary, AUDIT_SUMMARY_REPORT)
    LOGGER.info("Wrote %s", AUDIT_SUMMARY_REPORT.name)
    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_reporting(
    df: pd.DataFrame | None = None,
    vendor_risk: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Generate every chart and report.

    Args:
        df: Scored transaction table. Loaded from disk when omitted.
        vendor_risk: Vendor risk table. Loaded from disk when omitted.
        alerts: Long-format rule alerts. Loaded from disk when omitted.

    Returns:
        Mapping with ``summary``, ``charts`` and ``high_risk_extract``.
    """
    ensure_directories()

    if df is None:
        df = load_dataframe(SCORED_TRANSACTIONS)
    if vendor_risk is None:
        vendor_risk = load_dataframe(VENDOR_RISK_TABLE)
    if alerts is None:
        from src.utils import RULE_ALERTS_CSV

        alerts = pd.read_csv(RULE_ALERTS_CSV) if RULE_ALERTS_CSV.exists() else pd.DataFrame(
            columns=["transaction_id", "rule_key", "rule_label", "rule_score", "risk_reason"]
        )

    metrics = load_json(MODEL_METRICS_JSON) if MODEL_METRICS_JSON.exists() else {}
    quality = load_json(DATA_QUALITY_REPORT) if DATA_QUALITY_REPORT.exists() else {}
    benford = load_json(BENFORD_RESULTS_JSON) if BENFORD_RESULTS_JSON.exists() else {}

    # Rebuild the Benford result object so the chart can use the digit table.
    from src.benford import first_digit_test

    benford_result = first_digit_test(df["debit_amount"])

    charts: list[Path] = [
        chart_risk_level_distribution(df),
        chart_risk_score_distribution(df),
        chart_monthly_trend(df),
        chart_benford(benford_result),
        chart_top_risk_vendors(vendor_risk),
        chart_top_risk_accounts(df),
        chart_rule_alert_summary(alerts, len(df)),
        chart_component_contributions(df),
    ]
    for optional in (chart_confusion_matrix(metrics), chart_precision_at_k(metrics)):
        if optional is not None:
            charts.append(optional)

    extract = write_high_risk_extract(df)
    summary = build_audit_summary(df, vendor_risk, metrics, alerts, quality, benford)

    return {"summary": summary, "charts": charts, "high_risk_extract": extract}


def main() -> int:
    """CLI entry point."""
    with Timer("reporting"):
        result = run_reporting()

    summary = result["summary"]
    LOGGER.info("--- Engagement summary ---")
    LOGGER.info("  vouchers                %s", f"{summary['population']['total_transactions']:,}")
    LOGGER.info("  total value             %s", format_cny(summary['population']['total_amount']))
    LOGGER.info("  flagged by rules        %s (%.2f%%)",
                f"{summary['triage']['flagged_by_rules']:,}", summary['triage']['flagged_by_rules_pct'])
    LOGGER.info("  high or critical        %s (%.2f%%)",
                f"{summary['triage']['high_or_critical']:,}", summary['triage']['high_or_critical_pct'])
    LOGGER.info("  charts written          %s", len(result["charts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
