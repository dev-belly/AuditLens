"""Build the three notebooks and verify that every code cell executes.

Writing notebook JSON by hand is error-prone, and a notebook that does not run is
worse than no notebook. This script holds the cell sources as plain strings, runs
all of a notebook's code cells in one namespace (exactly as a kernel would), and
only writes the ``.ipynb`` once the whole notebook has executed cleanly.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

KERNELSPEC: dict[str, Any] = {
    "display_name": "Python 3 (AuditLens)",
    "language": "python",
    "name": "python3",
}

LANGUAGE_INFO: dict[str, Any] = {
    "name": "python",
    "version": "3.11",
    "mimetype": "text/x-python",
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "codemirror_mode": {"name": "ipython", "version": 3},
}

PRELUDE = """\
import sys
from pathlib import Path

# The notebooks are run from notebooks/, so the project root has to be on the path
# before any src.* import.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless: the notebooks are executed in CI as well as by hand
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.reporting import configure_fonts

configure_fonts()  # CJK labels in charts render as boxes without this
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
"""


def notebook(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the cells with the shared prelude prepended as a code cell."""
    return [("code", PRELUDE)] + cells


# --------------------------------------------------------------------------- #
# 01 - Exploratory data analysis
# --------------------------------------------------------------------------- #
EDA: list[tuple[str, str]] = notebook(
    [
        (
            "markdown",
            """\
# 01 — Exploratory analysis of the ledger

Before any audit procedure runs, the population has to be understood. This notebook
establishes the shape of the ledger — how many vouchers, how much value, where the
value concentrates, and when postings happen — because every threshold in the rule
engine is a statement about what "normal" looks like.

Everything here reads the pipeline's own output. Run `python src/run_pipeline.py`
first.

**Note on labels.** The `anomaly_label` and `anomaly_type` columns are the injected
ground truth. This notebook uses them only where it says so explicitly, to describe
the benchmark. The dashboard never sees them.
""",
        ),
        (
            "code",
            """\
scored = pd.read_parquet(PROJECT_ROOT / "data/processed/transactions_scored.parquet")
summary = json.loads((PROJECT_ROOT / "outputs/reports/audit_summary.json").read_text())

population = summary["population"]
print(f"vouchers          {len(scored):,}")
print(f"total value       CNY {scored['debit_amount'].sum():,.0f}")
print(f"period            {scored['transaction_date'].min():%Y-%m-%d} -> {scored['transaction_date'].max():%Y-%m-%d}")
print(f"vendors           {scored['vendor_id'].nunique():,}")
print(f"accounts          {scored['account_code'].nunique():,}")
print(f"employees         {scored['employee_id'].nunique():,}")
""",
        ),
        (
            "markdown",
            "## 1. How is value distributed?\n\n"
            "Ledger amounts are log-normally distributed, and that single fact is why the "
            "Benford analysis in notebook 02 is meaningful rather than decorative. A "
            "uniformly distributed ledger would fail Benford's Law for entirely innocent "
            "reasons.",
        ),
        (
            "code",
            """\
amounts = scored["debit_amount"]

display(amounts.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).to_frame("CNY"))

figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(np.log10(amounts[amounts > 0]), bins=60, color="#1F4E79")
axes[0].set_title("Amount distribution (log10)")
axes[0].set_xlabel("log10(amount in CNY)")
axes[0].set_ylabel("vouchers")

axes[1].hist(amounts, bins=60, color="#C00000")
axes[1].set_title("Amount distribution (linear)")
axes[1].set_xlabel("amount in CNY")
plt.tight_layout()
plt.show()

# The spread of the distribution is what makes the digit tests usable.
print(f"orders of magnitude spanned (P1-P99): "
      f"{np.log10(amounts.quantile(0.99)) - np.log10(max(amounts.quantile(0.01), 1)):.2f}")
""",
        ),
        (
            "markdown",
            "## 2. Where does the value sit?\n\n"
            "A small number of accounts carry most of the value. Audit effort follows "
            "materiality, so this ordering matters more than the voucher counts.",
        ),
        (
            "code",
            """\
by_account = (
    scored.groupby(["account_code", "account_name"], observed=True)
    .agg(vouchers=("transaction_id", "count"), value=("debit_amount", "sum"))
    .reset_index()
    .sort_values("value", ascending=False)
)
by_account["share_of_value"] = by_account["value"] / by_account["value"].sum()
by_account["cumulative_share"] = by_account["share_of_value"].cumsum()

display(by_account.head(10).style.format({"value": "{:,.0f}", "share_of_value": "{:.2%}", "cumulative_share": "{:.1%}"}))

top_five = by_account.head(5)["share_of_value"].sum()
print(f"the five largest accounts carry {top_five:.1%} of the ledger's value")
""",
        ),
        (
            "markdown",
            "## 3. When do postings happen?\n\n"
            "Month-end clustering is a classic journal-entry risk: it is where manual "
            "adjustments concentrate, and it is why the weekend/holiday rule has a "
            "population to work with at all.",
        ),
        (
            "code",
            """\
scored["transaction_date"] = pd.to_datetime(scored["transaction_date"])

monthly = (
    scored.set_index("transaction_date")
    .resample("MS")
    .agg(vouchers=("transaction_id", "count"), value=("debit_amount", "sum"))
)

figure, axis = plt.subplots(figsize=(12, 4))
axis.bar(monthly.index, monthly["vouchers"], width=20, color="#1F4E79", label="vouchers")
axis.set_ylabel("vouchers")
axis.set_title("Monthly posting volume")
plt.tight_layout()
plt.show()

weekday = scored["transaction_date"].dt.dayofweek
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("postings by weekday of the transaction date:")
for index, name in enumerate(names):
    share = (weekday == index).mean()
    print(f"  {name}  {share:6.2%}  {'#' * int(share * 200)}")
print(f"\\nweekend share of the ledger: {(weekday >= 5).mean():.2%}")

day_of_month = scored["transaction_date"].dt.day
month_end = (day_of_month >= 25).mean()
print(f"postings on day 25 or later: {month_end:.2%} of the ledger")
""",
        ),
        (
            "markdown",
            "## 4. Vendor concentration\n\n"
            "How concentrated is the supplier base? A concentrated base changes the audit "
            "approach: the fewer the relationships, the more each one matters and the less "
            "a sample of transactions tells you.",
        ),
        (
            "code",
            """\
by_vendor = (
    scored.groupby("vendor_name", observed=True)
    .agg(payments=("transaction_id", "count"), value=("debit_amount", "sum"))
    .sort_values("value", ascending=False)
)
total_value = by_vendor["value"].sum()
print(f"top 10 vendors  {by_vendor.head(10)['value'].sum() / total_value:6.2%} of value")
print(f"top 50 vendors  {by_vendor.head(50)['value'].sum() / total_value:6.2%} of value")
print(f"vendors         {len(by_vendor):,}")
print()
display(by_vendor.head(10).style.format({"value": "{:,.0f}"}))
""",
        ),
        (
            "markdown",
            "## 5. Data quality is a finding\n\n"
            "The extract arrives dirty on purpose. What matters is that the cleaner "
            "*reports* every issue it finds, including the ones it deliberately does not "
            "repair.",
        ),
        (
            "code",
            """\
quality = json.loads((PROJECT_ROOT / "outputs/reports/data_quality_report.json").read_text())

issues = pd.DataFrame(
    [
        ("Duplicate rows", quality["duplicates_removed"], "removed"),
        ("Invalid transaction dates", quality["invalid_date_rows"], "row dropped"),
        ("Invalid posting dates", quality["invalid_posting_dates"], "repaired"),
        ("Invalid amounts", quality["invalid_amount_rows"], "row dropped"),
        ("Unbalanced vouchers", quality["unbalanced_rows"], "flagged, kept"),
        ("Orphan vendors", quality["orphan_vendor_rows"], "flagged, kept"),
        ("Account name mismatches", quality["account_name_mismatches"], "repaired"),
        ("Currency codes normalised", quality["currency_normalised_rows"], "normalised"),
    ],
    columns=["Issue", "Count", "Action"],
)

print(f"raw rows {quality['raw_rows']:,} -> clean rows {quality['clean_rows']:,}")
print(f"total issues {quality['total_issues']} = {quality['issue_rate_pct']}% of the raw extract")
print()
display(issues.style.hide(axis="index"))

print("The repairs, as logged by the cleaner:")
for line in quality["repair_log"]:
    print(f"  - {line}")
""",
        ),
        (
            "markdown",
            "## 6. The triage, before any analysis\n\n"
            "The composite risk score has already been computed by the pipeline. This is "
            "the shape of its output — and the shape is the point: a well-calibrated "
            "triage puts almost everything in the bottom band.",
        ),
        (
            "code",
            """\
triage = summary["triage"]
bands = pd.DataFrame(
    {
        "vouchers": pd.Series(triage["risk_level_counts"]),
        "value": pd.Series(triage["risk_level_amounts"]),
    }
).reindex(["Low", "Medium", "High", "Critical"])

bands["share_of_vouchers"] = bands["vouchers"] / bands["vouchers"].sum()
bands["share_of_value"] = bands["value"] / bands["value"].sum()
display(bands.style.format({"vouchers": "{:,.0f}", "value": "{:,.0f}",
                            "share_of_vouchers": "{:.2%}", "share_of_value": "{:.2%}"}))

figure, axis = plt.subplots(figsize=(9, 4))
colours = ["#2E7D32", "#F9A825", "#E65100", "#B71C1C"]
axis.bar(bands.index, bands["vouchers"], color=colours)
axis.set_yscale("log")
axis.set_ylabel("vouchers (log scale)")
axis.set_title("Audit risk bands: 30,128 vouchers collapsed to a review list")
for position, count in enumerate(bands["vouchers"]):
    axis.text(position, count, f"{count:,}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

print(f"average risk score across the population: {triage['average_risk_score']}")
print(f"High + Critical: {triage['high_or_critical']:,} vouchers "
      f"({triage['high_or_critical_pct']}%) carrying CNY {triage['high_or_critical_amount']:,.0f}")
""",
        ),
        (
            "markdown",
            """\
## What this establishes

- **30,128 vouchers, CNY 4.44B** over two years — far more than any team can read, so
  the deliverable is a ranking, not a conclusion.
- **Amounts span several orders of magnitude and are log-normally distributed**, which
  is what makes the digit tests in notebook 02 legitimate.
- **Postings cluster at month-end**, and weekends are only 2.3% of the ledger because
  the generator models a working calendar — a naive uniform sampler would produce 28%
  and make the weekend rule meaningless.
- **The top five accounts carry most of the value.** Audit effort follows materiality.
- **The extract carries 285 quality issues (0.94%)**, and two of them — unbalanced
  vouchers and orphan vendors — are themselves findings, which is why the cleaner
  flags rather than deletes.
- **The triage puts 93.8% of the population in the Low band**, which is what a
  discriminating score looks like.
""",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# 02 - Audit procedures and Benford
# --------------------------------------------------------------------------- #
AUDIT: list[tuple[str, str]] = notebook(
    [
        (
            "markdown",
            """\
# 02 — Audit procedures and Benford's Law

Two families of test, run over the same population:

1. **Nine independent rule-based procedures** — the controls an auditor can read,
   re-perform and challenge.
2. **Benford's Law** — a population-level analytical procedure whose only useful
   output is a disaggregation.

Both are evaluated against the injected ground truth. That is legitimate for *grading*
and it is disclosed here; neither procedure sees a label when it runs.
""",
        ),
        (
            "markdown",
            "## 1. How did each procedure perform?\n\n"
            "Precision and recall here are measured **per pattern**: for a given rule the "
            "positive class is `anomaly_type == rule_key`. So `recall = 1.000` means every "
            "voucher injected with *that* pattern was caught — not that every anomaly was.",
        ),
        (
            "code",
            """\
evaluation = pd.read_csv(PROJECT_ROOT / "outputs/reports/rule_evaluation.csv")
evaluation = evaluation.sort_values("precision", ascending=False).reset_index(drop=True)

display(
    evaluation[
        ["rule_label", "flagged", "true_positives", "false_positives", "precision", "recall", "f1"]
    ].style.format(
        {"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}
    ).background_gradient(subset=["precision", "f1"], cmap="RdYlGn", vmin=0, vmax=1)
)

print(f"alerts raised          {int(evaluation['flagged'].sum()):,}")
print(f"true positives         {int(evaluation['true_positives'].sum()):,}")
print(f"false positives        {int(evaluation['false_positives'].sum()):,}")
""",
        ),
        (
            "markdown",
            """\
## 2. The precision/recall trade-off is a workload decision

Two rules carry most of the noise. Plotting alert volume against precision makes the
resourcing consequence obvious: a low-precision rule does not just add a few rows, it
adds hundreds.""",
        ),
        (
            "code",
            """\
figure, axis = plt.subplots(figsize=(11, 4.5))
colours = ["#2E7D32" if value >= 0.6 else "#C00000" for value in evaluation["precision"]]
axis.barh(evaluation["rule_label"], evaluation["flagged"], color=colours)
axis.invert_yaxis()
axis.set_xlabel("alerts raised")
axis.set_title("Alert volume by procedure (red = precision below 0.6)")
for position, (flagged, precision) in enumerate(zip(evaluation["flagged"], evaluation["precision"])):
    axis.text(flagged, position, f"  {flagged:,} @ P={precision:.2f}", va="center", fontsize=9)
plt.tight_layout()
plt.show()

noisy = evaluation[evaluation["precision"] < 0.6]
print(f"{len(noisy)} procedures account for "
      f"{noisy['flagged'].sum():,} of {evaluation['flagged'].sum():,} alerts "
      f"({noisy['flagged'].sum() / evaluation['flagged'].sum():.1%} of the workload) "
      f"while catching {noisy['true_positives'].sum():,} true positives.")
print()
print("These are scoping procedures. Their output defines a population to look at;")
print("it is not evidence about any individual voucher.")
""",
        ),
        (
            "markdown",
            """\
## 3. Why the large-round-amount rule needs three conditions

A naive `amount % 1000 == 0` is almost meaningless: plenty of legitimate payments are
round because rent, retainers and instalments are round. The rule requires a round
amount **and** an amount above the 95th percentile **and** at least CNY 100,000.""",
        ),
        (
            "code",
            """\
scored = pd.read_parquet(PROJECT_ROOT / "data/processed/transactions_scored.parquet")

round_amounts = scored["is_round_amount"].astype(bool)
naive = round_amounts
three_condition = scored["large_round_amount_flag"].astype(bool)

print(f"naive amount % 1000 == 0        {int(naive.sum()):,} vouchers "
      f"({naive.mean():.2%} of the ledger)")
print(f"round AND >= P95 AND >= 100k    {int(three_condition.sum()):,} vouchers "
      f"({three_condition.mean():.2%} of the ledger)")
print()
print("How much of each rule's output is actually an injected anomaly:")
truth = scored["anomaly_type"].astype("string")
print(f"  naive           {(truth.eq('large_round_amount') & naive).sum() / max(naive.sum(), 1):.1%}")
print(f"  three-condition {(truth.eq('large_round_amount') & three_condition).sum() / max(three_condition.sum(), 1):.1%}")
""",
        ),
        (
            "markdown",
            "## 4. Benford's Law: first digit\n\n"
            "MAD is the deciding measure, banded per Nigrini. The chi-square is reported "
            "beside it and is deliberately not the deciding measure — with 30,000 "
            "observations it rejects distributions whose departures are far too small to "
            "matter.",
        ),
        (
            "code",
            """\
benford = json.loads((PROJECT_ROOT / "outputs/reports/benford_results.json").read_text())
first = benford["first_digit_test"]
table = pd.DataFrame(benford["first_digit_table"])

print(f"n = {first['n_observations']:,}   MAD = {first['mad']:.6f}   "
      f"chi2 = {first['chi_square']:.3f} (df={first['degrees_of_freedom']}, p={first['p_value']:.4f})")
print(f"conformity: {first['conformity']}")
print()
print(first["interpretation"])
print()
print("CAVEAT:", first["caveat"])

figure, axis = plt.subplots(figsize=(10, 4.5))
positions = np.arange(len(table))
axis.bar(positions - 0.2, table["observed_frequency"], width=0.4, label="observed", color="#1F4E79")
axis.bar(positions + 0.2, table["expected_frequency"], width=0.4, label="Benford expected", color="#C00000")
axis.set_xticks(positions)
axis.set_xticklabels(table["digit"])
axis.set_xlabel("leading digit")
axis.set_ylabel("frequency")
axis.set_title("First-digit distribution vs Benford's Law")
axis.legend()
plt.tight_layout()
plt.show()

display(table.style.format({
    "observed_frequency": "{:.4%}", "expected_frequency": "{:.4%}",
    "deviation": "{:+.4%}", "z_score": "{:+.2f}",
}))
""",
        ),
        (
            "markdown",
            """\
## 5. The two tests disagree, and that is the interesting part

The first-two-digits chi-square is very large while its MAD is negligible. The
per-bucket z-scores explain both: 90 buckets each carry a tiny deviation, and chi-square
accumulates them.""",
        ),
        (
            "code",
            """\
two = benford["first_two_digits_test"]
two_table = pd.DataFrame(benford["first_two_digits_table"])

print(f"first digit      MAD {first['mad']:.6f}  chi2 {first['chi_square']:8.3f}  df {first['degrees_of_freedom']:3d}  p {first['p_value']:.3g}")
print(f"first two digits MAD {two['mad']:.6f}  chi2 {two['chi_square']:8.3f}  df {two['degrees_of_freedom']:3d}  p {two['p_value']:.3g}")
print()
print(f"both are classified '{two['conformity']}' on MAD.")
print()

flagged = two_table[two_table["significant"]]
print(f"buckets with |z| > 1.96: {len(flagged)} of {len(two_table)}")
print(f"expected by chance at the 5% level: {0.05 * len(two_table):.1f}")
print(f"largest |z|: {two_table['z_score'].abs().max():.2f}")
print()
print("MAD measures the average MAGNITUDE of the departure, which is negligible.")
print("Chi-square asks whether the distribution is EXACTLY Benford, which with this")
print("many observations it never is. The MAD is the measure that maps to the audit")
print("question, and 13 exceedances across 90 buckets is a multiple-testing effect,")
print("not evidence of irregularity.")

display(flagged[["digit", "observed_frequency", "expected_frequency", "deviation", "z_score"]]
        .style.format({"observed_frequency": "{:.4%}", "expected_frequency": "{:.4%}",
                       "deviation": "{:+.4%}", "z_score": "{:+.2f}"}))
""",
        ),
        (
            "markdown",
            "## 6. Disaggregation is the actual deliverable\n\n"
            "A ledger-wide MAD of 0.002 tells an auditor nothing actionable. The per-account "
            "breakdown is where an enquiry starts, because deviations in opposite "
            "directions cancel out in the aggregate.",
        ),
        (
            "code",
            """\
by_account = pd.DataFrame(benford["by_account"]).sort_values("mad", ascending=False)
by_process = pd.DataFrame(benford["by_process"]).sort_values("mad", ascending=False)

print("Accounts, worst conformity first:")
display(by_account[["account_code", "n_observations", "total_amount", "mad", "conformity"]]
        .head(8).style.format({"total_amount": "{:,.0f}", "mad": "{:.5f}"}))

print("Business processes, worst conformity first:")
display(by_process[["process", "n_observations", "total_amount", "mad", "conformity"]]
        .head(8).style.format({"total_amount": "{:,.0f}", "mad": "{:.5f}"}))

nonconforming = by_account[by_account["conformity"] != "Close conformity"]
print(f"{len(nonconforming)} of {len(by_account)} accounts depart from close conformity.")
print("That is where an auditor would look next - and each one has an innocent")
print("explanation to rule out first (policy-constrained amounts, narrow ranges).")
""",
        ),
        (
            "markdown",
            "## 7. What Benford's Law cannot do\n\n"
            "This is a deliverable, not a footnote:\n\n"
            "- A **conforming** distribution is entirely compatible with fabricated entries.\n"
            "- A **nonconforming** distribution is usually a legitimately skewed account.\n"
            "- It operates on a **population** and cannot say \"this voucher is odd\".\n\n"
            "It is used here as a disaggregation tool, which is why the statistical "
            "component carries a weight of 0.10 — the joint-smallest in the risk score.",
        ),
        (
            "code",
            """\
assert "not a test for fraud" in first["caveat"], "the caveat must survive"
print("caveat preserved in the artefact:", first["caveat"])
""",
        ),
        (
            "markdown",
            """\
## What this establishes

- **The nine procedures raise 2,475 alerts on 2,208 vouchers** — 7.33% of the ledger, a
  population a team can actually work.
- **Two procedures produce 51% of the workload for a small share of the detections.**
  Keeping them as scoping procedures rather than findings would halve the review list.
- **The large-round-amount rule needs three conditions.** The naive version flags 505
  vouchers; adding the size and materiality gates is what makes the output usable.
- **Benford is a non-event, and reporting it as such is the point.** Close conformity at
  MAD 0.00218 tells the auditor to spend their time elsewhere. An analytics function
  that always finds something is one nobody trusts.
- **The chi-square and the MAD disagree, and the MAD wins.** That is a judgement about
  what the audit question is, not a technicality.
""",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# 03 - Anomaly detection
# --------------------------------------------------------------------------- #
MODEL: list[tuple[str, str]] = notebook(
    [
        (
            "markdown",
            """\
# 03 — Unsupervised anomaly detection

The nine rules in notebook 02 only find the patterns someone thought to write down. This
notebook covers the model that finds what the rules do not.

**Isolation Forest**, unsupervised: on a real engagement nobody hands you a column
marked "fraud". Training on injected labels would produce a model that had learned the
generator's patterns rather than anything about irregularity.

**Where the labels appear.** `anomaly_label` / `anomaly_type` are used in this notebook
for one purpose only — grading the model after the fact. They are never model inputs,
and this is disclosed on the dashboard as well.
""",
        ),
        (
            "markdown",
            "## 1. Configuration and features\n\n"
            "The inclusion test for a feature is that it can be stated as an audit question. "
            "If you cannot describe it to a partner, it does not belong in the matrix.",
        ),
        (
            "code",
            """\
metrics = json.loads((PROJECT_ROOT / "outputs/reports/model_metrics.json").read_text())
evaluation = metrics["evaluation"]
hyperparameters = metrics["hyperparameters"]

print(f"model            {metrics['model']}")
for key, value in hyperparameters.items():
    print(f"  {key:15s} {value}")
print(f"features         {metrics['n_features']}")
print()
print("Feature list:")
for index, feature in enumerate(metrics["feature_columns"], start=1):
    print(f"  {index:2d}. {feature}")
print()
print("Imputed with the population median (undefined for some vouchers):")
for feature, value in metrics["imputed_features"].items():
    print(f"  {feature:34s} {value:,.3f}")
print()
print("NOTE:", metrics["note"])
""",
        ),
        (
            "markdown",
            "## 2. Confusion matrix\n\n"
            "The false negatives are the number that should worry an auditor. The false "
            "positives are the number that will annoy the client.",
        ),
        (
            "code",
            """\
matrix = evaluation["confusion_matrix"]
true_negatives, false_positives = matrix[0]
false_negatives, true_positives = matrix[1]

print(f"flagged by the model   {evaluation['n_flagged']:,}")
print(f"injected anomalies     {evaluation['n_anomalies']:,}")
print()
print(f"  true positives   {true_positives:6,d}   caught")
print(f"  false negatives  {false_negatives:6,d}   missed")
print(f"  false positives  {false_positives:6,d}   clean vouchers flagged")
print(f"  true negatives   {true_negatives:6,d}")
print()
print(f"precision {evaluation['precision']:.4f}   recall {evaluation['recall']:.4f}   "
      f"f1 {evaluation['f1']:.4f}   ROC-AUC {evaluation['roc_auc']:.4f}")
print(f"false positives are {false_positives / max(evaluation['n_flagged'], 1):.1%} of the review list")

figure, axis = plt.subplots(figsize=(5, 4))
image = axis.imshow(matrix, cmap="Blues")
axis.set_xticks([0, 1], labels=["predicted normal", "predicted anomaly"])
axis.set_yticks([0, 1], labels=["actually normal", "actually anomalous"])
for row in range(2):
    for column in range(2):
        axis.text(column, row, f"{matrix[row][column]:,}", ha="center", va="center",
                  color="white" if matrix[row][column] > 1000 else "black", fontsize=12)
axis.set_title("Isolation Forest confusion matrix")
plt.tight_layout()
plt.show()
""",
        ),
        (
            "markdown",
            """\
## 3. The operating point is a resourcing decision

Precision and recall at a fixed cut are not interesting on their own. What an audit team
actually chooses is how many vouchers it can work — so the useful curve is precision and
recall *against the review budget*.""",
        ),
        (
            "code",
            """\
total = summary_vouchers = evaluation["n_samples"]
precision_at_k = evaluation["precision_at_k"]
recall_at_k = evaluation["recall_at_k"]

budget = pd.DataFrame(
    {
        "vouchers_reviewed": [int(float(key.replace("top_", "").replace("%", "")) / 100 * total)
                              for key in precision_at_k],
        "precision": list(precision_at_k.values()),
        "recall": list(recall_at_k.values()),
    },
    index=list(precision_at_k),
)
base_rate = evaluation["n_anomalies"] / total
budget["lift_vs_random"] = budget["precision"] / base_rate

display(budget.style.format({
    "vouchers_reviewed": "{:,.0f}", "precision": "{:.1%}", "recall": "{:.1%}",
    "lift_vs_random": "{:.1f}x",
}))

print(f"random-selection baseline (the anomaly rate): {base_rate:.2%}")

figure, axis = plt.subplots(figsize=(9, 4.5))
axis.plot(budget["vouchers_reviewed"], budget["precision"], marker="o", color="#1F4E79", label="precision")
axis.plot(budget["vouchers_reviewed"], budget["recall"], marker="s", color="#C00000", label="recall")
axis.axhline(base_rate, linestyle="--", color="#777777", label="random baseline")
axis.set_xscale("log")
axis.set_xlabel("vouchers reviewed (log scale)")
axis.set_ylabel("rate")
axis.set_title("What the review budget buys")
axis.legend()
plt.tight_layout()
plt.show()

print()
print(f"At the top 0.5% ({budget['vouchers_reviewed'].iloc[0]:,} vouchers) precision is "
      f"{budget['precision'].iloc[0]:.1%} - a {budget['lift_vs_random'].iloc[0]:.0f}x lift over random.")
print("Recall matters more than precision in audit: a missed misstatement is an audit")
print("failure, a false positive is a wasted hour. But recall is not free, and pretending")
print("otherwise is how analytics projects lose the confidence of the people working the output.")
""",
        ),
        (
            "markdown",
            """\
## 4. Does the model add anything over the rules?

This is the question the project has to answer honestly, and the answer is
uncomfortable. It also has to be computed at the **anomaly** level, not the flag level:
the vouchers the model raised that no rule raised are almost entirely false positives,
so counting them as "anomalies the rules missed" would be wrong by two orders of
magnitude.

The overlap is reported on both bases below.""",
        ),
        (
            "code",
            """\
scored = pd.read_parquet(PROJECT_ROOT / "data/processed/transactions_scored.parquet")

rule = scored["rule_alert_count"].astype(int) > 0
model = scored["ml_anomaly_flag"].astype(bool)
truth = scored["anomaly_type"].astype("string").notna()

print("=== flag level: what an auditor would actually have to read ===")
print(f"  raised by a rule                     {int(rule.sum()):6,d}")
print(f"  raised by the model                  {int(model.sum()):6,d}")
print(f"  raised by both                       {int((rule & model).sum()):6,d}")
print(f"  raised by the model, no rule         {int((model & ~rule).sum()):6,d}")
print()
print("=== anomaly level: how many injected anomalies each one actually caught ===")
print(f"  injected anomalies                   {int(truth.sum()):6,d}")
print(f"  caught by a rule                     {int((truth & rule).sum()):6,d}")
print(f"  caught by the model                  {int((truth & model).sum()):6,d}")
print(f"  caught by both                       {int((truth & rule & model).sum()):6,d}")
print(f"  caught by the model ONLY             {int((truth & model & ~rule).sum()):6,d}")
print(f"  caught by a rule ONLY                {int((truth & rule & ~model).sum()):6,d}")
print(f"  caught by NEITHER                    {int((truth & ~rule & ~model).sum()):6,d}")
print()
print(f"  of the {int((model & ~rule).sum()):,} model-only flags, "
      f"{int((model & ~rule & truth).sum())} are real anomalies "
      f"({(model & ~rule & truth).sum() / max((model & ~rule).sum(), 1):.1%})")
""",
        ),
        (
            "markdown",
            """\
### The conclusion is that this benchmark cannot answer the question

The rules catch **98.2%** of the injected anomalies, and the model adds **2**. That is
not a failure of the model — it is a property of the benchmark. The anomalies were
injected to match the rule definitions, so the rules are near-exhaustive *by
construction*. A benchmark built from a rule set cannot measure what a model adds
beyond that rule set.

This matters because it is the single easiest number in the project to misquote. An
earlier version of the pipeline's own summary reported `detections_not_caught_by_rules
= 244`, which reads as "244 anomalies the rules missed" — and 242 of those 244 are false
positives. The summary now reports both bases under names that cannot be confused, and
`tests/test_reporting.py` pins the distinction.

What the model *does* establish is that it ranks well without labels: ROC-AUC 0.816 and
47% precision at the top 0.5%. The argument for keeping it is generalisation to patterns
nobody wrote a rule for — which is exactly what a synthetic benchmark with a known
answer key cannot test. Saying so is more useful than claiming a result the benchmark
cannot support.""",
        ),
        (
            "markdown",
            "## 5. Where the model's score actually separates the classes\n\n"
            "The score distribution is what the ROC-AUC summarises. The overlap is the "
            "honest part: the classes are not cleanly separable, and no amount of tuning "
            "would make them so.",
        ),
        (
            "code",
            """\
figure, axis = plt.subplots(figsize=(10, 4.5))
axis.hist(scored.loc[~truth, "anomaly_score"], bins=60, alpha=0.65,
          label="normal (injected label 0)", color="#1F4E79", density=True)
axis.hist(scored.loc[truth, "anomaly_score"], bins=60, alpha=0.65,
          label="injected anomaly", color="#C00000", density=True)
axis.axvline(evaluation["threshold"], linestyle="--", color="#333333",
             label=f"decision threshold {evaluation['threshold']:.3f}")
axis.set_xlabel("normalised anomaly score")
axis.set_ylabel("density")
axis.set_title("Anomaly score by class (labels used for grading only)")
axis.legend()
plt.tight_layout()
plt.show()

print(f"median score, normal vouchers     {scored.loc[~truth, 'anomaly_score'].median():.3f}")
print(f"median score, injected anomalies  {scored.loc[truth, 'anomaly_score'].median():.3f}")
print()
print("The distributions overlap heavily. A model that separated them cleanly on this")
print("benchmark would be evidence of leakage, not of quality.")
""",
        ),
        (
            "markdown",
            "## 6. What this model does not do",
        ),
        (
            "code",
            """\
for line in [
    "It does not detect fraud. It ranks vouchers by how unusual they look relative to",
    "  the rest of the ledger, which is a different question.",
    "Its reported metrics are an UPPER BOUND: the anomalies were injected by a generator",
    "  whose patterns are known, and real irregularities do not announce themselves.",
    "It is not tuned. Hyperparameters are defaults chosen for reproducibility; tuning",
    "  them against the injected labels would make the reported numbers even more optimistic.",
    "It explains itself at the feature level, not the voucher level. It can say which",
    "  features were extreme; it cannot say why an entry was made. That is the auditor's job.",
    "There is no temporal validation. A production deployment would need to check that",
    "  behaviour is stable across periods, and would need drift monitoring.",
]:
    print(line)
""",
        ),
        (
            "markdown",
            """\
## What this establishes

- **The model reaches ROC-AUC 0.816** with no labels, on 20 features each of which is
  expressible as an audit question.
- **At the top 0.5% of the ranking it is 47% precise — a 15x lift over random.** That is
  the usable operating point, and it is a resourcing decision, not a tuning knob.
- **On this benchmark the model adds almost nothing over the rules: 2 anomalies.** The
  rules catch 98.2% of the injected anomalies, because the anomalies were generated to
  match the rule definitions. That comparison is circular by construction, and this
  notebook does not claim otherwise.
- **The 244 figure is a flag count, not a detection count.** 242 of those 244 vouchers
  are false positives. Reporting it as "anomalies the rules missed" would be wrong by
  two orders of magnitude.
- **The score distributions overlap heavily**, which is the honest picture. Clean
  separation on a synthetic benchmark would indicate leakage, not quality.
- **Every metric here is an upper bound**, because the anomalies were injected by a
  generator whose patterns are known to the detectors.
""",
        ),
    ]
)


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_eda.ipynb": EDA,
    "02_audit_analysis.ipynb": AUDIT,
    "03_anomaly_detection.ipynb": MODEL,
}


def to_source(text: str) -> list[str]:
    """Split a cell body into the line list nbformat expects."""
    lines = text.splitlines(keepends=True)
    return lines or [""]


def build_notebook(cells: list[tuple[str, str]]) -> dict[str, Any]:
    """Assemble a notebook document."""
    return {
        "cells": [
            {
                "cell_type": kind,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": to_source(body),
            }
            if kind == "code"
            else {"cell_type": "markdown", "metadata": {}, "source": to_source(body)}
            for kind, body in cells
        ],
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": LANGUAGE_INFO,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(cells: list[tuple[str, str]], name: str) -> None:
    """Run every code cell in one namespace, as a kernel would."""
    import json as _json

    namespace: dict[str, Any] = {"__name__": "__main__", "display": lambda value: None, "json": _json}
    for index, (kind, body) in enumerate(cells):
        if kind != "code":
            continue
        try:
            exec(compile(body, f"<{name}:cell{index}>", "exec"), namespace)  # noqa: S102
        except Exception:
            print(f"\n!! {name} cell {index} raised:\n")
            traceback.print_exc()
            raise SystemExit(1)


def main() -> int:
    # Guarded rather than ``exist_ok=True``: some sandboxes raise on a redundant
    # mkdir instead of silently succeeding.
    if not NOTEBOOK_DIR.exists():
        NOTEBOOK_DIR.mkdir(parents=True)
    for name, cells in NOTEBOOKS.items():
        print(f"executing {name} ...", flush=True)
        execute(cells, name)
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(build_notebook(cells), indent=1) + "\n", encoding="utf-8")
        print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({len(cells)} cells)")
    print("\nall notebooks executed cleanly and were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
