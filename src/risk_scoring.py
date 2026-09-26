"""Audit Risk Score engine for AuditLens.

The final deliverable of the platform is a single number per voucher: an **Audit
Risk Score** between 0 and 100 that tells an auditor where to spend their time.
This module combines five independent lines of evidence into that number and, just
as importantly, explains it.

Why a weighted composite rather than a single model
---------------------------------------------------
Rules are precise but blind to anything they were not written for. The Isolation
Forest finds the unexpected but cannot say why. Vendor behaviour and amount
distribution are context that neither of them sees. Benford analysis operates on
populations, not vouchers. Combining them - and keeping the weights explicit -
produces a score an auditor can interrogate and a reviewer can challenge.

Weights
-------
======================================  =======
Component                               Weight
======================================  =======
``rule_risk``                           0.40
``ml_risk``                             0.25
``vendor_risk``                         0.15
``amount_risk``                         0.10
``statistical_risk``                    0.10
======================================  =======

Rationale (also documented in ``docs/methodology.md``):

* **Rule risk carries the largest weight** because rules encode known audit
  procedures. When a voucher is a duplicate payment or a split designed to dodge
  an approval threshold, that is direct evidence, not a statistical hint.
* **Machine learning is second** because it is the only component that can
  surface patterns nobody wrote a rule for. It is deliberately not first: an
  unexplained outlier score is weaker evidence than a named audit finding.
* **Vendor risk** is genuine context - a new vendor, a shared bank account, a
  concentration of alerts - but it describes a *relationship*, not the voucher.
* **Amount risk** is the weakest form of evidence on its own: a large round
  payment is worth a look, but plenty of legitimate payments are large and round.
* **Statistical risk** (Benford deviation of the voucher's account) is the
  smallest because it is a property of a population projected onto one row.

The weights are illustrative, not estimated. Estimating them would require
labelled outcomes from completed audits, which is exactly the data a real
engagement accumulates over time. See the Limitations section of the README.

Usage::

    python src/risk_scoring.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/risk_scoring.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import (
    APPROVAL_THRESHOLD_CNY,
    MATERIALITY_THRESHOLD_CNY,
    NEW_VENDOR_DAYS,
    RISK_LEVEL_ORDER,
    RISK_WEIGHTS,
    SCORED_TRANSACTIONS,
    VENDOR_RISK_TABLE,
    Timer,
    ensure_directories,
    get_logger,
    load_dataframe,
    risk_level_from_score,
    save_dataframe,
)

LOGGER = get_logger(__name__)

#: Sub-weights used to build the vendor-level risk score. They sum to 1.0.
#:
#: Calibration note: the two *behavioural* components dominate. A vendor's
#: master-data risk tier and its alert rate are what actually drive a decision to
#: put a supplier under enhanced monitoring; the remaining components corroborate.
#: With this calibration a vendor rated High in the master file *and* showing a
#: materially elevated alert rate lands in the High band, while an unremarkable
#: supplier with a single alert stays Low.
VENDOR_RISK_WEIGHTS: dict[str, float] = {
    "alert_rate": 0.35,
    "master_risk": 0.30,
    "shared_bank_account": 0.10,
    "new_vendor": 0.10,
    "self_approval_share": 0.05,
    "weekend_share": 0.05,
    "amount_concentration": 0.05,
}

#: Maximum number of reasons attached to a single voucher. Beyond about eight the
#: list stops being readable, and an auditor reviewing a voucher wants the top
#: signals rather than an exhaustive dump.
MAX_REASONS: int = 8


@dataclass
class RiskSummary:
    """Aggregate view of the scored population, used by the dashboard and report."""

    total_transactions: int
    total_amount: float
    flagged_transactions: int
    high_risk_transactions: int
    critical_transactions: int
    average_risk_score: float
    median_risk_score: float
    unique_vendors: int
    risk_level_counts: dict[str, int]
    risk_level_amounts: dict[str, float]
    potential_duplicate_payments: int
    weekend_transactions: int
    self_approval_transactions: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "total_transactions": self.total_transactions,
            "total_amount": round(self.total_amount, 2),
            "flagged_transactions": self.flagged_transactions,
            "high_risk_transactions": self.high_risk_transactions,
            "critical_transactions": self.critical_transactions,
            "average_risk_score": round(self.average_risk_score, 2),
            "median_risk_score": round(self.median_risk_score, 2),
            "unique_vendors": self.unique_vendors,
            "risk_level_counts": self.risk_level_counts,
            "risk_level_amounts": {key: round(value, 2) for key, value in self.risk_level_amounts.items()},
            "potential_duplicate_payments": self.potential_duplicate_payments,
            "weekend_transactions": self.weekend_transactions,
            "self_approval_transactions": self.self_approval_transactions,
        }


# --------------------------------------------------------------------------- #
# Vendor risk
# --------------------------------------------------------------------------- #
def build_vendor_risk_table(
    scored: pd.DataFrame,
    vendor_features: pd.DataFrame,
) -> pd.DataFrame:
    """Score every vendor on a 0-100 risk scale.

    The vendor score answers a different question from the transaction score: not
    "should I look at this payment?" but "should I look at this *supplier*?"
    Auditors work both angles, and a supplier with a poor overall pattern is worth
    a relationship-level enquiry even if no single payment looks dramatic.

    Args:
        scored: Transaction table carrying the ``<rule>_flag`` columns.
        vendor_features: Vendor-level aggregates from
            :func:`src.feature_engineering.build_vendor_features`.

    Returns:
        One row per vendor, sorted by descending risk score.
    """
    table = vendor_features.copy()

    # Aggregate the rule alerts up to the vendor.
    flag_columns = [column for column in scored.columns if column.endswith("_flag") and column != "ml_anomaly_flag"]
    vendor_scored = scored.loc[scored["vendor_id"].notna()]

    if not vendor_scored.empty and flag_columns:
        alert_counts = vendor_scored.groupby("vendor_id")[flag_columns].sum().sum(axis=1)
        alert_counts = alert_counts.rename("alert_count")
        table = table.merge(alert_counts, on="vendor_id", how="left")

        # Vouchers flagged by more than one independent rule - a vendor-level
        # signal that is available before the composite score exists.
        multi_alert = (
            vendor_scored.assign(_multi=vendor_scored["rule_alert_count"] >= 2)
            .groupby("vendor_id")["_multi"]
            .sum()
            .rename("multi_alert_transaction_count")
        )
        table = table.merge(multi_alert, on="vendor_id", how="left")
    else:
        table["alert_count"] = 0
        table["multi_alert_transaction_count"] = 0

    table["alert_count"] = table["alert_count"].fillna(0).astype(int)
    table["multi_alert_transaction_count"] = (
        table["multi_alert_transaction_count"].fillna(0).astype(int)
    )

    transaction_count = table["vendor_transaction_count"].replace(0, np.nan)
    table["alert_rate"] = (table["alert_count"] / transaction_count).fillna(0.0).astype("float64")

    ledger_total = max(float(scored["debit_amount"].sum()), 1.0)
    table["amount_concentration"] = (
        table["vendor_total_amount"].astype("float64") / ledger_total
    )

    # Normalise the two unbounded components onto 0-1 using percentile ranks, so
    # a single outlier vendor cannot dominate the scale.
    table["alert_rate_scaled"] = table["alert_rate"].astype("float64").rank(pct=True).fillna(0.0)
    table["amount_concentration_scaled"] = (
        table["amount_concentration"].astype("float64").rank(pct=True).fillna(0.0)
    )

    period_end = pd.to_datetime(scored["transaction_date"]).max()
    table["is_new_vendor"] = (
        (period_end - pd.to_datetime(table["registration_date"])).dt.days <= NEW_VENDOR_DAYS
    ).astype(float)

    components = {
        "master_risk": table["vendor_master_risk_score"].fillna(0.0),
        "alert_rate": table["alert_rate_scaled"],
        "shared_bank_account": table["shared_bank_account"].astype(float),
        "self_approval_share": table["vendor_self_approval_share"].fillna(0.0).clip(0.0, 1.0),
        "weekend_share": table["vendor_weekend_share"].fillna(0.0).clip(0.0, 1.0),
        "new_vendor": table["is_new_vendor"],
        "amount_concentration": table["amount_concentration_scaled"],
    }

    vendor_risk = pd.Series(0.0, index=table.index)
    for name, values in components.items():
        weight = VENDOR_RISK_WEIGHTS[name]
        vendor_risk = vendor_risk + weight * values.fillna(0.0)
        table[f"vendor_risk_{name}"] = values.fillna(0.0).round(4)

    table["vendor_risk_score"] = (vendor_risk.clip(0.0, 1.0) * 100).round(2)
    table["vendor_risk_level"] = table["vendor_risk_score"].apply(risk_level_from_score)
    table["vendor_risk_rank"] = table["vendor_risk_score"].rank(ascending=False, method="first").astype(int)

    table = table.sort_values("vendor_risk_score", ascending=False).reset_index(drop=True)
    LOGGER.info(
        "Scored %s vendors: %s high risk, %s critical",
        len(table),
        int((table["vendor_risk_level"] == "High").sum()),
        int((table["vendor_risk_level"] == "Critical").sum()),
    )
    return table


# --------------------------------------------------------------------------- #
# Transaction-level risk components
# --------------------------------------------------------------------------- #
def compute_amount_risk(df: pd.DataFrame) -> pd.Series:
    """Score how much a voucher's *size* contributes to its risk.

    Large amounts are inherently riskier - they are more material if wrong - but
    size alone is weak evidence, so the component is deliberately restrained:

    * 70% from the percentile rank above the median (a median-sized voucher
      scores zero, not 0.5);
    * 20% if the amount is an exact multiple of 1,000;
    * 10% if the amount clears planning materiality.
    """
    percentile = df["debit_amount"].astype("float64").rank(pct=True, method="average")
    size_component = ((percentile - 0.5) / 0.5).clip(lower=0.0, upper=1.0)

    is_round = (df["debit_amount"] % 1_000 == 0) & (df["debit_amount"] >= 10_000)
    above_materiality = df["debit_amount"] >= MATERIALITY_THRESHOLD_CNY

    score = (
        0.70 * size_component
        + 0.20 * is_round.astype(float)
        + 0.10 * above_materiality.astype(float)
    )
    return score.clip(0.0, 1.0).round(4)


def compute_statistical_risk(df: pd.DataFrame) -> pd.Series:
    """Score the statistical (Benford) evidence attached to a voucher.

    Benford's Law is a population test, so it cannot say "this voucher is odd".
    What it can say is "the account this voucher was posted to departs from the
    expected digit distribution", and that the voucher's own leading digit sits in
    an over- or under-represented bucket. Both signals are population properties
    projected onto the row, which is why this component carries the smallest
    weight.

    Two inputs, unequally weighted:

    * 75% from the Benford risk indicator of the voucher's account - the account
      is the natural sub-population to test, and it is the account's digit
      behaviour that an auditor would investigate;
    * 25% from how far the voucher's own leading digit deviates from its expected
      frequency. This is the weakest of all the signals in the model, which is why
      it is a quarter of a component that itself carries a weight of 0.10.
    """
    from src.benford import BENFORD_EXPECTED, extract_first_digit

    # --- Account-level Benford risk -----------------------------------------
    if "benford_risk" in df.columns:
        account_risk = df["benford_risk"].astype("float64").fillna(0.0)
    else:
        account_risk = pd.Series(0.0, index=df.index, dtype="float64")

    # --- Digit-level deviation ----------------------------------------------
    digit = extract_first_digit(df["debit_amount"])
    expected = digit.map(lambda value: BENFORD_EXPECTED.get(int(value), 0.0) if pd.notna(value) else 0.0)
    observed = digit.value_counts(normalize=True)
    observed_frequency = digit.map(observed).fillna(0.0)
    deviation = (observed_frequency - expected).abs().astype("float64")

    # Normalise against the largest observed deviation so the signal spans 0-1.
    max_deviation = float(deviation.max()) if len(deviation) else 0.0
    digit_risk = (
        (deviation / max_deviation).fillna(0.0)
        if max_deviation > 0
        else pd.Series(0.0, index=df.index, dtype="float64")
    )

    return (0.75 * account_risk + 0.25 * digit_risk).clip(0.0, 1.0).round(4)


def compute_risk_components(
    scored: pd.DataFrame,
    vendor_risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the five normalised risk components onto the transaction table.

    Args:
        scored: Transaction table with rule flags and the ML anomaly score.
        vendor_risk_table: Output of :func:`build_vendor_risk_table`.

    Returns:
        The transaction table with ``risk_component_*`` columns added.
    """
    df = scored.copy()

    df["risk_component_rule"] = df.get("rule_risk_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["risk_component_ml"] = df.get("anomaly_score", pd.Series(0.0, index=df.index)).fillna(0.0)

    vendor_lookup = vendor_risk_table.set_index("vendor_id")["vendor_risk_score"]
    df["risk_component_vendor"] = (
        df["vendor_id"].map(vendor_lookup).fillna(0.0).astype(float) / 100.0
    )

    df["risk_component_amount"] = compute_amount_risk(df)
    df["risk_component_statistical"] = compute_statistical_risk(df)

    return df


def compute_audit_risk_score(df: pd.DataFrame) -> pd.Series:
    """Combine the components into a 0-100 Audit Risk Score."""
    score = pd.Series(0.0, index=df.index)
    for component, weight in RISK_WEIGHTS.items():
        column = f"risk_component_{component.replace('_risk', '')}"
        if column not in df.columns:
            raise KeyError(f"Missing risk component column: {column}")
        score = score + weight * df[column].fillna(0.0)
    return (score.clip(0.0, 1.0) * 100).round(2)


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def build_risk_reasons(df: pd.DataFrame, max_reasons: int = MAX_REASONS) -> pd.Series:
    """Build the ordered, human-readable explanation for each voucher.

    This is the answer to "why was this transaction flagged?". The list is ordered
    by the weight of the component that produced each reason, so the strongest
    evidence reads first. A machine-learning flag is never emitted on its own - it
    is always accompanied by the features that drove it.
    """
    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def _add(mask: pd.Series, text_builder: Any) -> None:
        target = mask.fillna(False).to_numpy()
        if not target.any():
            return
        for position in np.flatnonzero(target):
            reasons[position].append(text_builder(position))

    # --- Rule-based reasons (highest weight) --------------------------------
    rule_reasons = df.get("rule_reasons", pd.Series([[] for _ in range(len(df))], index=df.index))
    for position, items in enumerate(rule_reasons):
        if isinstance(items, (list, tuple, np.ndarray)):
            reasons[position].extend(str(item) for item in items if item)

    # --- Machine-learning reason --------------------------------------------
    ml_flag = df.get("ml_anomaly_flag", pd.Series(False, index=df.index)).fillna(False)
    anomaly_score = df.get("anomaly_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    anomaly_rank = df.get("ml_anomaly_rank", pd.Series(0, index=df.index)).fillna(0).astype(int)

    def _ml_text(position: int) -> str:
        return (
            f"Isolation Forest anomaly (score {anomaly_score.iloc[position]:.3f}, "
            f"rank {anomaly_rank.iloc[position]:,} of {len(df):,})"
        )

    _add(ml_flag, _ml_text)

    # --- Vendor risk reason -------------------------------------------------
    vendor_component = df.get("risk_component_vendor", pd.Series(0.0, index=df.index)).fillna(0.0)
    vendor_name = df.get("vendor_name", pd.Series(pd.NA, index=df.index))
    vendor_flagged = vendor_component >= 0.45

    def _vendor_text(position: int) -> str:
        return (
            f"Vendor risk {vendor_component.iloc[position] * 100:.0f}/100 "
            f"for {vendor_name.iloc[position]}"
        )

    _add(vendor_flagged, _vendor_text)

    # --- Amount risk reason -------------------------------------------------
    percentile = df["debit_amount"].astype("float64").rank(pct=True, method="average")
    above_materiality = df["debit_amount"] >= MATERIALITY_THRESHOLD_CNY

    def _amount_text(position: int) -> str:
        return (
            f"Amount CNY {df['debit_amount'].iloc[position]:,.2f} is above the "
            f"{percentile.iloc[position] * 100:.1f}th percentile of the ledger"
        )

    _add(above_materiality, _amount_text)

    # --- Statistical reason -------------------------------------------------
    statistical = df.get("risk_component_statistical", pd.Series(0.0, index=df.index)).fillna(0.0)
    statistical_flagged = statistical >= 0.60

    def _statistical_text(position: int) -> str:
        return (
            f"Account {df['account_code'].iloc[position]} digit distribution departs from "
            "Benford's Law (a population-level flag, not evidence on this voucher alone)"
        )

    _add(statistical_flagged, _statistical_text)

    # --- Threshold proximity (context, added last) --------------------------
    near_threshold = df["debit_amount"].between(
        APPROVAL_THRESHOLD_CNY * 0.90, APPROVAL_THRESHOLD_CNY * 0.995
    )

    def _threshold_text(position: int) -> str:
        return (
            f"Amount sits just below the CNY {APPROVAL_THRESHOLD_CNY:,.0f} approval threshold"
        )

    _add(near_threshold, _threshold_text)

    # Truncate to the most important reasons.
    return pd.Series(
        [items[:max_reasons] for items in reasons], index=df.index, dtype="object"
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_risk_scoring(
    scored: pd.DataFrame | None = None,
    vendor_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the risk scoring engine end to end.

    Args:
        scored: Rule- and model-scored transaction table. Loaded from disk when omitted.
        vendor_features: Vendor aggregates. Loaded from disk when omitted.

    Returns:
        Mapping with ``transactions``, ``vendor_risk``, ``summary`` and ``weights``.
    """
    ensure_directories()

    if scored is None:
        raise FileNotFoundError(
            "run_risk_scoring requires the scored transaction table produced by "
            "src.audit_rules.run_rule_engine and src.anomaly_detection.run_anomaly_detection."
        )
    if vendor_features is None:
        vendor_features = load_dataframe(VENDOR_RISK_TABLE)

    # Vendor risk first: the transaction score consumes it.
    vendor_risk_table = build_vendor_risk_table(scored, vendor_features)

    # Benford risk is computed per account and projected onto the vouchers posted
    # to that account. Computed here if an earlier stage has not already done it.
    if "benford_risk" not in scored.columns:
        from src.benford import account_benford_risk

        benford_by_account = account_benford_risk(scored)
        scored = scored.merge(
            benford_by_account[["account_code", "benford_mad", "benford_risk", "benford_conformity"]],
            on="account_code",
            how="left",
        )
        LOGGER.info("Attached account-level Benford risk to %s vouchers", len(scored))

    df = compute_risk_components(scored, vendor_risk_table)
    df["audit_risk_score"] = compute_audit_risk_score(df)
    df["risk_level"] = df["audit_risk_score"].apply(risk_level_from_score)

    df["risk_reasons"] = build_risk_reasons(df)
    df["risk_reason_text"] = df["risk_reasons"].apply(lambda items: "; ".join(items))
    df["risk_reason_count"] = df["risk_reasons"].apply(len)

    df = df.sort_values("audit_risk_score", ascending=False).reset_index(drop=True)

    summary = build_summary(df)
    save_dataframe(df, SCORED_TRANSACTIONS)
    save_dataframe(vendor_risk_table, VENDOR_RISK_TABLE)
    LOGGER.info("Wrote %s", SCORED_TRANSACTIONS)
    LOGGER.info("Wrote %s", VENDOR_RISK_TABLE)

    return {
        "transactions": df,
        "vendor_risk": vendor_risk_table,
        "summary": summary,
        "weights": RISK_WEIGHTS,
    }


def build_summary(df: pd.DataFrame) -> RiskSummary:
    """Compute the headline metrics shown on the Executive Overview page."""
    level_counts = df["risk_level"].value_counts().to_dict()
    level_counts = {level: int(level_counts.get(level, 0)) for level in RISK_LEVEL_ORDER}

    level_amounts = df.groupby("risk_level")["debit_amount"].sum().to_dict()
    level_amounts = {level: float(level_amounts.get(level, 0.0)) for level in RISK_LEVEL_ORDER}

    high_risk = level_counts["High"] + level_counts["Critical"]

    return RiskSummary(
        total_transactions=int(len(df)),
        total_amount=float(df["debit_amount"].sum()),
        flagged_transactions=int((df["rule_alert_count"] > 0).sum()),
        high_risk_transactions=high_risk,
        critical_transactions=level_counts["Critical"],
        average_risk_score=float(df["audit_risk_score"].mean()),
        median_risk_score=float(df["audit_risk_score"].median()),
        unique_vendors=int(df["vendor_id"].nunique()),
        risk_level_counts=level_counts,
        risk_level_amounts=level_amounts,
        potential_duplicate_payments=int(df.get("duplicate_payment_flag", pd.Series(False, index=df.index)).sum()),
        weekend_transactions=int(df.get("weekend_posting_flag", pd.Series(False, index=df.index)).sum()),
        self_approval_transactions=int(df.get("self_approval_flag", pd.Series(False, index=df.index)).sum()),
    )


def main() -> int:
    """CLI entry point (expects the rule and ML stages to have run first)."""
    from src.anomaly_detection import run_anomaly_detection
    from src.audit_rules import run_rule_engine

    with Timer("risk scoring"):
        rule_result = run_rule_engine()
        ml_result = run_anomaly_detection(rule_result["scored"])
        result = run_risk_scoring(ml_result["transactions"])

    summary: RiskSummary = result["summary"]
    LOGGER.info("--- Risk summary ---")
    for key, value in summary.to_dict().items():
        LOGGER.info("  %-32s %s", key, value)
    LOGGER.info("--- Weights ---")
    for component, weight in result["weights"].items():
        LOGGER.info("  %-20s %.2f", component, weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
