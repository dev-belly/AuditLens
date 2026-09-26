"""Feature engineering for AuditLens.

Every downstream component - the rule engine, Benford's Law, the Isolation
Forest and the risk scoring engine - reads the feature table produced here.
Keeping feature construction in one module means a definition such as
"vendor average amount" is computed exactly once and used consistently.

The features fall into four families:

* **Amount features** - level, log level and position in the distribution.
* **Vendor behavioural features** - how this payment compares with everything
  else that vendor has ever done. This is where most of the audit signal lives:
  an absolute amount of CNY 400,000 means nothing until you know the vendor's
  normal range.
* **Control features** - weekend posting, self-approval, payment speed,
  proximity to the approval threshold.
* **Text features** - description length and emptiness.

Usage::

    python src/feature_engineering.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/feature_engineering.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import (
    APPROVAL_THRESHOLD_CNY,
    RAPID_PAYMENT_HOURS,
    SPLIT_THRESHOLD_HIGH_RATIO,
    SPLIT_THRESHOLD_LOW_RATIO,
    TRANSACTIONS_CLEAN,
    TRANSACTIONS_FEATURES,
    VENDOR_RISK_TABLE,
    VENDORS_CLEAN,
    Timer,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
)

LOGGER = get_logger(__name__)

#: Features handed to the Isolation Forest. Ordered for readability in the
#: model's feature-importance discussion in ``docs/methodology.md``.
ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "amount",
    "log_amount",
    "amount_percentile",
    "vendor_transaction_count",
    "vendor_average_amount",
    "vendor_total_amount",
    "vendor_amount_ratio",
    "vendor_max_amount",
    "vendor_age_days",
    "days_since_last_transaction",
    "account_transaction_frequency",
    "is_weekend",
    "is_round_amount",
    "days_since_vendor_registration",
    "approval_time_hours",
    "payment_delay_hours",
    "same_vendor_daily_transactions",
    "description_length",
    "is_self_approval",
    "near_approval_threshold",
)

#: Human-readable descriptions used by the dashboard's model documentation.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "amount": "Voucher value in CNY",
    "log_amount": "Natural log of the voucher value - compresses the heavy right tail",
    "amount_percentile": "Percentile rank of the amount across the whole ledger",
    "vendor_transaction_count": "Number of vouchers ever raised against this vendor",
    "vendor_average_amount": "Mean voucher value for this vendor",
    "vendor_total_amount": "Cumulative value paid to this vendor",
    "vendor_amount_ratio": "Voucher value divided by the vendor's average - a spike detector",
    "vendor_max_amount": "Largest voucher ever raised against this vendor",
    "vendor_age_days": "Age of the vendor on the transaction date",
    "days_since_last_transaction": "Days since the previous voucher for this vendor",
    "account_transaction_frequency": "Share of all vouchers posted to this account",
    "is_weekend": "Transaction date falls on a Saturday or Sunday",
    "is_round_amount": "Amount is divisible by 1,000 (a manual-entry signature)",
    "days_since_vendor_registration": "Days between vendor registration and the voucher",
    "approval_time_hours": "Hours from the source document (invoice, or voucher date) to approval",
    "payment_delay_hours": "Hours between invoice receipt and payment",
    "same_vendor_daily_transactions": "Number of vouchers for this vendor on this date",
    "description_length": "Character length of the narration",
    "is_self_approval": "The voucher creator is also the approver",
    "near_approval_threshold": "Amount sits within 10% below the approval threshold",
}


# --------------------------------------------------------------------------- #
# Vendor-level aggregates
# --------------------------------------------------------------------------- #
def build_vendor_features(transactions: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vendor behaviour from the ledger and join the master data.

    Args:
        transactions: Clean transaction table.
        vendors: Clean vendor master file.

    Returns:
        One row per vendor with activity, amount and lifecycle statistics.
    """
    has_vendor = transactions["vendor_id"].notna()
    vendor_tx = transactions.loc[has_vendor]

    if vendor_tx.empty:
        LOGGER.warning("No vendor-linked transactions found; vendor features will be empty.")
        return vendors.copy()

    grouped = vendor_tx.groupby("vendor_id")
    features = pd.DataFrame(
        {
            "vendor_transaction_count": grouped["transaction_id"].count(),
            "vendor_total_amount": grouped["debit_amount"].sum(),
            "vendor_average_amount": grouped["debit_amount"].mean(),
            "vendor_median_amount": grouped["debit_amount"].median(),
            "vendor_max_amount": grouped["debit_amount"].max(),
            "vendor_amount_std": grouped["debit_amount"].std(),
            "vendor_first_transaction": grouped["transaction_date"].min(),
            "vendor_last_transaction": grouped["transaction_date"].max(),
            "vendor_unique_accounts": grouped["account_code"].nunique(),
            "vendor_unique_departments": grouped["department"].nunique(),
        }
    ).reset_index()

    # Weekday share: a vendor that is only ever paid at weekends stands out.
    vendor_tx = vendor_tx.assign(_weekend=vendor_tx["transaction_date"].dt.dayofweek >= 5)
    weekend_share = vendor_tx.groupby("vendor_id")["_weekend"].mean().rename("vendor_weekend_share")
    features = features.merge(weekend_share.reset_index(), on="vendor_id", how="left")

    # Self-approval share per vendor.
    vendor_tx = vendor_tx.assign(_self=vendor_tx["created_by"] == vendor_tx["approved_by"])
    self_share = vendor_tx.groupby("vendor_id")["_self"].mean().rename("vendor_self_approval_share")
    features = features.merge(self_share.reset_index(), on="vendor_id", how="left")

    features = features.merge(
        vendors[
            [
                "vendor_id",
                "vendor_name",
                "vendor_category",
                "bank_account",
                "registration_date",
                "country",
                "risk_level",
            ]
        ],
        on="vendor_id",
        how="left",
    )

    # Master-data risk level encoded as a number for the scoring engine.
    features["vendor_master_risk_score"] = (
        features["risk_level"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0}).fillna(0.0)
    )

    # Vendors that share a bank account with another vendor - a shell-company
    # indicator worth surfacing in the vendor risk table.
    bank_counts = features["bank_account"].value_counts()
    features["shared_bank_account"] = features["bank_account"].map(bank_counts).gt(1).fillna(False)

    features["vendor_amount_std"] = features["vendor_amount_std"].fillna(0.0)
    features["vendor_weekend_share"] = features["vendor_weekend_share"].fillna(0.0)
    features["vendor_self_approval_share"] = features["vendor_self_approval_share"].fillna(0.0)

    LOGGER.info("Built vendor features for %s vendors", len(features))
    return features


# --------------------------------------------------------------------------- #
# Transaction-level features
# --------------------------------------------------------------------------- #
def _add_vendor_features(transactions: pd.DataFrame, vendor_features: pd.DataFrame) -> pd.DataFrame:
    """Join vendor aggregates and derive per-transaction vendor comparisons."""
    columns = [
        "vendor_id",
        "vendor_transaction_count",
        "vendor_total_amount",
        "vendor_average_amount",
        "vendor_median_amount",
        "vendor_max_amount",
        "vendor_amount_std",
        "vendor_last_transaction",
        "vendor_first_transaction",
        "registration_date",
        "vendor_master_risk_score",
        "vendor_weekend_share",
        "vendor_self_approval_share",
        "shared_bank_account",
    ]
    available = [column for column in columns if column in vendor_features.columns]
    df = transactions.merge(
        vendor_features[available], on="vendor_id", how="left", suffixes=("", "_vendor")
    )

    df["vendor_age_days"] = (df["transaction_date"] - df["registration_date"]).dt.days
    df["days_since_vendor_registration"] = df["vendor_age_days"]

    # How far this voucher sits above the vendor's own norm.
    safe_average = df["vendor_average_amount"].replace(0, np.nan)
    df["vendor_amount_ratio"] = (df["debit_amount"] / safe_average).fillna(0.0)

    # Days since the vendor's previous voucher (NaN for the first one).
    df = df.sort_values(["vendor_id", "transaction_date", "transaction_id"])
    previous = df.groupby("vendor_id", dropna=True)["transaction_date"].shift(1)
    df["days_since_last_transaction"] = (df["transaction_date"] - previous).dt.days
    df = df.sort_index()

    # Vendors with no ledger activity at all (dormant master-file entries).
    df["vendor_transaction_count"] = df["vendor_transaction_count"].fillna(0)
    df["vendor_total_amount"] = df["vendor_total_amount"].fillna(0.0)
    df["vendor_average_amount"] = df["vendor_average_amount"].fillna(0.0)
    df["vendor_max_amount"] = df["vendor_max_amount"].fillna(0.0)
    df["vendor_amount_std"] = df["vendor_amount_std"].fillna(0.0)
    df["vendor_master_risk_score"] = df["vendor_master_risk_score"].fillna(0.0)
    df["shared_bank_account"] = df["shared_bank_account"].fillna(False)
    return df


def _add_account_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add account-level frequency and value features."""
    df = transactions
    counts = df["account_code"].value_counts(normalize=True)
    df["account_transaction_frequency"] = df["account_code"].map(counts).fillna(0.0)

    totals = df.groupby("account_code")["debit_amount"].sum()
    df["account_total_amount"] = df["account_code"].map(totals).fillna(0.0)

    # Frequency of the *contra* account matters too: an unusual credit leg is
    # just as interesting as an unusual debit leg.
    credit_counts = df["credit_account_code"].value_counts(normalize=True)
    df["credit_account_frequency"] = df["credit_account_code"].map(credit_counts).fillna(0.0)
    return df


def _add_control_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add weekend, approval, timing and threshold-proximity features."""
    df = transactions

    df["is_weekend"] = df["transaction_date"].dt.dayofweek >= 5
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["month"] = df["transaction_date"].dt.month

    # A round amount is a manual-entry signature. Dividing by 1,000 catches the
    # amounts a human types by hand.
    df["is_round_amount"] = (df["debit_amount"] % 1_000 == 0) & (df["debit_amount"] >= 1_000)
    df["round_amount_level"] = np.where(
        df["debit_amount"] % 10_000 == 0,
        2,
        np.where(df["debit_amount"] % 1_000 == 0, 1, 0),
    )

    df["is_self_approval"] = df["created_by"] == df["approved_by"]

    # Approval speed: how long the approval control took, measured from the
    # source document. For invoice-backed vouchers the clock starts when the
    # invoice arrives; for everything else it starts when the voucher is raised.
    # Either way the value is non-negative, which keeps the feature comparable.
    approval_anchor = df["invoice_time"].where(df["invoice_time"].notna(), df["transaction_date"])
    approval_gap = df["approval_time"] - approval_anchor
    df["approval_time_hours"] = (approval_gap.dt.total_seconds() / 3_600).round(2)
    df["approval_time_hours"] = df["approval_time_hours"].clip(lower=0.0)

    # Payment speed: invoice receipt to cash leaving the bank.
    payment_gap = df["payment_time"] - df["invoice_time"]
    df["payment_delay_hours"] = (payment_gap.dt.total_seconds() / 3_600).round(2)
    df["is_rapid_payment"] = df["payment_delay_hours"] < RAPID_PAYMENT_HOURS

    # Proximity to the approval threshold.
    lower = APPROVAL_THRESHOLD_CNY * SPLIT_THRESHOLD_LOW_RATIO
    upper = APPROVAL_THRESHOLD_CNY * SPLIT_THRESHOLD_HIGH_RATIO
    df["near_approval_threshold"] = df["debit_amount"].between(lower, upper)
    df["distance_to_threshold"] = APPROVAL_THRESHOLD_CNY - df["debit_amount"]

    # Same-vendor activity on the same day - the substrate of the split rule.
    same_day = df.groupby(["vendor_id", "transaction_date"])["transaction_id"].transform("count")
    df["same_vendor_daily_transactions"] = same_day.fillna(1).astype(int)

    # Text features.
    description = df["description"].astype("string")
    df["description_length"] = description.fillna("").str.len()
    df["description_missing"] = df["description_missing"].fillna(True)
    df["description_is_short"] = df["description_length"] < 10

    # Debit / credit integrity.
    df["debit_credit_match"] = (
        (df["debit_amount"] - df["credit_amount"]).abs() <= 0.01
    )
    return df


def _add_amount_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add amount-level and distributional features."""
    df = transactions
    df["log_amount"] = np.log1p(df["debit_amount"].clip(lower=0))
    df["amount_percentile"] = df["debit_amount"].rank(pct=True)

    for percentile in (90, 95, 99):
        df[f"amount_above_p{percentile}"] = df["debit_amount"] > df["debit_amount"].quantile(percentile / 100)

    df["amount_zscore"] = (
        (df["debit_amount"] - df["debit_amount"].mean()) / max(df["debit_amount"].std(), 1e-9)
    ).round(4)
    return df


def build_features(
    transactions: pd.DataFrame,
    vendors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full feature table.

    Args:
        transactions: Clean transaction table.
        vendors: Clean vendor master file.

    Returns:
        Tuple of ``(transactions with features, vendor feature table)``.
    """
    vendor_features = build_vendor_features(transactions, vendors)

    df = transactions.copy()
    df = _add_vendor_features(df, vendor_features)
    df = _add_account_features(df)
    df = _add_control_features(df)
    df = _add_amount_features(df)

    missing = [column for column in ML_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Feature engineering did not produce: {missing}")

    # Boolean flags are cast to 0/1 so that StandardScaler can consume them.
    # Continuous features are deliberately left as NaN where the value is
    # semantically undefined - a voucher with no supplier invoice genuinely has no
    # invoice-to-payment cycle, and pretending otherwise (imputing zero) would
    # make every such voucher look like an instant payment. Imputation happens
    # once, in :func:`prepare_model_matrix`, where it can be documented as a
    # modelling choice rather than smuggled into the feature definitions.
    for column in ML_FEATURE_COLUMNS:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            df[column] = series.fillna(False).astype("int8")

    df = df.sort_values("transaction_date").reset_index(drop=True)

    LOGGER.info(
        "Feature table: %s rows x %s columns (%s model features)",
        len(df),
        df.shape[1],
        len(ML_FEATURE_COLUMNS),
    )
    return df, vendor_features


def prepare_model_matrix(
    transactions: pd.DataFrame,
    feature_columns: tuple[str, ...] = ML_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select and impute the model feature matrix.

    The Isolation Forest cannot consume ``NaN``. Rather than dropping rows - which
    would silently exclude every voucher without a supplier invoice - continuous
    features are imputed with their **median**, and count-like features with
    **zero**. Median imputation pushes an undefined value towards the middle of
    the distribution, so those vouchers are treated as unremarkable rather than
    anomalous, which is the conservative choice for an audit tool.

    Args:
        transactions: Feature-enriched transaction table.
        feature_columns: Columns to include in the matrix.

    Returns:
        Tuple of ``(imputed feature matrix, mapping of column -> fill value)``.
    """
    matrix = transactions.loc[:, list(feature_columns)].copy()
    fill_values: dict[str, float] = {}

    count_like = (
        "vendor_transaction_count",
        "same_vendor_daily_transactions",
        "description_length",
    )

    for column in feature_columns:
        if not matrix[column].isna().any():
            continue
        if column in count_like:
            fill = 0.0
        else:
            fill = float(matrix[column].median())
            if not np.isfinite(fill):
                fill = 0.0
        matrix[column] = matrix[column].astype("float64").fillna(fill)
        fill_values[column] = round(fill, 4)

    matrix = matrix.astype("float64")
    if fill_values:
        LOGGER.info("Imputed %s model features: %s", len(fill_values), fill_values)
    return matrix, fill_values


def run_feature_engineering() -> dict[str, Any]:
    """Load the clean tables, build features and persist the vendor feature table.

    Returns:
        Mapping with ``transactions`` and ``vendor_features``.
    """
    ensure_directories()
    transactions = load_dataframe(TRANSACTIONS_CLEAN)
    vendors = load_dataframe(VENDORS_CLEAN)

    transactions, vendor_features = build_features(transactions, vendors)
    save_dataframe(transactions, TRANSACTIONS_FEATURES)
    save_dataframe(vendor_features, VENDOR_RISK_TABLE)
    LOGGER.info("Wrote %s", TRANSACTIONS_FEATURES)
    LOGGER.info("Wrote %s", VENDOR_RISK_TABLE)

    return {"transactions": transactions, "vendor_features": vendor_features}


def main() -> int:
    """CLI entry point."""
    with Timer("feature engineering"):
        result = run_feature_engineering()

    df = result["transactions"]
    LOGGER.info("Model features (%s):", len(ML_FEATURE_COLUMNS))
    for column in ML_FEATURE_COLUMNS:
        LOGGER.info("  %-34s mean=%12.3f  std=%12.3f", column, float(df[column].mean()), float(df[column].std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
