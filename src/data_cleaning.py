"""Data-quality pipeline for AuditLens.

An audit analytics engagement never starts with clean data. Before any rule or
model can be trusted, the ledger extract has to be profiled and repaired, and
the repairs have to be documented - because "we dropped 1,400 rows" is itself a
finding that belongs in the audit file.

This module performs the standard sequence an audit data analytics team runs:

1. **Profile** - missing values, duplicates, invalid dates, invalid amounts,
   unbalanced entries, orphan vendor keys, inconsistent account names and
   messy currency codes.
2. **Repair** - normalise currencies, re-parse dates, drop exact duplicates,
   drop rows with unusable amounts, restore account names from the chart of
   accounts, quarantine orphan vendor references.
3. **Report** - a machine-readable :class:`DataQualityReport` written to
   ``outputs/reports/data_quality_report.json`` and rendered in the dashboard.

One deliberate design choice: **missing descriptions are flagged, not filled**.
An empty description is an audit red flag, so imputing "No description
provided" would destroy the very signal the suspicious-description rule is
looking for.

Usage::

    python src/data_cleaning.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/data_cleaning.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import (
    ACCOUNTS,
    BASE_CURRENCY,
    DATA_QUALITY_REPORT,
    EMPLOYEES_CLEAN,
    EMPLOYEES_RAW,
    FX_RATES_TO_CNY,
    TRANSACTIONS_CLEAN,
    TRANSACTIONS_RAW,
    VENDORS_CLEAN,
    VENDORS_RAW,
    Timer,
    ensure_directories,
    get_logger,
    save_json,
)

LOGGER = get_logger(__name__)

#: Currency aliases found in real ERP extracts, mapped to ISO codes.
CURRENCY_ALIASES: dict[str, str] = {
    "RMB": "CNY",
    "YUAN": "CNY",
    "人民币": "CNY",
    "US$": "USD",
    "USD$": "USD",
    "HK$": "HKD",
    "EUR€": "EUR",
}

#: Columns whose text values should be trimmed before comparison.
TEXT_COLUMNS: tuple[str, ...] = (
    "account_code",
    "account_name",
    "credit_account_code",
    "credit_account_name",
    "currency",
    "department",
    "payment_method",
    "document_type",
    "description",
    "vendor_id",
    "vendor_name",
    "employee_id",
    "created_by",
    "approved_by",
    "process",
)

DATE_COLUMNS: tuple[str, ...] = ("transaction_date", "posting_date")
TIMESTAMP_COLUMNS: tuple[str, ...] = ("approval_time", "invoice_time", "payment_time")


@dataclass
class DataQualityReport:
    """Structured summary of the data-quality assessment.

    Attributes:
        raw_rows: Row count of the raw extract.
        clean_rows: Row count after cleaning.
        duplicates_removed: Number of exact duplicate rows dropped.
        invalid_date_rows: Rows with an unparseable transaction date.
        invalid_posting_dates: Rows whose posting date was unparseable and had
            to be rebuilt from the transaction date.
        repaired_posting_dates: Rows whose posting date was actually rebuilt.
        invalid_amount_rows: Rows with a missing, zero or negative voucher amount.
        unbalanced_rows: Rows where the debit leg does not equal the credit leg.
        orphan_vendor_rows: Rows referencing a vendor missing from the master file.
        account_name_mismatches: Rows whose account name disagreed with the chart
            of accounts and were repaired from the account code.
        currency_normalised_rows: Rows whose currency code needed normalising.
        missing_values: Missing-value counts per column (raw extract).
        date_range: ``[min, max]`` transaction date of the clean ledger.
        total_transactions: Vouchers in the clean ledger.
        total_transaction_amount: Sum of the debit leg, in CNY.
        unique_vendors: Distinct vendors referenced.
        unique_accounts: Distinct accounts used.
        unique_voucher_creators: Distinct employees who raised a voucher, i.e.
            ``created_by``. Deliberately *not* the size of the employee master
            file, and the name says so. Roughly a quarter of the roster holds
            roles (HR, sales, executive) that never raise an AP voucher, so the
            two figures legitimately differ. An earlier version called this
            ``unique_employees``, which read as a contradiction of the
            ``employees`` table in the same set of reports.
    """

    raw_rows: int = 0
    clean_rows: int = 0
    duplicates_removed: int = 0
    invalid_date_rows: int = 0
    invalid_posting_dates: int = 0
    repaired_posting_dates: int = 0
    invalid_amount_rows: int = 0
    unbalanced_rows: int = 0
    orphan_vendor_rows: int = 0
    account_name_mismatches: int = 0
    currency_normalised_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    date_range: list[str] = field(default_factory=list)
    total_transactions: int = 0
    total_transaction_amount: float = 0.0
    unique_vendors: int = 0
    unique_accounts: int = 0
    unique_voucher_creators: int = 0
    repair_log: list[str] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        """Total number of data-quality issues detected."""
        return (
            self.duplicates_removed
            + self.invalid_date_rows
            + self.invalid_amount_rows
            + self.unbalanced_rows
            + self.orphan_vendor_rows
            + self.account_name_mismatches
            + self.currency_normalised_rows
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        payload = asdict(self)
        payload["total_issues"] = self.total_issues
        payload["issue_rate_pct"] = round(self.total_issues / max(1, self.raw_rows) * 100, 3)
        return payload


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw extracts produced by :mod:`src.data_generator`.

    Returns:
        Tuple of ``(transactions, vendors, employees)`` dataframes.

    Raises:
        FileNotFoundError: If the raw extract has not been generated yet.
    """
    if not TRANSACTIONS_RAW.exists():
        raise FileNotFoundError(
            f"{TRANSACTIONS_RAW} not found. Run `python src/data_generator.py` first."
        )
    transactions = pd.read_csv(TRANSACTIONS_RAW, dtype=str, keep_default_na=True)
    vendors = pd.read_csv(VENDORS_RAW, dtype=str)
    employees = pd.read_csv(EMPLOYEES_RAW, dtype=str)
    LOGGER.info("Loaded raw extract: %s transactions, %s vendors, %s employees",
                len(transactions), len(vendors), len(employees))
    return transactions, vendors, employees


# --------------------------------------------------------------------------- #
# Cleaning primitives
# --------------------------------------------------------------------------- #
def normalise_text(df: pd.DataFrame, columns: tuple[str, ...] = TEXT_COLUMNS) -> pd.DataFrame:
    """Trim whitespace and collapse repeated spaces in text columns."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            series = df[column].astype("string")
            df[column] = series.str.strip().str.replace(r"\s+", " ", regex=True)
            # Empty strings are missing values, not valid text.
            df[column] = df[column].mask(df[column] == "", pd.NA)
    return df


def normalise_currency(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalise currency codes to ISO 4217 and validate them.

    Handles the messy codes that survive ERP exports: lowercase, padded
    whitespace and local aliases such as ``RMB``.

    Returns:
        Tuple of ``(dataframe, number of rows whose code changed)``.
    """
    df = df.copy()
    original = df["currency"].astype("string").fillna("")

    cleaned = original.str.strip().str.upper()
    cleaned = cleaned.replace(CURRENCY_ALIASES)
    # Unknown codes fall back to the reporting currency but are counted, because
    # silently defaulting a currency would misstate the amount.
    unknown_mask = ~cleaned.isin(list(FX_RATES_TO_CNY.keys()))
    n_unknown = int(unknown_mask.sum())
    if n_unknown:
        LOGGER.warning("%s rows carry an unrecognised currency code; defaulted to %s",
                       n_unknown, BASE_CURRENCY)
    cleaned = cleaned.where(~unknown_mask, BASE_CURRENCY)

    df["currency"] = cleaned.astype("string")
    changed = int((df["currency"].fillna("") != original).sum())
    return df, changed


def parse_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Parse date and timestamp columns, repairing what can be repaired.

    * ``transaction_date`` - unparseable values become ``NaT``; the rows are
      dropped later because a voucher with no date cannot be placed in a period.
    * ``posting_date`` - if unparseable, it is rebuilt from ``transaction_date``
      (the standard repair: the posting date is almost always the transaction
      date or within a few days of it).

    Returns:
        Tuple of ``(dataframe, invalid_transaction_dates, invalid_posting_dates,
        repaired_posting_dates)``. The first two are counted *before* repair so
        that the quality report reflects what actually arrived from the ERP.
    """
    df = df.copy()

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

    for column in TIMESTAMP_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

    invalid_transaction_dates = int(df["transaction_date"].isna().sum())

    missing_posting = df["posting_date"].isna()
    invalid_posting_dates = int(missing_posting.sum())
    repaired = invalid_posting_dates
    if repaired:
        df.loc[missing_posting, "posting_date"] = df.loc[missing_posting, "transaction_date"]

    return df, invalid_transaction_dates, invalid_posting_dates, repaired


#: Characters that separate thousands in exported figures, plus the non-breaking
#: space that spreadsheets substitute for a normal space. Written as literals
#: rather than ``\u`` escapes because pyarrow's regex engine (RE2) rejects the
#: ``\uXXXX`` form, and pandas routes string replacement through pyarrow.
_THOUSANDS_NOISE = ",\u00a0 \t"
#: Anything that is not part of a number once the noise above is removed.
_CURRENCY_NOISE = "[^0-9.+eE-]"


def coerce_numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce the given columns to ``float``, turning junk into ``NaN``.

    Money columns arriving as text from an ERP export routinely carry thousands
    separators (``1,234.56``), non-breaking spaces and currency symbols, and
    finance systems write negatives as ``(1,234.56)``. A bare ``pd.to_numeric``
    turns every one of those into ``NaN``, which the caller then treats as a
    missing amount - so the row is dropped and the population is understated
    without anyone noticing. They are cleaned up first.

    Columns that are already numeric take a direct path, so this is a no-op for
    the pipeline's own intermediate files and only bites on genuinely dirty input.
    """
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue

        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            df[column] = pd.to_numeric(series, errors="coerce")
            continue

        text = series.astype("string")
        # Accounting convention: a bracketed figure is negative.
        negative = text.str.strip().str.match(r"^\(.*\)$", na=False)
        cleaned = text.str.replace(f"[{_THOUSANDS_NOISE}]", "", regex=True)
        cleaned = cleaned.str.replace(_CURRENCY_NOISE, "", regex=True)
        cleaned = cleaned.mask(cleaned.str.strip() == "", pd.NA)
        converted = pd.to_numeric(cleaned, errors="coerce")
        df[column] = converted.mask(negative, -converted.abs())
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows, keeping the first occurrence.

    Duplicates are compared on the full business key rather than on
    ``transaction_id`` alone, because a duplicated ``transaction_id`` with a
    different amount is a *different* problem (a broken key) from a fully
    duplicated extract row.
    """
    business_key = [
        column
        for column in (
            "transaction_id",
            "journal_id",
            "transaction_date",
            "account_code",
            "debit_amount",
            "credit_amount",
            "vendor_id",
            "description",
        )
        if column in df.columns
    ]
    before = len(df)
    deduped = df.drop_duplicates(subset=business_key, keep="first").reset_index(drop=True)
    return deduped, before - len(deduped)


def repair_account_names(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Restore account names from the chart of accounts.

    When a voucher's account name disagrees with its account code, the code is
    authoritative - the name is a descriptive attribute that is frequently
    broken by manual journal uploads.
    """
    df = df.copy()
    authoritative = df["account_code"].astype("string").map(
        lambda code: ACCOUNTS.get(str(code), {}).get("name")
    )
    mismatch = authoritative.notna() & (df["account_name"].astype("string") != authoritative)
    n_mismatch = int(mismatch.sum())
    if n_mismatch:
        df.loc[mismatch, "account_name"] = authoritative[mismatch]
    return df, n_mismatch


def flag_orphan_vendors(
    df: pd.DataFrame, vendors: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Flag vouchers that reference a vendor absent from the master file.

    Orphan foreign keys are kept in the ledger (dropping them would understate
    the population) but they are marked so that the auditor can chase the
    master-data gap.
    """
    df = df.copy()
    valid_ids = set(vendors["vendor_id"].dropna().astype(str))
    has_vendor = df["vendor_id"].notna()
    orphan = has_vendor & ~df["vendor_id"].astype("string").isin(valid_ids)
    n_orphan = int(orphan.sum())
    df["orphan_vendor_flag"] = orphan
    return df, n_orphan


def flag_unbalanced(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Flag vouchers where the debit leg does not equal the credit leg.

    This is the single most important control in double-entry bookkeeping, so a
    breach is flagged rather than silently repaired.
    """
    df = df.copy()
    difference = (df["debit_amount"] - df["credit_amount"]).abs()
    unbalanced = difference > 0.01
    df["unbalanced_flag"] = unbalanced.fillna(False)
    return df, int(unbalanced.sum())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def clean_transactions(
    transactions: pd.DataFrame,
    vendors: pd.DataFrame,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Run the full cleaning pipeline over the transaction extract.

    Args:
        transactions: Raw voucher extract.
        vendors: Vendor master data (used for referential-integrity checks).

    Returns:
        Tuple of ``(clean transactions, quality report)``.
    """
    report = DataQualityReport(raw_rows=len(transactions))

    # ---- 1. Profile the raw extract before touching anything ----------------
    missing = transactions.isna().sum()
    report.missing_values = {
        column: int(count) for column, count in missing.items() if int(count) > 0
    }

    df = normalise_text(transactions)
    df, currency_fixed = normalise_currency(df)
    report.currency_normalised_rows = currency_fixed
    if currency_fixed:
        report.repair_log.append(f"Normalised {currency_fixed} messy currency codes to ISO 4217.")

    df, invalid_dates, invalid_posting, repaired_posting = parse_dates(df)
    report.invalid_date_rows = invalid_dates + invalid_posting
    report.invalid_posting_dates = invalid_posting
    report.repaired_posting_dates = repaired_posting
    if repaired_posting:
        report.repair_log.append(
            f"Rebuilt {repaired_posting} posting dates from the transaction date."
        )

    df = coerce_numeric(
        df,
        ("debit_amount", "credit_amount", "amount", "amount_original", "fx_rate"),
    )

    # Rows with an unusable amount cannot be analysed and are removed.
    unusable_amount = df["amount"].isna() | (df["amount"] <= 0)
    report.invalid_amount_rows = int(unusable_amount.sum())
    if report.invalid_amount_rows:
        report.repair_log.append(
            f"Removed {report.invalid_amount_rows} vouchers with a missing, zero or negative amount."
        )
    df = df.loc[~unusable_amount].copy()

    # Rows with no transaction date cannot be assigned to an audit period.
    undated = df["transaction_date"].isna()
    if int(undated.sum()):
        report.repair_log.append(f"Removed {int(undated.sum())} vouchers with an unparseable date.")
    df = df.loc[~undated].copy()

    # ---- 2. Structural repairs ---------------------------------------------
    df, duplicates_removed = remove_duplicate_rows(df)
    report.duplicates_removed = duplicates_removed
    if duplicates_removed:
        report.repair_log.append(f"Removed {duplicates_removed} exact duplicate rows.")

    df, mismatches = repair_account_names(df)
    report.account_name_mismatches = mismatches
    if mismatches:
        report.repair_log.append(
            f"Repaired {mismatches} account names from the chart of accounts."
        )

    df, orphan_count = flag_orphan_vendors(df, vendors)
    report.orphan_vendor_rows = orphan_count
    if orphan_count:
        report.repair_log.append(
            f"Flagged {orphan_count} vouchers referencing an unknown vendor (master-data gap)."
        )

    df, unbalanced_count = flag_unbalanced(df)
    report.unbalanced_rows = unbalanced_count
    if unbalanced_count:
        report.repair_log.append(
            f"Flagged {unbalanced_count} unbalanced vouchers (debit leg != credit leg)."
        )

    # ---- 3. Derived columns used downstream ---------------------------------
    # The functional currency is CNY; foreign documents are translated so that
    # every downstream aggregate is apples-to-apples.
    df["amount_cny"] = (df["amount_original"] * df["fx_rate"]).round(2)
    # Where the translation is missing, fall back to the functional amount.
    df["amount_cny"] = df["amount_cny"].fillna(df["amount"]).round(2)

    df["description_missing"] = df["description"].isna()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["posting_date"] = pd.to_datetime(df["posting_date"])
    df["year_month"] = df["transaction_date"].dt.to_period("M").astype(str)

    df = df.sort_values("transaction_date").reset_index(drop=True)

    # ---- 4. Summary statistics ---------------------------------------------
    report.clean_rows = len(df)
    report.total_transactions = len(df)
    report.total_transaction_amount = float(df["debit_amount"].sum())
    report.unique_vendors = int(df["vendor_id"].nunique())
    report.unique_accounts = int(df["account_code"].nunique())
    report.unique_voucher_creators = int(df["created_by"].nunique())
    report.date_range = [
        str(df["transaction_date"].min().date()),
        str(df["transaction_date"].max().date()),
    ]

    LOGGER.info(
        "Cleaning complete: %s -> %s rows | %s issues | %.3f%% issue rate",
        report.raw_rows,
        report.clean_rows,
        report.total_issues,
        report.to_dict()["issue_rate_pct"],
    )
    return df, report


def clean_master_data(
    vendors: pd.DataFrame, employees: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the vendor and employee master files.

    Master data is small and authoritative, so cleaning is limited to trimming
    text, parsing the registration date and de-duplicating on the primary key.
    """
    vendors = normalise_text(vendors, ("vendor_id", "vendor_name", "vendor_category", "bank_account", "country", "risk_level"))
    vendors["registration_date"] = pd.to_datetime(vendors["registration_date"], errors="coerce", format="mixed")
    vendors = vendors.dropna(subset=["vendor_id"]).drop_duplicates(subset=["vendor_id"], keep="first")

    employees = normalise_text(employees, ("employee_id", "employee_name", "department", "role"))
    employees = employees.dropna(subset=["employee_id"]).drop_duplicates(subset=["employee_id"], keep="first")

    LOGGER.info("Cleaned master data: %s vendors, %s employees", len(vendors), len(employees))
    return vendors.reset_index(drop=True), employees.reset_index(drop=True)


def run_cleaning() -> dict[str, Any]:
    """Execute the cleaning pipeline end to end and persist its outputs.

    Returns:
        Mapping with ``transactions``, ``vendors``, ``employees`` and ``report``.
    """
    ensure_directories()
    transactions_raw, vendors_raw, employees_raw = load_raw_data()

    transactions, report = clean_transactions(transactions_raw, vendors_raw)
    vendors, employees = clean_master_data(vendors_raw, employees_raw)

    transactions.to_parquet(TRANSACTIONS_CLEAN, index=False)
    vendors.to_parquet(VENDORS_CLEAN, index=False)
    employees.to_parquet(EMPLOYEES_CLEAN, index=False)
    save_json(report.to_dict(), DATA_QUALITY_REPORT)

    LOGGER.info("Wrote %s", TRANSACTIONS_CLEAN)
    LOGGER.info("Wrote %s", DATA_QUALITY_REPORT)

    return {
        "transactions": transactions,
        "vendors": vendors,
        "employees": employees,
        "report": report,
    }


def main() -> int:
    """CLI entry point."""
    with Timer("data cleaning"):
        result = run_cleaning()

    report: DataQualityReport = result["report"]
    LOGGER.info("--- Data Quality Report ---")
    for key, value in report.to_dict().items():
        if key in ("missing_values", "repair_log"):
            continue
        LOGGER.info("  %-28s %s", key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
