"""Shared utilities for AuditLens: project paths, logging, IO helpers and business constants.

Every module in ``src`` imports its paths and business constants from here so that
there is a single source of truth. All paths are derived from the location of this
file, so the project can be cloned and run from anywhere (no hard-coded absolute
paths anywhere in the code base).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # allows `python src/<module>.py` to work
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
CHART_DIR: Path = OUTPUT_DIR / "charts"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
SQL_DIR: Path = PROJECT_ROOT / "sql"

DB_PATH: Path = DATA_DIR / "auditlens.db"

# Raw / processed artefacts produced by the pipeline.
TRANSACTIONS_RAW: Path = RAW_DIR / "transactions_raw.csv"
VENDORS_RAW: Path = RAW_DIR / "vendors_raw.csv"
EMPLOYEES_RAW: Path = RAW_DIR / "employees_raw.csv"

TRANSACTIONS_CLEAN: Path = PROCESSED_DIR / "transactions_clean.parquet"
VENDORS_CLEAN: Path = PROCESSED_DIR / "vendors_clean.parquet"
EMPLOYEES_CLEAN: Path = PROCESSED_DIR / "employees_clean.parquet"

SCORED_TRANSACTIONS: Path = PROCESSED_DIR / "transactions_scored.parquet"
TRANSACTIONS_FEATURES: Path = PROCESSED_DIR / "transactions_features.parquet"
VENDOR_RISK_TABLE: Path = PROCESSED_DIR / "vendor_risk.parquet"

DATA_QUALITY_REPORT: Path = REPORT_DIR / "data_quality_report.json"
AUDIT_SUMMARY_REPORT: Path = REPORT_DIR / "audit_summary.json"
RULE_ALERTS_CSV: Path = REPORT_DIR / "rule_alerts.csv"
RULE_EVALUATION_CSV: Path = REPORT_DIR / "rule_evaluation.csv"
MODEL_METRICS_JSON: Path = REPORT_DIR / "model_metrics.json"
BENFORD_RESULTS_JSON: Path = REPORT_DIR / "benford_results.json"
HIGH_RISK_CSV: Path = REPORT_DIR / "high_risk_transactions.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Business constants
# --------------------------------------------------------------------------- #
#: Payment approval threshold (CNY). Payments at or above this value require a
#: second-level approval. Transactions sitting *just below* this line are a
#: classic split-payment red flag.
APPROVAL_THRESHOLD_CNY: float = 50_000.0

#: Planning-materiality proxy (CNY) used by the large-round-amount rule. Auditors
#: set a materiality threshold at the start of an engagement and concentrate on
#: items above it; below it, findings are rarely worth reporting. Without a floor,
#: a "large round amount" test flags thousands of immaterial round payments.
MATERIALITY_THRESHOLD_CNY: float = 100_000.0

#: Window used when defining "just below the approval threshold".
SPLIT_THRESHOLD_LOW_RATIO: float = 0.90  # 45,000
SPLIT_THRESHOLD_HIGH_RATIO: float = 0.995  # 49,750

#: Payments made within this many days of the invoice date are "rapid".
RAPID_PAYMENT_HOURS: float = 6.0

#: A vendor is considered "new" when registered less than this many days before
#: the transaction.
NEW_VENDOR_DAYS: int = 90

#: Illustrative PRC statutory public holidays covering the ledger period.
#: A production system would load the official State Council calendar; this
#: hard-coded list keeps the project self-contained and reproducible. Dates are
#: inclusive. Weekends are handled separately.
PUBLIC_HOLIDAYS: tuple[str, ...] = (
    # 2024
    "2024-01-01",
    "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16",
    "2024-04-04", "2024-04-05",
    "2024-05-01", "2024-05-02", "2024-05-03",
    "2024-06-10",
    "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07",
    # 2025
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-03", "2025-02-04",
    "2025-04-04",
    "2025-05-01", "2025-05-02", "2025-05-05",
    "2025-06-02",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08",
)

#: Default reporting currency. Foreign-currency journals are translated using
#: ``FX_RATES_TO_CNY``.
BASE_CURRENCY: str = "CNY"

FX_RATES_TO_CNY: dict[str, float] = {
    "CNY": 1.0,
    "USD": 7.15,
    "EUR": 7.80,
    "HKD": 0.92,
    "JPY": 0.048,
    "SGD": 5.30,
}

#: Chart of accounts (simplified Chinese Accounting Standards for Business
#: Enterprises, 企业会计准则). ``normal_balance`` is "D" for debit-normal
#: accounts (assets, expenses) and "C" for credit-normal accounts
#: (liabilities, equity, revenue).
ACCOUNTS: dict[str, dict[str, str]] = {
    "1001": {"name": "库存现金 Cash on hand", "type": "Asset", "normal_balance": "D"},
    "1002": {"name": "银行存款 Cash at bank", "type": "Asset", "normal_balance": "D"},
    "1122": {"name": "应收账款 Accounts receivable", "type": "Asset", "normal_balance": "D"},
    "1123": {"name": "预付账款 Prepayments", "type": "Asset", "normal_balance": "D"},
    "1221": {"name": "其他应收款 Other receivables", "type": "Asset", "normal_balance": "D"},
    "1403": {"name": "原材料 Raw materials", "type": "Asset", "normal_balance": "D"},
    "1405": {"name": "库存商品 Finished goods", "type": "Asset", "normal_balance": "D"},
    "1601": {"name": "固定资产 Fixed assets", "type": "Asset", "normal_balance": "D"},
    "1602": {"name": "累计折旧 Accumulated depreciation", "type": "Asset", "normal_balance": "C"},
    "2202": {"name": "应付账款 Accounts payable", "type": "Liability", "normal_balance": "C"},
    "2203": {"name": "预收账款 Advances from customers", "type": "Liability", "normal_balance": "C"},
    "2211": {"name": "应付职工薪酬 Payroll payable", "type": "Liability", "normal_balance": "C"},
    "2221": {"name": "应交税费 Taxes payable", "type": "Liability", "normal_balance": "C"},
    "6001": {"name": "主营业务收入 Revenue", "type": "Revenue", "normal_balance": "C"},
    "6051": {"name": "其他业务收入 Other operating income", "type": "Revenue", "normal_balance": "C"},
    "6401": {"name": "主营业务成本 Cost of sales", "type": "Expense", "normal_balance": "D"},
    "6601": {"name": "销售费用 Selling expenses", "type": "Expense", "normal_balance": "D"},
    "6602": {"name": "管理费用 Admin expenses", "type": "Expense", "normal_balance": "D"},
    "660201": {"name": "管理费用-差旅费 Travel expenses", "type": "Expense", "normal_balance": "D"},
    "660202": {"name": "管理费用-办公费 Office expenses", "type": "Expense", "normal_balance": "D"},
    "660203": {"name": "管理费用-业务招待费 Entertainment", "type": "Expense", "normal_balance": "D"},
    "660204": {"name": "管理费用-会议费 Conference expenses", "type": "Expense", "normal_balance": "D"},
    "660205": {"name": "管理费用-咨询费 Consulting fees", "type": "Expense", "normal_balance": "D"},
    "660206": {"name": "管理费用-水电费 Utilities", "type": "Expense", "normal_balance": "D"},
    "6603": {"name": "财务费用 Finance costs", "type": "Expense", "normal_balance": "D"},
}

#: Accounts that are only expected to be used occasionally. Heavy or unusual
#: postings to these accounts are an audit red flag ("rare account usage").
RARE_ACCOUNTS: tuple[str, ...] = ("1221", "1405", "1602", "2203", "6051", "660204")

DEPARTMENTS: tuple[str, ...] = (
    "Finance",
    "Procurement",
    "Sales",
    "Operations",
    "IT",
    "HR",
    "Marketing",
    "R&D",
)

VENDOR_CATEGORIES: tuple[str, ...] = (
    "IT Services",
    "Logistics",
    "Raw Materials",
    "Professional Services",
    "Facilities",
    "Marketing Agency",
    "Travel Agency",
    "Equipment Supplier",
)

PAYMENT_METHODS: tuple[str, ...] = (
    "Bank Transfer",
    "Cheque",
    "Corporate Card",
    "Cash",
    "Letter of Credit",
)

DOCUMENT_TYPES: tuple[str, ...] = (
    "Journal Voucher",
    "Vendor Invoice",
    "Payment Voucher",
    "Expense Claim",
    "Receipt",
)

#: Business processes driving the synthetic ledger. Each entry maps a process to
#: (debit account, credit account, typical monthly volume weight, amount band).
BUSINESS_PROCESSES: dict[str, dict[str, Any]] = {
    "procurement": {"debit": "1403", "credit": "2202", "weight": 0.16, "amount": (5_000, 400_000)},
    "vendor_payment": {"debit": "2202", "credit": "1002", "weight": 0.20, "amount": (8_000, 500_000)},
    "sales": {"debit": "1122", "credit": "6001", "weight": 0.15, "amount": (10_000, 900_000)},
    "cost_of_sales": {"debit": "6401", "credit": "1405", "weight": 0.08, "amount": (8_000, 300_000)},
    "travel": {"debit": "660201", "credit": "1002", "weight": 0.11, "amount": (800, 45_000)},
    "office_expense": {"debit": "660202", "credit": "1002", "weight": 0.09, "amount": (500, 30_000)},
    "entertainment": {"debit": "660203", "credit": "1002", "weight": 0.04, "amount": (1_000, 25_000)},
    "consulting": {"debit": "660205", "credit": "2202", "weight": 0.04, "amount": (20_000, 350_000)},
    "utilities": {"debit": "660206", "credit": "1002", "weight": 0.04, "amount": (2_000, 60_000)},
    "payroll": {"debit": "2211", "credit": "1002", "weight": 0.05, "amount": (200_000, 1_200_000)},
    "fixed_asset": {"debit": "1601", "credit": "1002", "weight": 0.02, "amount": (50_000, 1_500_000)},
    "tax_payment": {"debit": "2221", "credit": "1002", "weight": 0.02, "amount": (30_000, 600_000)},
}

#: Vendor-facing processes (a vendor is attached to these journals).
VENDOR_PROCESSES: tuple[str, ...] = (
    "procurement",
    "vendor_payment",
    "cost_of_sales",
    "consulting",
    "utilities",
    "office_expense",
)

#: Employee-facing processes (an employee / expense claimant is attached).
EMPLOYEE_PROCESSES: tuple[str, ...] = ("travel", "entertainment", "office_expense")

#: Processes that are backed by a supplier invoice, and therefore carry the full
#: invoice -> approval -> payment control timeline.
INVOICE_PROCESSES: tuple[str, ...] = (
    "vendor_payment",
    "procurement",
    "consulting",
    "utilities",
    "cost_of_sales",
)

# --------------------------------------------------------------------------- #
# Anomaly taxonomy (ground truth injected into the synthetic data)
# --------------------------------------------------------------------------- #
ANOMALY_TYPES: tuple[str, ...] = (
    "duplicate_payment",
    "weekend_posting",
    "large_round_amount",
    "unusual_vendor",
    "split_transaction",
    "self_approval",
    "rapid_payment",
    "suspicious_description",
    "rare_account_usage",
)

ANOMALY_TYPE_LABELS: dict[str, str] = {
    "duplicate_payment": "Duplicate Payment",
    "weekend_posting": "Weekend / Holiday Posting",
    "large_round_amount": "Large Round Amount",
    "unusual_vendor": "Unusual Vendor",
    "split_transaction": "Split Transaction",
    "self_approval": "Self Approval",
    "rapid_payment": "Rapid Payment",
    "suspicious_description": "Suspicious Description",
    "rare_account_usage": "Rare Account Usage",
}

#: Keywords used by the description-based rule. Deliberately simple and
#: transparent - a keyword list an auditor can read and challenge, not a black box.
SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "urgent",
    "manual adjustment",
    "special payment",
    "miscellaneous",
    "temporary",
    "adjustment",
    "misc",
    "other",
    "rounding",
    "no invoice",
    "pending invoice",
    "advance payment",
    "one-off",
    "unknown",
)

# --------------------------------------------------------------------------- #
# Benford's Law - expected first-digit distribution
# --------------------------------------------------------------------------- #
BENFORD_EXPECTED: dict[int, float] = {
    1: 0.30103,
    2: 0.17609,
    3: 0.12494,
    4: 0.09691,
    5: 0.07918,
    6: 0.06695,
    7: 0.05799,
    8: 0.05115,
    9: 0.04576,
}

#: Nigrini's mean-absolute-deviation conformity thresholds for first-digit tests.
BENFORD_MAD_THRESHOLDS: dict[str, float] = {
    "close_conformity": 0.006,
    "acceptable_conformity": 0.012,
    "marginal_conformity": 0.015,
}

# --------------------------------------------------------------------------- #
# Risk scoring configuration
# --------------------------------------------------------------------------- #
#: Weights of the composite Audit Risk Score (must sum to 1.0).
#: Rationale is documented in ``docs/methodology.md`` and in ``src/risk_scoring.py``.
RISK_WEIGHTS: dict[str, float] = {
    "rule_risk": 0.40,
    "ml_risk": 0.25,
    "vendor_risk": 0.15,
    "amount_risk": 0.10,
    "statistical_risk": 0.10,
}

#: Risk bands (lower bound inclusive, upper bound exclusive).
RISK_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Low", 0.0, 30.0),
    ("Medium", 30.0, 60.0),
    ("High", 60.0, 80.0),
    ("Critical", 80.0, 100.01),
)

RISK_LEVEL_ORDER: tuple[str, ...] = ("Low", "Medium", "High", "Critical")

# --------------------------------------------------------------------------- #
# Presentation palette (kept consistent between charts and the dashboard)
# --------------------------------------------------------------------------- #
RISK_COLORS: dict[str, str] = {
    "Low": "#2E9E6B",
    "Medium": "#E8A33D",
    "High": "#E2703A",
    "Critical": "#C0392B",
}

THEME_PRIMARY: str = "#1F4E79"
THEME_ACCENT: str = "#3C8DBC"
THEME_MUTED: str = "#8A9BA8"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Args:
        name: Logger name, normally ``__name__`` of the calling module.
        level: Logging level, defaults to :data:`logging.INFO`.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def ensure_directories() -> None:
    """Create every directory the pipeline writes to (idempotent)."""
    for directory in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        OUTPUT_DIR,
        CHART_DIR,
        REPORT_DIR,
        SQL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path, indent: int = 2) -> Path:
    """Serialise ``payload`` to JSON, creating parent directories as needed.

    ``numpy`` scalars are converted to native Python types so the output is
    readable and portable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=_default)
    return path


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file produced by :func:`save_json`."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Persist a dataframe to parquet (or CSV when the suffix is ``.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_parquet(path, index=False)
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe written by :func:`save_dataframe`."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def format_cny(value: float) -> str:
    """Format a number as a compact CNY string, e.g. ``12.4M`` / ``850.3K``."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"¥{value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"¥{value / 1_000_000:,.2f}M"
    if abs_value >= 1_000:
        return f"¥{value / 1_000:,.1f}K"
    return f"¥{value:,.0f}"


def risk_level_from_score(score: float) -> str:
    """Map a 0-100 risk score onto one of the four audit risk bands."""
    for label, lower, upper in RISK_BANDS:
        if lower <= score < upper:
            return label
    return "Critical"


def percentile_rank(series: pd.Series) -> pd.Series:
    """Return the percentile rank (0-1) of each value in ``series``."""
    return series.rank(pct=True, method="average")


def as_list(value: Any, separator: str | None = None) -> list[Any]:
    """Coerce ``value`` into a list (``None`` becomes an empty list).

    Explanation columns travel in three shapes depending on where they were read
    from: a Python list (in memory), a ``numpy.ndarray`` (after a parquet
    round-trip) and a delimited string (the ``*_reason_text`` columns written to
    CSV). Passing ``separator`` makes the string case split back into items, so
    the dashboard can render any of the three identically.

    Args:
        value: The raw cell value.
        separator: When set and ``value`` is a string, split on this delimiter.

    Returns:
        A list of items. Scalars are wrapped in a single-element list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        if separator:
            return [item.strip() for item in value.split(separator) if item.strip()]
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, Iterable):
        return list(value)
    return [value]


#: Delimiter used when explanation lists are flattened into a single text column.
REASON_SEPARATOR: str = "; "


def as_reason_list(value: Any) -> list[str]:
    """Normalise an explanation cell into a list of human-readable reasons."""
    return [str(item) for item in as_list(value, separator=REASON_SEPARATOR)]


def as_flag_series(series: pd.Series) -> pd.Series:
    """Coerce a 0/1 column into booleans, whatever dtype it arrived in.

    The ground-truth label survives a parquet round-trip as **text** (``"0"`` /
    ``"1"``), and text breaks every obvious way of reading it:

    * ``series == 1`` is ``False`` everywhere, because ``"1" != 1``.
    * ``series.astype(bool)`` is ``True`` everywhere, because a non-empty string is
      truthy - so ``"0"`` reads as an anomaly.

    Both of those bugs were present in this codebase at different times. The second
    one is the more dangerous, because it fails silently in the direction of
    over-flagging rather than under-flagging. Coercing through numeric first is the
    only version that is correct for every dtype the label actually takes.

    Args:
        series: A 0/1 column as int, float, bool, nullable integer or string.

    Returns:
        A boolean Series aligned to ``series.index``.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float).gt(0.5)


@dataclass
class Timer:
    """Tiny context manager used to time pipeline stages."""

    label: str

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        elapsed = time.perf_counter() - self._start
        get_logger("auditlens.timer").info("%s finished in %.2fs", self.label, elapsed)
