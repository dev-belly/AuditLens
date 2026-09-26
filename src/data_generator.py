"""Synthetic financial dataset generator for AuditLens.

Real corporate ledgers cannot be published, so AuditLens ships a generator that
produces a *realistic* general-ledger extract for a mid-sized Chinese
manufacturing / trading group. The generator is not random noise: it models
double-entry bookkeeping, a chart of accounts, business processes, vendor
lifecycles, approval hierarchies and working-day behaviour.

Two things are deliberately injected on top of the clean ledger:

1. **Anomalies** (2%-5% of vouchers) across nine audit-relevant patterns, each
   labelled with ``anomaly_label`` / ``anomaly_type``. These labels are the
   *ground truth* used to evaluate the rules and the machine-learning model.
   They are never shown on the auditor-facing dashboard.
2. **Data-quality defects** (~0.6% of rows): missing values, duplicate rows,
   malformed dates, unbalanced entries, invalid vendor references and messy
   currency codes. These make the cleaning pipeline meaningful instead of
   decorative.

Typical usage::

    python src/data_generator.py --journals 30000

Outputs (see ``src/utils.py`` for the exact paths):

* ``data/raw/transactions_raw.csv``
* ``data/raw/vendors_raw.csv``
* ``data/raw/employees_raw.csv``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/data_generator.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import (
    ACCOUNTS,
    BUSINESS_PROCESSES,
    DEPARTMENTS,
    DOCUMENT_TYPES,
    EMPLOYEE_PROCESSES,
    EMPLOYEES_RAW,
    FX_RATES_TO_CNY,
    INVOICE_PROCESSES,
    PAYMENT_METHODS,
    PUBLIC_HOLIDAYS,
    RANDOM_SEED,
    TRANSACTIONS_RAW,
    VENDOR_CATEGORIES,
    VENDOR_PROCESSES,
    VENDORS_RAW,
    Timer,
    ensure_directories,
    get_logger,
)

from src.anomaly_injection import inject_anomalies

LOGGER = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Name banks - used to build believable vendor / employee master data
# --------------------------------------------------------------------------- #
VENDOR_NAME_PREFIXES: tuple[str, ...] = (
    "Apex", "Blue Ridge", "Cathay", "Delta", "Everest", "Fortune", "Global",
    "Hanwei", "Innovate", "Jinlong", "Kaiyuan", "Longhe", "Mingda", "Nordic",
    "Orient", "Pacific", "Qianhai", "Ruixin", "Sunrise", "Tianhe", "Union",
    "Vertex", "Wanhe", "Xinrun", "Yongsheng", "Zhongtai", "Zenith", "Bright",
    "Concord", "Dynamic", "Hengtai", "Silverline", "Northwind", "Kunlun",
)

VENDOR_NAME_CORES: tuple[str, ...] = (
    "Technology", "Logistics", "Materials", "Consulting", "Services",
    "Industrial", "Trading", "Engineering", "Media", "Supply Chain", "Systems",
    "Facilities", "Automation", "Packaging", "Electronics",
)

VENDOR_NAME_SUFFIXES: tuple[str, ...] = (
    "Co., Ltd.", "Group", "Holdings", "(Shanghai) Co., Ltd.",
    "(Shenzhen) Co., Ltd.", "(Beijing) Co., Ltd.", "(Guangzhou) Co., Ltd.",
)

SURNAMES: tuple[str, ...] = (
    "Chen", "Li", "Wang", "Zhang", "Liu", "Yang", "Huang", "Zhao", "Wu",
    "Zhou", "Xu", "Sun", "Ma", "Zhu", "Hu", "Guo", "He", "Lin", "Gao", "Luo",
)

GIVEN_NAMES: tuple[str, ...] = (
    "Wei", "Fang", "Min", "Jing", "Lei", "Qiang", "Yan", "Hui", "Tao", "Na",
    "Bo", "Xin", "Yu", "Kai", "Rui", "Cheng", "Dan", "Feng", "Gang", "Hong",
)

#: Employee roles. ``approval_authority`` marks roles that are allowed to
#: approve payments; ``creator`` marks roles that raise vouchers.
ROLE_CATALOGUE: tuple[dict[str, Any], ...] = (
    {"role": "AP Clerk", "department": "Finance", "creator": True, "approver": False, "share": 0.14},
    {"role": "Staff Accountant", "department": "Finance", "creator": True, "approver": False, "share": 0.14},
    {"role": "General Ledger Accountant", "department": "Finance", "creator": True, "approver": False, "share": 0.10},
    {"role": "Treasury Analyst", "department": "Finance", "creator": True, "approver": False, "share": 0.08},
    {"role": "Finance Manager", "department": "Finance", "creator": False, "approver": True, "share": 0.09},
    {"role": "Financial Controller", "department": "Finance", "creator": False, "approver": True, "share": 0.05},
    {"role": "Chief Financial Officer", "department": "Finance", "creator": False, "approver": True, "share": 0.02},
    {"role": "Procurement Officer", "department": "Procurement", "creator": True, "approver": False, "share": 0.10},
    {"role": "Procurement Manager", "department": "Procurement", "creator": False, "approver": True, "share": 0.06},
    {"role": "Sales Executive", "department": "Sales", "creator": True, "approver": False, "share": 0.08},
    {"role": "Operations Specialist", "department": "Operations", "creator": True, "approver": False, "share": 0.06},
    {"role": "Department Head", "department": "Operations", "creator": False, "approver": True, "share": 0.05},
    {"role": "HR Specialist", "department": "HR", "creator": True, "approver": False, "share": 0.03},
)

COUNTRY_CHOICES: tuple[str, ...] = (
    "China", "China", "China", "China", "China", "China", "China",
    "United States", "Germany", "Japan", "Singapore", "Hong Kong, China",
)

CITIES: tuple[str, ...] = (
    "Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou", "Chengdu",
    "Wuhan", "Nanjing", "Xi'an", "Suzhou",
)

OFFICE_ITEMS: tuple[str, ...] = (
    "stationery", "printer toner", "office chairs", "laptops", "monitors",
    "meeting room equipment", "pantry supplies", "cleaning services",
)

#: Description templates per business process. ``{vendor}`` / ``{employee}`` /
#: ``{city}`` / ``{item}`` / ``{po}`` placeholders are filled at generation time.
DESCRIPTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "procurement": (
        "Purchase of raw materials - PO{po}",
        "Material procurement from {vendor} - PO{po}",
        "Goods receipt against PO{po}",
        "Bulk purchase of components - PO{po}",
    ),
    "vendor_payment": (
        "Payment to {vendor} for invoice {invoice}",
        "Settlement of {vendor} invoice {invoice}",
        "Bank transfer to {vendor} - invoice {invoice}",
        "Vendor settlement {vendor} / {invoice}",
    ),
    "sales": (
        "Sales invoice to {vendor} - SO{po}",
        "Revenue recognition for order SO{po}",
        "Goods delivered to {vendor} - SO{po}",
    ),
    "cost_of_sales": (
        "Cost of goods sold - shipment SO{po}",
        "Transfer of finished goods to cost of sales",
    ),
    "travel": (
        "Travel reimbursement - {city} business trip",
        "Business trip expenses to {city}",
        "Travel and accommodation - {city} client visit",
    ),
    "office_expense": (
        "Office supplies purchase - {item}",
        "Purchase of {item}",
        "Administrative supplies - {item}",
    ),
    "entertainment": (
        "Client entertainment - {city}",
        "Business meals with client representatives",
        "Customer hospitality expenses",
    ),
    "consulting": (
        "Consulting services from {vendor} - contract {po}",
        "Advisory fee to {vendor}",
        "Professional services {vendor} - engagement {po}",
    ),
    "utilities": (
        "Monthly utility charges - {city} plant",
        "Electricity and water charges",
        "Utility settlement for {city} facility",
    ),
    "payroll": (
        "Monthly payroll - {department} department",
        "Salary payment for {department} team",
        "Payroll disbursement {department}",
    ),
    "fixed_asset": (
        "Acquisition of production equipment",
        "Purchase of fixed asset - machinery",
        "Capital expenditure on plant equipment",
    ),
    "tax_payment": (
        "VAT payment for the period",
        "Corporate income tax instalment",
        "Tax settlement - {department}",
    ),
}

#: Account that a given process is *normally* posted to on the debit side.
#: Used by the "rare account usage" anomaly and by the account-frequency
#: features.
PROCESS_DEPARTMENT: dict[str, str] = {
    "procurement": "Procurement",
    "vendor_payment": "Finance",
    "sales": "Sales",
    "cost_of_sales": "Operations",
    "travel": "Sales",
    "office_expense": "Operations",
    "entertainment": "Sales",
    "consulting": "Operations",
    "utilities": "Operations",
    "payroll": "HR",
    "fixed_asset": "Operations",
    "tax_payment": "Finance",
}

#: Monthly seasonality multiplier. December carries a year-end close spike and
#: February is depressed by the Chinese New Year holiday - both visible in the
#: dashboard's monthly trend chart.
MONTHLY_SEASONALITY: tuple[float, ...] = (
    0.90, 0.72, 1.05, 1.00, 1.02, 1.10, 0.95, 0.96, 1.05, 1.00, 1.12, 1.38,
)


@dataclass
class GeneratorConfig:
    """Configuration for a synthetic ledger run.

    Attributes:
        n_transactions: Number of voucher rows to generate.
        n_vendors: Size of the vendor master file.
        n_employees: Size of the employee master file.
        anomaly_rate: Target share of anomalous vouchers (must sit in 2%-5%).
        dirty_rate: Share of rows that receive deliberate data-quality defects.
        start_date: First day of the ledger period.
        end_date: Last day of the ledger period.
        seed: Random seed for reproducibility.
    """

    n_transactions: int = 30_000
    n_vendors: int = 420
    n_employees: int = 140
    anomaly_rate: float = 0.030
    dirty_rate: float = 0.006
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    seed: int = RANDOM_SEED

    @property
    def period_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.start_date)

    @property
    def period_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_date)


# --------------------------------------------------------------------------- #
# Master data
# --------------------------------------------------------------------------- #
def _build_vendor_name(rng: np.random.Generator) -> str:
    """Compose a plausible company name from the name banks."""
    prefix = rng.choice(VENDOR_NAME_PREFIXES)
    core = rng.choice(VENDOR_NAME_CORES)
    suffix = rng.choice(VENDOR_NAME_SUFFIXES)
    if suffix.startswith("("):
        return f"{prefix} {core} {suffix}"
    return f"{prefix} {core} {suffix}"


def _bank_account(rng: np.random.Generator) -> str:
    """Build a plausible 20-character corporate bank account number.

    ``numpy`` cannot sample directly above the int64 ceiling, so the number is
    assembled from two smaller chunks.
    """
    return f"CN{rng.integers(10**8, 10**9)}{rng.integers(10**10, 10**11)}"


def generate_vendors(config: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the vendor master file.

    A deliberately small cohort of vendors is registered shortly before the end
    of the ledger period (``NEW_VENDOR_DAYS`` window) because "newly registered
    vendor receiving a large payment" is one of the highest-signal audit red
    flags. A handful of vendors also share bank accounts, which is a classic
    shell-company indicator and is exposed by one of the SQL queries.

    Returns:
        Dataframe with one row per vendor.
    """
    period_end = config.period_end
    records: list[dict[str, Any]] = []

    # ~12% of the vendor base is "new" - registered within the 90 days before
    # the period end - so that the unusual-vendor rule has something to find.
    n_new = max(1, int(config.n_vendors * 0.12))
    new_vendor_ids = set(range(config.n_vendors - n_new, config.n_vendors))

    # ~10% of the master file is dormant: vendors that exist on the books but
    # have not transacted at all during the period. Real vendor masters always
    # contain a long tail like this, and a sudden payment to one of them is a
    # recognised red flag.
    n_dormant = max(1, int(config.n_vendors * 0.10))
    dormant_candidates = [idx for idx in range(config.n_vendors) if idx not in new_vendor_ids]
    dormant_vendor_ids = set(rng.choice(dormant_candidates, size=n_dormant, replace=False).tolist())

    shared_bank_accounts = [_bank_account(rng) for _ in range(4)]

    for idx in range(config.n_vendors):
        vendor_id = f"V{idx + 1:05d}"

        if idx in new_vendor_ids:
            reg_start = period_end - pd.Timedelta(days=88)
            reg_end = period_end - pd.Timedelta(days=5)
        else:
            reg_start = pd.Timestamp("2013-01-01")
            reg_end = period_end - pd.Timedelta(days=180)

        span_days = max(1, (reg_end - reg_start).days)
        registration_date = reg_start + pd.Timedelta(days=int(rng.integers(0, span_days)))

        category = str(rng.choice(VENDOR_CATEGORIES))

        # Risk level is loosely correlated with category so that the vendor
        # risk page is not uniform noise.
        risk_probs = np.array([0.74, 0.19, 0.07])
        if category in ("Professional Services", "Marketing Agency"):
            risk_probs = np.array([0.58, 0.28, 0.14])
        risk_level = str(rng.choice(["Low", "Medium", "High"], p=risk_probs))

        if idx in new_vendor_ids and rng.random() < 0.35:
            risk_level = "High"

        # 1.5% of vendors share a bank account with another vendor.
        if rng.random() < 0.015:
            bank_account = str(rng.choice(shared_bank_accounts))
        else:
            bank_account = _bank_account(rng)

        records.append(
            {
                "vendor_id": vendor_id,
                "vendor_name": _build_vendor_name(rng),
                "vendor_category": category,
                "bank_account": bank_account,
                "registration_date": registration_date.normalize(),
                "country": str(rng.choice(COUNTRY_CHOICES)),
                "risk_level": risk_level,
                "is_new_vendor": idx in new_vendor_ids,
                "is_dormant": idx in dormant_vendor_ids,
            }
        )

    vendors = pd.DataFrame(records)
    LOGGER.info(
        "Generated %s vendors (%s newly registered, %s dormant with no period activity)",
        len(vendors),
        n_new,
        n_dormant,
    )
    return vendors


def _build_employee_name(rng: np.random.Generator) -> str:
    """Compose a plausible employee name."""
    return f"{rng.choice(SURNAMES)} {rng.choice(GIVEN_NAMES)}"


def generate_employees(config: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the employee / user master file.

    Roles carry an approval authority flag so the generator can enforce a
    believable segregation of duties: clerks raise vouchers, managers approve
    them. Violations of that split are injected later as ``self_approval``
    anomalies.
    """
    shares = np.array([role["share"] for role in ROLE_CATALOGUE], dtype=float)
    shares = shares / shares.sum()
    chosen = rng.choice(len(ROLE_CATALOGUE), size=config.n_employees, p=shares)

    records: list[dict[str, Any]] = []
    for idx, role_idx in enumerate(chosen):
        role = ROLE_CATALOGUE[int(role_idx)]
        records.append(
            {
                "employee_id": f"E{idx + 1:04d}",
                "employee_name": _build_employee_name(rng),
                "department": role["department"],
                "role": role["role"],
                "can_create": bool(role["creator"]),
                "can_approve": bool(role["approver"]),
            }
        )

    employees = pd.DataFrame(records)
    LOGGER.info("Generated %s employees across %s departments", len(employees), employees["department"].nunique())
    return employees


# --------------------------------------------------------------------------- #
# Ledger generation
# --------------------------------------------------------------------------- #
def _sample_dates(config: GeneratorConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample voucher dates with monthly seasonality, on business days.

    Returns:
        Array of :class:`pandas.Timestamp` values, one per voucher.
    """
    months = pd.date_range(config.period_start, config.period_end, freq="MS")
    weights = np.array(
        [MONTHLY_SEASONALITY[month.month - 1] * month.days_in_month for month in months],
        dtype=float,
    )
    weights = weights / weights.sum()

    month_idx = rng.choice(len(months), size=n, p=weights)
    dates = []
    for idx in month_idx:
        month = months[int(idx)]
        day = int(rng.integers(1, month.days_in_month + 1))
        dates.append(month + pd.Timedelta(days=day - 1))
    sampled = pd.Series(pd.to_datetime(dates))

    # Real ledgers post almost nothing at the weekend or on a public holiday: the
    # accounting calendar follows the working calendar. Dates that land on a
    # non-working day are pulled forward to the next working day, leaving a small
    # residual of genuinely off-hours postings for the rule to find.
    holiday_index = pd.DatetimeIndex(pd.to_datetime(list(PUBLIC_HOLIDAYS)))
    non_working = (sampled.dt.dayofweek >= 5) | sampled.dt.normalize().isin(holiday_index)
    residual = non_working & (rng.random(len(sampled)) < 0.06)

    shifted = sampled.copy()
    to_shift = (non_working & ~residual).to_numpy()
    for position in np.flatnonzero(to_shift):
        candidate = sampled.iloc[position] + pd.Timedelta(days=1)
        while candidate.dayofweek >= 5 or candidate.normalize() in holiday_index:
            candidate = candidate + pd.Timedelta(days=1)
        shifted.iloc[position] = candidate
    return shifted.to_numpy()


def _sample_amount(band: tuple[float, float], rng: np.random.Generator) -> float:
    """Draw a voucher amount from a log-normal distribution inside ``band``.

    The spread is deliberately wide (``sigma`` is a third of the log-range). A
    narrow distribution would concentrate the leading digits around the middle of
    the band and break Benford's Law, which would make the digit analysis module
    report a deviation on data that contains nothing wrong at all. Genuine
    ledgers span several orders of magnitude within a single account because the
    amount is a quantity multiplied by a unit price.
    """
    low, high = band
    mu = (np.log(low) + np.log(high)) / 2.0
    sigma = (np.log(high) - np.log(low)) / 3.0
    value = float(rng.lognormal(mu, sigma))
    value = float(np.clip(value, low * 0.35, high * 2.5))
    return round(value, 2)


def _naturally_round(amount: float, rng: np.random.Generator) -> float:
    """Round some amounts the way real ledgers look.

    Some genuine payments are round numbers - rent, retainers, monthly
    instalments - but most are not, because an invoice is a quantity multiplied
    by a unit price. Without *any* round amounts the "large round amount" rule
    would be trivially perfect and its precision meaningless; with too many, the
    rule drowns in legitimate round payments. The rates below produce roughly one
    round amount in eight.

    The rounding granularity is chosen from the magnitude of the amount so that
    a CNY 800 travel claim is never rounded down to zero.
    """
    draw = rng.random()
    if amount >= 100_000:
        if draw < 0.03:
            return float(round(amount / 10_000) * 10_000)
        if draw < 0.07:
            return float(round(amount / 1_000) * 1_000)
        if draw < 0.12:
            return float(round(amount / 100) * 100)
    elif amount >= 10_000:
        if draw < 0.04:
            return float(round(amount / 1_000) * 1_000)
        if draw < 0.09:
            return float(round(amount / 100) * 100)
        if draw < 0.13:
            return float(round(amount / 10) * 10)
    elif amount >= 1_000:
        if draw < 0.05:
            return float(round(amount / 100) * 100)
        if draw < 0.11:
            return float(round(amount / 10) * 10)
    elif draw < 0.08:
        return float(round(amount))
    return round(amount, 2)


def _build_description(
    process: str,
    vendor_name: str | None,
    employee_name: str | None,
    rng: np.random.Generator,
) -> str:
    """Render a business-process description from the template bank."""
    templates = DESCRIPTION_TEMPLATES.get(process, ("General journal entry",))
    template = str(rng.choice(templates))
    return (
        template.replace("{vendor}", vendor_name or "supplier")
        .replace("{employee}", employee_name or "staff")
        .replace("{city}", str(rng.choice(CITIES)))
        .replace("{item}", str(rng.choice(OFFICE_ITEMS)))
        .replace("{po}", f"{rng.integers(10000, 99999)}")
        .replace("{invoice}", f"INV-{rng.integers(100000, 999999)}")
        .replace("{department}", str(rng.choice(DEPARTMENTS)))
        .replace("{so}", f"{rng.integers(10000, 99999)}")
    )


def generate_ledger(
    config: GeneratorConfig,
    vendors: pd.DataFrame,
    employees: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the clean double-entry ledger (before anomalies are injected).

    Each row is one balanced journal voucher. ``account_code`` is the primary
    (debit) account the voucher is posted to and ``credit_account_code`` is the
    contra account, so every row satisfies ``debit_amount == credit_amount``.

    Returns:
        Dataframe of vouchers.
    """
    n = config.n_transactions
    process_names = list(BUSINESS_PROCESSES.keys())
    process_weights = np.array([BUSINESS_PROCESSES[p]["weight"] for p in process_names], dtype=float)
    process_weights = process_weights / process_weights.sum()
    processes = rng.choice(process_names, size=n, p=process_weights)

    dates = _sample_dates(config, n, rng)

    creators = employees.loc[employees["can_create"]].reset_index(drop=True)
    approvers = employees.loc[employees["can_approve"]].reset_index(drop=True)
    all_employees = employees.reset_index(drop=True)

    # Only non-dormant vendors take part in ordinary trading. A vendor cannot be
    # paid before it was registered, so the eligible pool is rebuilt for each
    # calendar month rather than sampled from the whole master file - otherwise
    # a vendor registered in late 2025 would show up in 2024 vouchers.
    active_vendors = vendors.loc[~vendors["is_dormant"]].sort_values("registration_date").reset_index(drop=True)
    active_ids = active_vendors["vendor_id"].to_numpy()
    active_names = active_vendors["vendor_name"].to_numpy()
    active_registration = active_vendors["registration_date"].to_numpy()

    month_keys = pd.Series(pd.to_datetime(dates)).dt.to_period("M").astype(str)
    eligible_pool_by_month: dict[str, np.ndarray] = {}
    for month_key in month_keys.unique():
        month_end = pd.Period(month_key, freq="M").end_time
        eligible = np.flatnonzero(active_registration <= np.datetime64(month_end))
        eligible_pool_by_month[month_key] = eligible if len(eligible) else np.arange(len(active_ids))

    month_key_array = month_keys.to_numpy()

    # Payment-related processes get a higher chance of sitting near the
    # approval threshold, which is what makes the split-payment rule realistic.
    records: list[dict[str, Any]] = []

    for i in range(n):
        process = str(processes[i])
        spec = BUSINESS_PROCESSES[process]

        amount = _naturally_round(_sample_amount(spec["amount"], rng), rng)

        transaction_date = pd.Timestamp(dates[i])
        # Posting lag: most vouchers post the same day or within three days.
        posting_lag = int(rng.choice([0, 0, 0, 1, 1, 2, 3], p=[0.55, 0.15, 0.08, 0.09, 0.05, 0.05, 0.03]))
        posting_date = transaction_date + pd.Timedelta(days=posting_lag)

        vendor_idx = None
        if process in VENDOR_PROCESSES:
            pool = eligible_pool_by_month[str(month_key_array[i])]
            vendor_idx = int(pool[rng.integers(0, len(pool))])

        employee_idx = None
        if process in EMPLOYEE_PROCESSES or process == "payroll":
            employee_idx = int(rng.integers(0, len(all_employees)))

        creator = creators.iloc[int(rng.integers(0, len(creators)))]
        approver = approvers.iloc[int(rng.integers(0, len(approvers)))]

        # Currency: the group reports in CNY, ~8% of documents are foreign.
        if rng.random() < 0.08:
            currency = str(rng.choice(["USD", "EUR", "HKD", "JPY", "SGD"], p=[0.45, 0.25, 0.15, 0.08, 0.07]))
        else:
            currency = "CNY"
        fx_rate = FX_RATES_TO_CNY[currency]
        amount_original = round(amount / fx_rate, 2)

        # Timestamps. The control timeline is built backwards from the posting
        # date so that it is always coherent:
        #     invoice received  ->  approved  ->  paid  ->  posted
        # A voucher can never be approved before the invoice arrives, and cash
        # can never leave the bank before the payment is approved.
        if process in INVOICE_PROCESSES:
            payment_time = transaction_date + pd.Timedelta(
                hours=int(rng.integers(9, 18)), minutes=int(rng.integers(0, 60))
            )
            # Approval lag: median around 20 hours, with a long right tail.
            approval_lag_hours = float(rng.lognormal(mean=3.0, sigma=0.9))
            approval_time = payment_time - pd.Timedelta(hours=approval_lag_hours)
            # Invoices typically sit between one and thirty days before payment.
            invoice_age_hours = float(rng.uniform(24, 720))
            invoice_time = approval_time - pd.Timedelta(hours=invoice_age_hours)
        else:
            invoice_time = pd.NaT
            approval_time = transaction_date + pd.Timedelta(
                hours=int(rng.integers(9, 18)), minutes=int(rng.integers(0, 60))
            )
            payment_time = pd.NaT

        invoice_id = (
            f"INV-{transaction_date.year}-{rng.integers(10000, 99999)}"
            if process in INVOICE_PROCESSES
            else None
        )

        department = PROCESS_DEPARTMENT[process]
        if employee_idx is not None and process in EMPLOYEE_PROCESSES:
            department = str(all_employees.iloc[employee_idx]["department"])

        debit_account = spec["debit"]
        credit_account = spec["credit"]

        records.append(
            {
                "transaction_id": f"TXN{i + 1:08d}",
                "journal_id": f"JV{transaction_date.year}{i + 1:06d}",
                "invoice_id": invoice_id,
                "transaction_date": transaction_date,
                "posting_date": posting_date,
                "account_code": debit_account,
                "account_name": ACCOUNTS[debit_account]["name"],
                "credit_account_code": credit_account,
                "credit_account_name": ACCOUNTS[credit_account]["name"],
                "debit_amount": amount,
                "credit_amount": amount,
                "amount": amount,
                "currency": currency,
                "amount_original": amount_original,
                "fx_rate": fx_rate,
                "vendor_id": active_ids[vendor_idx] if vendor_idx is not None else None,
                "vendor_name": active_names[vendor_idx] if vendor_idx is not None else None,
                "employee_id": all_employees.iloc[employee_idx]["employee_id"] if employee_idx is not None else None,
                "employee_name": all_employees.iloc[employee_idx]["employee_name"] if employee_idx is not None else None,
                "department": department,
                "payment_method": str(rng.choice(PAYMENT_METHODS, p=[0.68, 0.08, 0.16, 0.03, 0.05])),
                "description": _build_description(
                    process,
                    active_names[vendor_idx] if vendor_idx is not None else None,
                    all_employees.iloc[employee_idx]["employee_name"] if employee_idx is not None else None,
                    rng,
                ),
                "created_by": creator["employee_id"],
                "approved_by": approver["employee_id"],
                "approval_time": approval_time,
                "invoice_time": invoice_time,
                "payment_time": payment_time,
                "document_type": str(rng.choice(DOCUMENT_TYPES, p=[0.34, 0.30, 0.22, 0.09, 0.05])),
                "process": process,
                "anomaly_label": 0,
                "anomaly_type": None,
            }
        )

    ledger = pd.DataFrame(records)
    LOGGER.info("Generated %s clean vouchers (%s)", len(ledger), ledger["transaction_date"].min().date())
    return ledger


# --------------------------------------------------------------------------- #
# Data-quality defects
# --------------------------------------------------------------------------- #
def _stringify_dates(ledger: pd.DataFrame) -> pd.DataFrame:
    """Render date and timestamp columns as text, the way an ERP extract arrives.

    Storing them as text is both authentic (CSV extracts are text) and
    necessary: it lets the generator inject an impossible calendar date such as
    ``2025-02-30``, which the cleaning pipeline must then detect and repair.
    """
    ledger = ledger.copy()
    for column in ("transaction_date", "posting_date"):
        ledger[column] = pd.to_datetime(ledger[column]).dt.strftime("%Y-%m-%d")
    for column in ("approval_time", "invoice_time", "payment_time"):
        ledger[column] = pd.to_datetime(ledger[column]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return ledger


def inject_data_quality_issues(
    ledger: pd.DataFrame, config: GeneratorConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Introduce realistic defects so the cleaning pipeline has work to do.

    Defects cover: missing values, duplicated rows, malformed dates,
    unbalanced entries, invalid vendor references, inconsistent account codes
    and messy currency codes.

    Args:
        ledger: Voucher dataframe with dates already rendered as text.
        config: Generator configuration.
        rng: Seeded random generator.

    Returns:
        The ledger with defects applied.
    """
    ledger = ledger.copy()
    n = len(ledger)
    n_defects = max(1, int(n * config.dirty_rate))

    # 1. Missing values in key descriptive columns.
    missing_targets = rng.choice(n, size=n_defects, replace=False)
    for row_idx in missing_targets:
        column = str(rng.choice(["description", "department", "payment_method", "account_name", "approved_by"]))
        ledger.at[row_idx, column] = None

    # 2. Fully duplicated rows (double-posted batch runs).
    duplicate_targets = rng.choice(n, size=max(1, n_defects // 3), replace=False)
    ledger = pd.concat([ledger, ledger.iloc[duplicate_targets]], ignore_index=True)

    # 3. Malformed dates - impossible calendar dates stored as text.
    date_targets = rng.choice(len(ledger), size=max(1, n_defects // 4), replace=False)
    for row_idx in date_targets:
        ledger.at[row_idx, "posting_date"] = "2025-02-30"  # never a real date

    # 4. Unbalanced entries: the debit leg no longer equals the credit leg.
    unbalanced_targets = rng.choice(len(ledger), size=max(1, n_defects // 4), replace=False)
    for row_idx in unbalanced_targets:
        ledger.at[row_idx, "credit_amount"] = round(float(ledger.at[row_idx, "credit_amount"]) * 0.97, 2)

    # 5. Negative amounts on vouchers that should always be positive.
    negative_targets = rng.choice(len(ledger), size=max(1, n_defects // 5), replace=False)
    for row_idx in negative_targets:
        ledger.at[row_idx, "amount"] = -abs(float(ledger.at[row_idx, "amount"]))

    # 6. Invalid vendor references (orphan foreign keys).
    vendor_targets = rng.choice(len(ledger), size=max(1, n_defects // 5), replace=False)
    for row_idx in vendor_targets:
        ledger.at[row_idx, "vendor_id"] = "V99999"
        ledger.at[row_idx, "vendor_name"] = "UNKNOWN VENDOR"

    # 7. Inconsistent account codes: the code exists but the name does not match.
    account_targets = rng.choice(len(ledger), size=max(1, n_defects // 6), replace=False)
    for row_idx in account_targets:
        ledger.at[row_idx, "account_name"] = "Suspense account (unmapped)"

    # 8. Messy currency codes that need normalisation.
    currency_targets = rng.choice(len(ledger), size=max(1, n_defects // 4), replace=False)
    messy = ["cny", " RMB ", "usd", "Rmb", "CNY ", "hkd"]
    for row_idx in currency_targets:
        ledger.at[row_idx, "currency"] = str(rng.choice(messy))

    LOGGER.info("Injected data-quality defects; ledger now has %s rows", len(ledger))
    return ledger


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate_dataset(config: GeneratorConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run the full generation pipeline and write the raw CSV extracts.

    Args:
        config: Optional configuration override.

    Returns:
        Mapping of ``{"transactions": df, "vendors": df, "employees": df}``.
    """
    config = config or GeneratorConfig()
    ensure_directories()
    rng = np.random.default_rng(config.seed)

    vendors = generate_vendors(config, rng)
    employees = generate_employees(config, rng)
    ledger = generate_ledger(config, vendors, employees, rng)
    ledger = inject_anomalies(ledger, vendors, config, rng)
    # Dates are rendered as text (as they would arrive from an ERP extract)
    # before the data-quality defects are applied.
    ledger = _stringify_dates(ledger)
    ledger = inject_data_quality_issues(ledger, config, rng)

    # Shuffle so that injected anomalies are not clustered at the end of the file.
    ledger = ledger.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

    ledger.to_csv(TRANSACTIONS_RAW, index=False, encoding="utf-8-sig")
    vendors.drop(columns=["is_new_vendor", "is_dormant"]).to_csv(
        VENDORS_RAW, index=False, encoding="utf-8-sig"
    )
    employees.drop(columns=["can_create", "can_approve"]).to_csv(
        EMPLOYEES_RAW, index=False, encoding="utf-8-sig"
    )

    LOGGER.info("Wrote %s", TRANSACTIONS_RAW)
    LOGGER.info("Wrote %s", VENDORS_RAW)
    LOGGER.info("Wrote %s", EMPLOYEES_RAW)

    return {"transactions": ledger, "vendors": vendors, "employees": employees}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate the AuditLens synthetic ledger.")
    parser.add_argument("--transactions", type=int, default=30_000, help="number of vouchers to generate")
    parser.add_argument("--vendors", type=int, default=420, help="number of vendors")
    parser.add_argument("--employees", type=int, default=140, help="number of employees")
    parser.add_argument("--anomaly-rate", type=float, default=0.030, help="share of anomalous vouchers (0.02-0.05)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    config = GeneratorConfig(
        n_transactions=args.transactions,
        n_vendors=args.vendors,
        n_employees=args.employees,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    with Timer("dataset generation"):
        result = generate_dataset(config)

    ledger = result["transactions"]
    LOGGER.info("Ledger summary: %s rows | total value ¥%.2f | anomaly rate %.2f%%",
                len(ledger),
                float(ledger["debit_amount"].sum()),
                float(ledger["anomaly_label"].mean()) * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
