"""Shared pytest fixtures for the AuditLens test suite.

The tests are deliberately hermetic. Rather than running the whole pipeline and
asserting against whatever it produced, each test builds the smallest ledger that
can exercise the behaviour under test. That keeps the suite fast, keeps the
expected values readable, and means a change to the synthetic data generator cannot
silently invalidate the assertions about the analytics.

Two dates matter throughout:

* ``WEEKDAY`` - Tuesday 2025-03-11, an ordinary working day.
* ``SATURDAY`` - 2025-03-15, a weekend.
* ``HOLIDAY`` - 2025-05-01, a PRC public holiday that falls on a Thursday, so it
  isolates the holiday branch from the weekend branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_rules import REQUIRED_FEATURE_COLUMNS
from src.utils import RANDOM_SEED

#: A Tuesday. Ordinary working day.
WEEKDAY = pd.Timestamp("2025-03-11")
#: The Saturday of the same week.
SATURDAY = pd.Timestamp("2025-03-15")
#: Labour Day 2025 - a Thursday, so it is a holiday but not a weekend.
HOLIDAY = pd.Timestamp("2025-05-01")

#: Defaults for a single, entirely unremarkable voucher. Tests override the fields
#: they care about, which makes each test's intent obvious from its overrides.
BASE_TRANSACTION: dict[str, object] = {
    "transaction_id": "TX0001",
    "journal_id": "JV0001",
    "invoice_id": "INV0001",
    "transaction_date": WEEKDAY,
    "posting_date": WEEKDAY,
    "account_code": "6401",
    "account_name": "主营业务成本 Cost of sales",
    "credit_account_code": "2202",
    "credit_account_name": "应付账款 Accounts payable",
    "debit_amount": 12_345.67,
    "credit_amount": 12_345.67,
    "amount": 12_345.67,
    # Present in the raw ERP extract: the document amount in its own currency and
    # the rate used to translate it into the CNY functional currency.
    "amount_original": 12_345.67,
    "fx_rate": 1.0,
    "currency": "CNY",
    "vendor_id": "V0001",
    "vendor_name": "Test Vendor One",
    "employee_id": "E0001",
    "employee_name": "Employee One",
    "department": "Procurement",
    "payment_method": "Bank transfer",
    "description": "Purchase of raw materials per PO 2025-0311",
    "created_by": "E0001",
    "approved_by": "E0002",
    "approval_time": WEEKDAY + pd.Timedelta(hours=8),
    "invoice_time": WEEKDAY - pd.Timedelta(days=3),
    "payment_time": WEEKDAY + pd.Timedelta(hours=20),
    "document_type": "Vendor invoice",
    "process": "procure_to_pay",
    # Engineered features the rule engine expects to find. Supplying them keeps the
    # tests independent of feature_engineering's disk artefacts.
    "days_since_vendor_registration": 1_200.0,
    "days_since_last_transaction": 45.0,
    "vendor_amount_ratio": 1.0,
    "vendor_transaction_count": 40.0,
    "vendor_master_risk_score": 10.0,
    "account_transaction_frequency": 0.08,
    "payment_delay_hours": 72.0,
}


def make_transactions(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a transaction frame from a list of field overrides.

    Args:
        rows: One dictionary per voucher. Each is merged over
            :data:`BASE_TRANSACTION`, and ``transaction_id`` is auto-numbered when
            the caller does not supply one.

    Returns:
        A dataframe carrying every column the rule engine needs.
    """
    records: list[dict[str, object]] = []
    for index, overrides in enumerate(rows, start=1):
        record = dict(BASE_TRANSACTION)
        record["transaction_id"] = f"TX{index:04d}"
        record.update(overrides)
        records.append(record)

    frame = pd.DataFrame(records)
    for column in ("transaction_date", "posting_date", "approval_time", "invoice_time", "payment_time"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture
def weekday_ledger() -> pd.DataFrame:
    """A small, entirely clean ledger: 20 unremarkable vouchers."""
    return make_transactions(
        [
            {
                "transaction_id": f"TX{i:04d}",
                "invoice_id": f"INV{i:04d}",
                "debit_amount": 5_000.0 + i * 137.0,
                "amount": 5_000.0 + i * 137.0,
                "credit_amount": 5_000.0 + i * 137.0,
                "vendor_id": f"V{i % 4:04d}",
                "vendor_name": f"Test Vendor {i % 4}",
                "account_code": "6401",
            }
            for i in range(1, 21)
        ]
    )


@pytest.fixture
def required_feature_columns() -> tuple[str, ...]:
    """The engineered columns the rule engine needs to avoid a disk rebuild."""
    return REQUIRED_FEATURE_COLUMNS


@pytest.fixture(autouse=True)
def _seed_numpy() -> None:
    """Pin the global numpy seed so any incidental randomness is reproducible."""
    np.random.seed(RANDOM_SEED)
