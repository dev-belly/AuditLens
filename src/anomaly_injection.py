"""Anomaly injection: the nine audit-relevant patterns and their ground truth.

The clean ledger built by :mod:`src.data_generator` contains no irregularities, so
this module plants them. Each pattern has its own injection function, and every
affected row is stamped with ``anomaly_label`` / ``anomaly_type``.

Those two columns are **ground truth**: they exist to grade the rule engine and the
model, and they are never shown on the auditor-facing dashboard. Keeping them here,
beside the code that creates them, makes that boundary easy to see.

Extracted from ``data_generator.py`` once that module passed a thousand lines. The
split is a pure move - the random draws happen in exactly the same order - so the
generated ledger is unchanged, which ``tests/test_reproducibility.py`` and a
byte-for-byte pipeline diff both verify.

The ``TYPE_CHECKING`` import below is deliberate: :class:`GeneratorConfig` is needed
only for annotations, and importing it at runtime would make this module and
:mod:`src.data_generator` import each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.utils import (
    ACCOUNTS,
    APPROVAL_THRESHOLD_CNY,
    FX_RATES_TO_CNY,
    RARE_ACCOUNTS,
    SUSPICIOUS_KEYWORDS,
    get_logger,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from src.data_generator import GeneratorConfig

LOGGER = get_logger(__name__)


#: How the anomaly budget is spread across the nine patterns.
ANOMALY_MIX: dict[str, float] = {
    "weekend_posting": 0.14,
    "self_approval": 0.14,
    "duplicate_payment": 0.12,
    "large_round_amount": 0.12,
    "suspicious_description": 0.12,
    "unusual_vendor": 0.10,
    "rapid_payment": 0.10,
    "split_transaction": 0.10,
    "rare_account_usage": 0.06,
}


def _mark(ledger: pd.DataFrame, indices: list[int] | np.ndarray, anomaly_type: str) -> None:
    """Flag the given row positions as anomalous, in place."""
    if len(indices) == 0:
        return
    ledger.loc[ledger.index[indices], "anomaly_label"] = 1
    ledger.loc[ledger.index[indices], "anomaly_type"] = anomaly_type


def _next_business_day(date: pd.Timestamp, rng: np.random.Generator, min_gap: int = 1) -> pd.Timestamp:
    """Return a business day at least ``min_gap`` days after ``date``.

    Keeping injected rows on business days avoids accidentally double-labelling
    a duplicate payment as a weekend posting, which would make the per-rule
    precision figures misleading.
    """
    candidate = date + pd.Timedelta(days=int(rng.integers(min_gap, min_gap + 8)))
    while candidate.dayofweek >= 5:  # Saturday / Sunday
        candidate = candidate + pd.Timedelta(days=1)
    return candidate


def _inject_duplicate_payments(
    ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, int]:
    """Clone payment vouchers so that the same vendor / invoice / amount is paid twice.

    Both the original and the duplicate are labelled, because the *pair* is the
    audit finding - the auditor reviews the pair, not a single row.

    Returns:
        Tuple of ``(updated ledger, number of flagged rows)``.
    """
    candidates = ledger.index[
        (ledger["process"] == "vendor_payment")
        & (ledger["invoice_id"].notna())
        & (ledger["anomaly_label"] == 0)
    ].to_numpy()
    if len(candidates) == 0:
        return ledger, 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    clones: list[dict[str, Any]] = []

    for position, row_idx in enumerate(chosen):
        original = ledger.loc[row_idx]
        clone_date = _next_business_day(pd.Timestamp(original["transaction_date"]), rng)
        clones.append(
            {
                **original.to_dict(),
                "transaction_id": f"TXN9{position:07d}",
                "journal_id": f"JV{clone_date.year}9{position:05d}",
                "transaction_date": clone_date,
                "posting_date": clone_date + pd.Timedelta(days=1),
                # The clone pays the *same invoice*, so the invoice and its
                # receipt timestamp are identical to the original - which is
                # exactly what the duplicate-payment rule keys on.
                "approval_time": clone_date - pd.Timedelta(hours=20),
                "payment_time": clone_date + pd.Timedelta(hours=13),
            }
        )

    ledger = pd.concat([ledger, pd.DataFrame(clones)], ignore_index=True)
    clone_indices = list(range(len(ledger) - len(clones), len(ledger)))
    _mark(ledger, list(chosen) + clone_indices, "duplicate_payment")
    return ledger, len(chosen) + len(clones)


def _inject_weekend_postings(
    ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator
) -> int:
    """Shift voucher dates onto a Saturday or Sunday."""
    candidates = ledger.index[ledger["anomaly_label"] == 0].to_numpy()
    if len(candidates) == 0:
        return 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    for row_idx in chosen:
        current = ledger.at[row_idx, "transaction_date"]
        offset = (5 - current.dayofweek) % 7  # move to Saturday
        if offset == 0:
            offset = 7
        weekend_date = current + pd.Timedelta(days=int(offset))
        ledger.at[row_idx, "transaction_date"] = weekend_date
        ledger.at[row_idx, "posting_date"] = weekend_date
        if pd.notna(ledger.at[row_idx, "approval_time"]):
            ledger.at[row_idx, "approval_time"] = weekend_date + pd.Timedelta(hours=10)

    _mark(ledger, chosen, "weekend_posting")
    return len(chosen)


def _inject_large_round_amounts(
    ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator
) -> int:
    """Set a handful of vouchers to very large, obviously round amounts."""
    high_percentile = float(ledger.loc[ledger["anomaly_label"] == 0, "amount"].quantile(0.95))
    round_values = np.array([100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000, 2_000_000], dtype=float)
    round_values = round_values[round_values > high_percentile]
    if len(round_values) == 0:
        round_values = np.array([round(high_percentile * 1.5, -4)], dtype=float)

    candidates = ledger.index[
        (ledger["anomaly_label"] == 0)
        & (ledger["process"].isin(["vendor_payment", "procurement", "consulting", "fixed_asset"]))
    ].to_numpy()
    if len(candidates) == 0:
        return 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    for row_idx in chosen:
        value = float(rng.choice(round_values))
        ledger.at[row_idx, "amount"] = value
        ledger.at[row_idx, "debit_amount"] = value
        ledger.at[row_idx, "credit_amount"] = value
        fx_rate = float(ledger.at[row_idx, "fx_rate"])
        ledger.at[row_idx, "amount_original"] = round(value / fx_rate, 2)

    _mark(ledger, chosen, "large_round_amount")
    return len(chosen)


def _inject_unusual_vendors(
    ledger: pd.DataFrame,
    vendors: pd.DataFrame,
    config: GeneratorConfig,
    n_targets: int,
    rng: np.random.Generator,
) -> int:
    """Route large payments through newly registered or long-dormant vendors.

    Two shapes are modelled: a vendor registered days ago that suddenly receives
    a payment many times the median voucher value, and a vendor that has been on
    the master file for years with no activity at all. When a newly registered
    vendor is used, the voucher date is pulled forward so the ledger stays
    internally consistent (a company cannot pay a vendor before it exists).
    """
    # A *sorted list*, not a set. The pool is sampled by index with
    # ``rng.choice``, so the collection's order is part of the random draw - and
    # ``list(set_of_strings)`` iterates in an order that depends on
    # ``PYTHONHASHSEED``, which differs between processes. Building it as a set
    # made the generator irreproducible: two runs of the same code disagreed on
    # the vendor count, the alert count and the model threshold, because the same
    # random index landed on a different vendor each time.
    #
    # ``tests/test_reproducibility.py`` runs the generator in two subprocesses
    # with different hash seeds and fails if this is ever changed back.
    new_vendor_ids = sorted(vendors.loc[vendors["is_new_vendor"], "vendor_id"].astype(str))
    vendor_lookup = vendors.set_index("vendor_id")

    # Dormant vendors are on the master file but transacted nothing during the
    # period, so any payment to one of them is inherently unusual.
    dormant_pool = list(vendors.loc[vendors["is_dormant"], "vendor_id"])
    if not dormant_pool:
        dormant_pool = list(new_vendor_ids)

    candidates = ledger.index[
        (ledger["anomaly_label"] == 0) & (ledger["vendor_id"].notna())
    ].to_numpy()
    if len(candidates) == 0:
        return 0

    median_amount = float(ledger.loc[ledger["anomaly_label"] == 0, "amount"].median())
    recent_start = config.period_end - pd.Timedelta(days=60)

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    for position, row_idx in enumerate(chosen):
        if position % 2 == 0 and new_vendor_ids:
            vendor_id = str(rng.choice(list(new_vendor_ids)))
            # New vendors can only be paid after they were registered.
            registration = pd.Timestamp(vendor_lookup.loc[vendor_id, "registration_date"])
            lower = max(registration + pd.Timedelta(days=2), recent_start)
            span = max(1, (config.period_end - lower).days)
            new_date = lower + pd.Timedelta(days=int(rng.integers(0, span)))
            while new_date.dayofweek >= 5:
                new_date = new_date + pd.Timedelta(days=1)
            # Shift the whole control timeline with the voucher date so that the
            # invoice -> approval -> payment ordering stays intact.
            shift = new_date - pd.Timestamp(ledger.at[row_idx, "transaction_date"])
            ledger.at[row_idx, "transaction_date"] = new_date
            ledger.at[row_idx, "posting_date"] = pd.Timestamp(ledger.at[row_idx, "posting_date"]) + shift
            for column in ("invoice_time", "approval_time", "payment_time"):
                if pd.notna(ledger.at[row_idx, column]):
                    ledger.at[row_idx, column] = pd.Timestamp(ledger.at[row_idx, column]) + shift
        else:
            vendor_id = str(rng.choice(dormant_pool))

        vendor_row = vendor_lookup.loc[vendor_id]
        ledger.at[row_idx, "vendor_id"] = vendor_id
        ledger.at[row_idx, "vendor_name"] = vendor_row["vendor_name"]

        # Large payment - between 3x and 9x the median voucher value.
        value = round(float(rng.uniform(3.0, 9.0)) * median_amount, -2)
        ledger.at[row_idx, "amount"] = value
        ledger.at[row_idx, "debit_amount"] = value
        ledger.at[row_idx, "credit_amount"] = value
        ledger.at[row_idx, "amount_original"] = round(value / float(ledger.at[row_idx, "fx_rate"]), 2)

    _mark(ledger, chosen, "unusual_vendor")
    return len(chosen)


def _inject_split_transactions(
    ledger: pd.DataFrame,
    vendors: pd.DataFrame,
    config: GeneratorConfig,
    n_targets: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """Create clusters of payments that sit just below the approval threshold.

    The pattern being modelled is a single obligation broken into several
    payments, each between 90% and 99.5% of the CNY 50,000 approval threshold,
    so that none of them triggers the second-level approval control.

    The clusters are self-contained (a fresh vendor / date group) rather than
    derived from an existing voucher, so that every row in the cluster is
    genuinely part of the pattern and the rule's precision is measured fairly.

    Returns:
        Tuple of ``(updated ledger, number of flagged rows)``.
    """
    if n_targets <= 0:
        return ledger, 0

    vendor_pool = vendors[["vendor_id", "vendor_name"]].reset_index(drop=True)
    templates = ledger.loc[
        ledger["process"].isin(["vendor_payment", "procurement", "consulting"])
    ].reset_index(drop=True)
    if templates.empty:
        return ledger, 0

    low = APPROVAL_THRESHOLD_CNY * 0.90
    high = APPROVAL_THRESHOLD_CNY * 0.995
    period_days = (config.period_end - config.period_start).days

    new_rows: list[dict[str, Any]] = []
    for position in range(n_targets):
        vendor = vendor_pool.iloc[int(rng.integers(0, len(vendor_pool)))]
        template = templates.iloc[int(rng.integers(0, len(templates)))]

        day_offset = int(rng.integers(0, period_days))
        base_date = config.period_start + pd.Timedelta(days=day_offset)
        while base_date.dayofweek >= 5:
            base_date = base_date + pd.Timedelta(days=1)

        n_splits = int(rng.integers(2, 4))  # 2 or 3 payments
        for split_no in range(n_splits):
            value = round(float(rng.uniform(low, high)), 2)
            currency = "CNY"
            new_rows.append(
                {
                    **template.to_dict(),
                    "transaction_id": f"TXN8{position:05d}{split_no:02d}",
                    "journal_id": f"JV{base_date.year}8{position:05d}{split_no}",
                    "invoice_id": f"INV-{base_date.year}-{rng.integers(10000, 99999)}",
                    "transaction_date": base_date,
                    "posting_date": base_date,
                    "vendor_id": vendor["vendor_id"],
                    "vendor_name": vendor["vendor_name"],
                    "amount": value,
                    "debit_amount": value,
                    "credit_amount": value,
                    "currency": currency,
                    "fx_rate": FX_RATES_TO_CNY[currency],
                    "amount_original": value,
                    "description": (
                        f"Partial payment to {vendor['vendor_name']} - "
                        f"instalment {split_no + 1}/{n_splits}"
                    ),
                    "created_by": template["created_by"],
                    "approved_by": template["approved_by"],
                    "approval_time": base_date + pd.Timedelta(hours=10),
                    "invoice_time": base_date - pd.Timedelta(days=2),
                    "payment_time": base_date + pd.Timedelta(hours=15),
                    "anomaly_label": 1,
                    "anomaly_type": "split_transaction",
                }
            )

    ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
    start = len(ledger) - len(new_rows)
    _mark(ledger, list(range(start, len(ledger))), "split_transaction")
    return ledger, len(new_rows)


def _inject_self_approval(ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator) -> int:
    """Make the voucher creator also the approver - a segregation-of-duties breach."""
    candidates = ledger.index[ledger["anomaly_label"] == 0].to_numpy()
    if len(candidates) == 0:
        return 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    for row_idx in chosen:
        ledger.at[row_idx, "approved_by"] = ledger.at[row_idx, "created_by"]

    _mark(ledger, chosen, "self_approval")
    return len(chosen)


def _inject_rapid_payments(ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator) -> int:
    """Compress the invoice-to-payment cycle into a few hours."""
    candidates = ledger.index[
        (ledger["anomaly_label"] == 0) & (ledger["invoice_time"].notna())
    ].to_numpy()
    if len(candidates) == 0:
        return 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    for row_idx in chosen:
        posting_date = pd.Timestamp(ledger.at[row_idx, "posting_date"])
        payment_time = posting_date + pd.Timedelta(hours=int(rng.integers(9, 18)))
        # Cash leaves the bank within a few hours of the invoice arriving.
        invoice_time = payment_time - pd.Timedelta(hours=float(rng.uniform(1.0, 5.5)))
        approval_time = invoice_time + pd.Timedelta(minutes=int(rng.integers(20, 90)))
        ledger.at[row_idx, "invoice_time"] = invoice_time
        ledger.at[row_idx, "approval_time"] = approval_time
        ledger.at[row_idx, "payment_time"] = payment_time

    _mark(ledger, chosen, "rapid_payment")
    return len(chosen)


def _inject_suspicious_descriptions(
    ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator
) -> int:
    """Attach vague, keyword-laden or missing descriptions."""
    candidates = ledger.index[ledger["anomaly_label"] == 0].to_numpy()
    if len(candidates) == 0:
        return 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    templates = (
        "Urgent manual adjustment - no invoice",
        "Special payment approved verbally",
        "Miscellaneous adjustment",
        "Temporary transfer - details to follow",
        "Adjustment for prior period difference",
        "Other - see email",
        "",
        "Misc",
    )

    for row_idx in chosen:
        template = str(rng.choice(templates))
        if template == "":
            ledger.at[row_idx, "description"] = None
        else:
            ledger.at[row_idx, "description"] = f"{template} {rng.choice(SUSPICIOUS_KEYWORDS)}".strip()

    _mark(ledger, chosen, "suspicious_description")
    return len(chosen)


def _inject_rare_account_usage(
    ledger: pd.DataFrame, n_targets: int, rng: np.random.Generator
) -> int:
    """Post transactions to accounts that are barely used by that process."""
    candidates = ledger.index[
        (ledger["anomaly_label"] == 0)
        & (ledger["process"].isin(["vendor_payment", "procurement", "consulting", "office_expense"]))
    ].to_numpy()
    if len(candidates) == 0:
        return 0

    chosen = rng.choice(candidates, size=min(n_targets, len(candidates)), replace=False)
    for row_idx in chosen:
        current_credit = ledger.at[row_idx, "credit_account_code"]
        options = [code for code in RARE_ACCOUNTS if code != current_credit]
        account_code = str(rng.choice(options))
        ledger.at[row_idx, "account_code"] = account_code
        ledger.at[row_idx, "account_name"] = ACCOUNTS[account_code]["name"]

    _mark(ledger, chosen, "rare_account_usage")
    return len(chosen)


def inject_anomalies(
    ledger: pd.DataFrame,
    vendors: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject all nine anomaly patterns into the clean ledger.

    The anomaly budget is expressed as a share of rows (``anomaly_rate``) and
    split across patterns using :data:`ANOMALY_MIX`. Patterns that create
    additional rows (duplicates, splits) are accounted for so the final rate
    stays inside the requested band.

    Args:
        ledger: Clean voucher dataframe.
        vendors: Vendor master data.
        config: Generator configuration.
        rng: Seeded random generator.

    Returns:
        Ledger with ``anomaly_label`` / ``anomaly_type`` populated.
    """
    target_rows = int(round(config.n_transactions * config.anomaly_rate))
    LOGGER.info("Injecting ~%s anomalous rows across %s patterns", target_rows, len(ANOMALY_MIX))

    budgets = {name: int(round(target_rows * share)) for name, share in ANOMALY_MIX.items()}

    # Order matters: patterns that depend on clean, "normal looking" rows run
    # first so that they do not stack on top of each other.
    # Duplicate payments create 2 flagged rows per target (original + clone)
    # and split transactions create 2-3, hence the divisors.
    ledger, n = _inject_duplicate_payments(ledger, budgets["duplicate_payment"] // 2, rng)
    LOGGER.info("  duplicate_payment      : %4d rows", n)

    n = _inject_weekend_postings(ledger, budgets["weekend_posting"], rng)
    LOGGER.info("  weekend_posting        : %4d rows", n)

    n = _inject_large_round_amounts(ledger, budgets["large_round_amount"], rng)
    LOGGER.info("  large_round_amount     : %4d rows", n)

    n = _inject_unusual_vendors(ledger, vendors, config, budgets["unusual_vendor"], rng)
    LOGGER.info("  unusual_vendor         : %4d rows", n)

    ledger, n = _inject_split_transactions(ledger, vendors, config, budgets["split_transaction"] // 2, rng)
    LOGGER.info("  split_transaction      : %4d rows", n)

    n = _inject_self_approval(ledger, budgets["self_approval"], rng)
    LOGGER.info("  self_approval          : %4d rows", n)

    n = _inject_rapid_payments(ledger, budgets["rapid_payment"], rng)
    LOGGER.info("  rapid_payment          : %4d rows", n)

    n = _inject_suspicious_descriptions(ledger, budgets["suspicious_description"], rng)
    LOGGER.info("  suspicious_description : %4d rows", n)

    n = _inject_rare_account_usage(ledger, budgets["rare_account_usage"], rng)
    LOGGER.info("  rare_account_usage     : %4d rows", n)

    rate = float(ledger["anomaly_label"].mean())
    LOGGER.info(
        "Injected %s anomalous rows -> final anomaly rate %.2f%% (%s rows)",
        int(ledger["anomaly_label"].sum()),
        rate * 100,
        len(ledger),
    )
    if not 0.02 <= rate <= 0.05:
        LOGGER.warning(
            "Anomaly rate %.4f is outside the intended 2%%-5%% band; adjust anomaly_rate.", rate
        )
    return ledger

