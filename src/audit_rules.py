"""Rule-based audit anomaly detection for AuditLens.

This is the heart of the platform. Nine deterministic audit procedures are
applied to every voucher, each producing a normalised 0-1 risk score and a
human-readable reason. Rules are deliberately *explainable*: an auditor must be
able to read the code, understand the logic and challenge it. That is the whole
point of a rules layer sitting in front of a machine-learning model.

Design notes
------------
* Each rule is an independent method that returns a score series aligned to the
  input frame. Rules never mutate the input.
* Scores are combined with a **noisy-OR** (probabilistic sum) rather than a
  weighted average, so that *several independent red flags on the same voucher*
  compound. A voucher that is a duplicate payment, posted at the weekend and
  approved by its own creator should score higher than any single rule can
  produce alone.
* Per-rule weights (see :data:`RULE_WEIGHTS`) encode how strongly each pattern
  suggests *intentional* misstatement rather than ordinary process noise.

Usage::

    python src/audit_rules.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/audit_rules.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import (
    ACCOUNTS,
    ANOMALY_TYPE_LABELS,
    APPROVAL_THRESHOLD_CNY,
    BUSINESS_PROCESSES,
    MATERIALITY_THRESHOLD_CNY,
    NEW_VENDOR_DAYS,
    PUBLIC_HOLIDAYS,
    RARE_ACCOUNTS,
    RAPID_PAYMENT_HOURS,
    RULE_ALERTS_CSV,
    RULE_EVALUATION_CSV,
    SPLIT_THRESHOLD_HIGH_RATIO,
    SPLIT_THRESHOLD_LOW_RATIO,
    SUSPICIOUS_KEYWORDS,
    TRANSACTIONS_CLEAN,
    TRANSACTIONS_FEATURES,
    VENDORS_CLEAN,
    Timer,
    as_flag_series,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
)

LOGGER = get_logger(__name__)

#: Processes that represent money leaving the organisation. Duplicate and split
#: payment detection only makes sense here.
PAYMENT_PROCESSES: tuple[str, ...] = ("vendor_payment", "procurement", "consulting", "cost_of_sales")

#: Debit account each process is normally posted to. Used by the rare-account rule.
PROCESS_DEFAULT_ACCOUNT: dict[str, str] = {
    process: spec["debit"] for process, spec in BUSINESS_PROCESSES.items()
}

#: How strongly each pattern suggests *intentional* misstatement rather than
#: ordinary process noise. Used in the noisy-OR combination. A duplicate payment
#: or a split designed to dodge an approval threshold is close to conclusive; a
#: weekend posting is often just an operational reality.
RULE_WEIGHTS: dict[str, float] = {
    "duplicate_payment": 0.95,
    "split_transaction": 0.85,
    "self_approval": 0.80,
    "unusual_vendor": 0.75,
    "large_round_amount": 0.60,
    "rapid_payment": 0.60,
    "suspicious_description": 0.55,
    "rare_account_usage": 0.50,
    "weekend_posting": 0.45,
}

#: Illustrative PRC public holidays, shared with the generator so that both the
#: synthetic ledger and the detection rule use the same calendar.
HOLIDAY_INDEX = pd.DatetimeIndex(pd.to_datetime(list(PUBLIC_HOLIDAYS)))

#: Engineered features the rules depend on. If any of these is absent the engine
#: rebuilds the whole feature table rather than silently defaulting to zero.
REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "days_since_vendor_registration",
    "days_since_last_transaction",
    "vendor_amount_ratio",
    "vendor_transaction_count",
    "vendor_master_risk_score",
    "account_transaction_frequency",
    "payment_delay_hours",
)

#: A description reused across more than this many *distinct vendors* is treated
#: as boilerplate narration copy-pasted between unrelated suppliers.
BOILERPLATE_VENDOR_THRESHOLD: int = 8

#: Repeat payments of the same amount to the same vendor within this many days
#: are treated as potential duplicates even when the invoice number differs.
DUPLICATE_WINDOW_DAYS: int = 5


@dataclass(frozen=True)
class RuleDefinition:
    """Static metadata describing an audit rule."""

    key: str
    label: str
    description: str
    audit_rationale: str
    weight: float

    @property
    def column(self) -> str:
        """Name of the score column this rule contributes."""
        return f"{self.key}_score"


RULE_DEFINITIONS: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        key="duplicate_payment",
        label="Duplicate Payment",
        description="The same vendor, amount and invoice paid more than once.",
        audit_rationale=(
            "Duplicate payments are the single most common recoverable error in "
            "accounts payable. The organisation pays twice for one obligation and "
            "the money is only recovered if someone looks."
        ),
        weight=RULE_WEIGHTS["duplicate_payment"],
    ),
    RuleDefinition(
        key="split_transaction",
        label="Split Transaction",
        description=(
            f"Two or more payments to the same vendor on the same day, each between "
            f"{SPLIT_THRESHOLD_LOW_RATIO:.0%} and {SPLIT_THRESHOLD_HIGH_RATIO:.0%} of the "
            f"CNY {APPROVAL_THRESHOLD_CNY:,.0f} approval threshold."
        ),
        audit_rationale=(
            "Splitting an obligation into amounts just below an approval threshold is "
            "a deliberate control override. It is one of the clearest indicators of "
            "intent because the amounts cluster immediately below the limit."
        ),
        weight=RULE_WEIGHTS["split_transaction"],
    ),
    RuleDefinition(
        key="self_approval",
        label="Self Approval",
        description="The employee who raised the voucher is also the approver.",
        audit_rationale=(
            "A breach of segregation of duties. It removes the four-eyes control that "
            "exists precisely to stop one person from moving money on their own."
        ),
        weight=RULE_WEIGHTS["self_approval"],
    ),
    RuleDefinition(
        key="unusual_vendor",
        label="Unusual Vendor",
        description=(
            "A newly registered or long-dormant vendor receiving a payment far above "
            "its own historical norm."
        ),
        audit_rationale=(
            "Newly created vendors and reactivated dormant vendors are the standard "
            "vehicle for fictitious supplier schemes, because nobody knows what normal "
            "looks like for them yet."
        ),
        weight=RULE_WEIGHTS["unusual_vendor"],
    ),
    RuleDefinition(
        key="large_round_amount",
        label="Large Round Amount",
        description="A large amount that is also an obviously round number.",
        audit_rationale=(
            "Genuine supplier invoices reflect actual quantities and unit prices, so "
            "they rarely land on exact round figures. Large round amounts usually mean "
            "a manually keyed journal."
        ),
        weight=RULE_WEIGHTS["large_round_amount"],
    ),
    RuleDefinition(
        key="rapid_payment",
        label="Rapid Payment",
        description=f"Payment made within {RAPID_PAYMENT_HOURS:.0f} hours of the invoice arriving.",
        audit_rationale=(
            "Normal supplier terms run to 30 or 60 days. Paying within hours means the "
            "invoice bypassed the usual matching and review cycle."
        ),
        weight=RULE_WEIGHTS["rapid_payment"],
    ),
    RuleDefinition(
        key="suspicious_description",
        label="Suspicious Description",
        description="Vague, missing or boilerplate narration.",
        audit_rationale=(
            "Narration is the auditor's primary evidence of business purpose. "
            "Descriptions such as 'urgent manual adjustment' or a blank field mean the "
            "purpose cannot be evidenced from the ledger alone."
        ),
        weight=RULE_WEIGHTS["suspicious_description"],
    ),
    RuleDefinition(
        key="rare_account_usage",
        label="Rare Account Usage",
        description="Posted to an account that is seldom used, or atypical for the process.",
        audit_rationale=(
            "Parking a payment in a rarely used account - typically 'other receivables' "
            "- obscures its nature and delays recognition of the expense."
        ),
        weight=RULE_WEIGHTS["rare_account_usage"],
    ),
    RuleDefinition(
        key="weekend_posting",
        label="Weekend / Holiday Posting",
        description="Voucher posted on a Saturday, Sunday or public holiday.",
        audit_rationale=(
            "Journal activity outside working hours is worth a look because it escapes "
            "the informal review that happens during the working week. On its own it is "
            "weak evidence, which is why it carries the lowest weight."
        ),
        weight=RULE_WEIGHTS["weekend_posting"],
    ),
)


@dataclass
class RuleResult:
    """Outcome of a single rule run."""

    definition: RuleDefinition
    scores: pd.Series
    reasons: pd.Series

    @property
    def flagged(self) -> pd.Series:
        """Boolean series marking flagged vouchers."""
        return self.scores > 0

    @property
    def n_flagged(self) -> int:
        """Number of flagged vouchers."""
        return int(self.flagged.sum())


@dataclass
class RuleEvaluation:
    """Precision / recall of one rule against the labelled benchmark."""

    rule_key: str
    rule_label: str
    flagged: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "rule_key": self.rule_key,
            "rule_label": self.rule_label,
            "flagged": self.flagged,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class AuditRuleEngine:
    """Applies the AuditLens rule set to a feature-enriched transaction table.

    Args:
        transactions: Clean transaction table (ideally feature-enriched).
        approval_threshold: Payment approval threshold in CNY.
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        approval_threshold: float = APPROVAL_THRESHOLD_CNY,
    ) -> None:
        if transactions.empty:
            raise ValueError("Cannot run the rule engine on an empty transaction table.")

        self.df = transactions.reset_index(drop=True).copy()
        self.approval_threshold = approval_threshold
        self._results: dict[str, RuleResult] = {}

        if "debit_amount" not in self.df.columns:
            raise KeyError("The transaction table must contain a 'debit_amount' column.")

        # Several rules depend on engineered features. Silently defaulting them to
        # zero would produce nonsense scores (for example every voucher flagged by
        # the rare-account rule), so missing features are computed on the fly.
        missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in self.df.columns]
        if missing:
            LOGGER.warning(
                "Missing %s engineered features (%s) - rebuilding them from the ledger.",
                len(missing),
                ", ".join(missing),
            )
            self.df = self._rebuild_features(self.df)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _rebuild_features(df: pd.DataFrame) -> pd.DataFrame:
        """Recompute the engineered features the rules depend on.

        Imported lazily to avoid a circular import at module load time
        (``feature_engineering`` does not import this module, but the rules
        module is imported by the pipeline before the features exist on disk).
        """
        from src.feature_engineering import build_features

        vendors = load_dataframe(VENDORS_CLEAN) if VENDORS_CLEAN.exists() else pd.DataFrame()
        enriched, _ = build_features(df, vendors)
        return enriched.reset_index(drop=True)

    @property
    def _amount(self) -> pd.Series:
        return self.df["debit_amount"]

    def _percentile(self, series: pd.Series) -> pd.Series:
        """Percentile rank in 0-1, used to scale several rules."""
        return series.rank(pct=True, method="average")

    @staticmethod
    def _reason(triggered: pd.Series, template: str, **kwargs: Any) -> pd.Series:
        """Build a reason string for every triggered row, blank elsewhere."""
        reasons = pd.Series("", index=triggered.index, dtype="string")
        if triggered.any():
            reasons.loc[triggered] = [template.format(**kwargs) for _ in range(int(triggered.sum()))]
        return reasons

    # ----------------------------------------------------------------- rules
    def rule_duplicate_payment(self) -> RuleResult:
        """Rule 1 - detect the same payment being made more than once.

        Two independent tests are applied:

        * **Same invoice** - identical vendor, invoice number and amount. This is
          the strongest form: one invoice, two payments.
        * **Repeat amount** - identical vendor and amount within
          :data:`DUPLICATE_WINDOW_DAYS` days, even if the invoice number differs.

        The returned score is the ``duplicate_risk_score`` referenced in the
        project specification.
        """
        df = self.df
        score = pd.Series(0.0, index=df.index)
        reason = pd.Series("", index=df.index, dtype="string")

        has_invoice = df["invoice_id"].notna() & (df["invoice_id"].astype("string") != "")
        invoice_key = (
            df["vendor_id"].astype("string")
            + "|"
            + df["invoice_id"].astype("string")
            + "|"
            + self._amount.round(2).astype("string")
        )
        invoice_counts = invoice_key.map(invoice_key.value_counts())
        same_invoice = has_invoice & (invoice_counts > 1)

        # Repeat amount test: compare each voucher with the previous and next
        # voucher that shares its vendor and amount.
        ordered = df.sort_values(["vendor_id", "debit_amount", "transaction_date"], na_position="last")
        group_keys = [ordered["vendor_id"], ordered["debit_amount"].round(2)]
        grouped = ordered.groupby(group_keys, dropna=True, sort=False, observed=True)
        previous_date = grouped["transaction_date"].shift(1)
        next_date = grouped["transaction_date"].shift(-1)
        gap_previous = (ordered["transaction_date"] - previous_date).dt.days
        gap_next = (next_date - ordered["transaction_date"]).dt.days
        repeat_amount = ((gap_previous <= DUPLICATE_WINDOW_DAYS) | (gap_next <= DUPLICATE_WINDOW_DAYS))
        repeat_amount = repeat_amount.reindex(df.index).fillna(False)
        repeat_amount = repeat_amount & df["vendor_id"].notna()

        score = score.mask(same_invoice, 1.0)
        score = score.mask(~same_invoice & repeat_amount, 0.65)

        vendor_label = df["vendor_name"].fillna(df["vendor_id"]).astype("string")
        reason = reason.mask(
            same_invoice,
            "Duplicate payment: invoice " + df["invoice_id"].astype("string") + " paid more than once to " + vendor_label,
        )
        reason = reason.mask(
            ~same_invoice & repeat_amount,
            "Repeat payment of CNY " + self._amount.round(2).astype("string")
            + f" to {vendor_label} within {DUPLICATE_WINDOW_DAYS} days",
        )
        return RuleResult(self._definition("duplicate_payment"), score, reason)

    def rule_split_transaction(self) -> RuleResult:
        """Rule 5 - detect payments split to stay below the approval threshold."""
        df = self.df
        low = self.approval_threshold * SPLIT_THRESHOLD_LOW_RATIO
        high = self.approval_threshold * SPLIT_THRESHOLD_HIGH_RATIO

        eligible = self._amount.between(low, high) & df["vendor_id"].notna()
        group_keys = [df["vendor_id"], df["transaction_date"]]
        # Vouchers with no vendor cannot belong to a vendor-day cluster, so the
        # group size is undefined for them; fall back to a single voucher.
        same_day_count = (
            df.groupby(group_keys, dropna=True, observed=True)["transaction_id"]
            .transform("count")
            .fillna(1.0)
            .astype(int)
        )
        same_day_value = (
            df.groupby(group_keys, dropna=True, observed=True)["debit_amount"]
            .transform("sum")
            .fillna(self._amount)
        )

        triggered = eligible & (same_day_count >= 2)

        # Escalate when the combined value of the cluster crosses the threshold -
        # that is the tell-tale signature of an obligation deliberately split.
        combined_over_threshold = triggered & (same_day_value > self.approval_threshold)

        score = pd.Series(0.0, index=df.index)
        score = score.mask(triggered, 0.65)
        score = score.mask(combined_over_threshold, 0.90)
        # Three or more near-threshold payments in one day is stronger still.
        score = score.mask(combined_over_threshold & (same_day_count >= 3), 1.0)

        reason = pd.Series("", index=df.index, dtype="string")
        reason = reason.mask(
            triggered,
            "Split payment: "
            + same_day_count.astype("int64").astype("string")
            + " payments to the same vendor on the same day, each just below the CNY "
            + f"{self.approval_threshold:,.0f}"
            + " approval threshold (combined CNY "
            + same_day_value.round(2).astype("string")
            + ")",
        )
        return RuleResult(self._definition("split_transaction"), score, reason)

    def rule_self_approval(self) -> RuleResult:
        """Rule 6 - detect segregation-of-duties breaches."""
        df = self.df
        triggered = (df["created_by"] == df["approved_by"]) & df["approved_by"].notna()
        above_p95 = triggered & (self._percentile(self._amount) > 0.95)

        score = pd.Series(0.0, index=df.index)
        score = score.mask(triggered, 0.85)
        score = score.mask(above_p95, 1.0)

        reason = pd.Series("", index=df.index, dtype="string")
        reason = reason.mask(
            triggered,
            "Self approval: voucher raised and approved by the same employee ("
            + df["created_by"].astype("string")
            + ") - segregation of duties breach",
        )
        return RuleResult(self._definition("self_approval"), score, reason)

    def rule_unusual_vendor(self) -> RuleResult:
        """Rule 4 - detect payments that are abnormal *for that vendor*.

        Four vendor-level signals are combined with a noisy-OR. Every signal is
        gated on the payment being *material* (above the 75th or 90th percentile
        of the ledger), because a small payment to a new vendor is unremarkable.

        * a newly registered vendor receiving a material payment
        * a dormant vendor suddenly receiving a material payment
        * a vendor with almost no history receiving a very large payment
        * a payment far above an established vendor's own median

        Two deliberate exclusions:

        * The vendor's master-data risk rating is **not** a trigger. It is a
          static attribute, so using it here would flag the same handful of
          vendors on every single payment - the classic way an audit rule loses
          credibility. It feeds the vendor risk component of the score instead.
        * A missing ``days_since_last_transaction`` (the vendor's first voucher)
          does not count as dormancy.
        """
        df = self.df
        percentile = self._percentile(self._amount)

        days_since_registration = df.get(
            "days_since_vendor_registration", pd.Series(np.nan, index=df.index)
        )
        days_since_last = df.get(
            "days_since_last_transaction", pd.Series(np.nan, index=df.index)
        )
        amount_ratio = df.get("vendor_amount_ratio", pd.Series(np.nan, index=df.index))
        txn_count = df.get("vendor_transaction_count", pd.Series(0.0, index=df.index)).fillna(0.0)

        has_vendor = df["vendor_id"].notna()
        material = percentile > 0.75
        large = percentile > 0.90

        signal_new = (
            has_vendor
            & days_since_registration.notna()
            & (days_since_registration <= NEW_VENDOR_DAYS)
            & material
        )
        signal_dormant = has_vendor & (
            (txn_count <= 2) | (days_since_last.notna() & (days_since_last > 180))
        ) & material
        signal_low_frequency = has_vendor & (txn_count <= 5) & large & (amount_ratio.fillna(0) > 4.0)
        signal_spike = has_vendor & (txn_count >= 5) & large & (amount_ratio.fillna(0) > 8.0)

        # Noisy-OR over the independent signals.
        score = pd.Series(0.0, index=df.index)
        for signal, weight in (
            (signal_new, 0.60),
            (signal_dormant, 0.55),
            (signal_spike, 0.55),
            (signal_low_frequency, 0.45),
        ):
            contribution = signal.fillna(False).astype(float) * weight
            score = 1.0 - (1.0 - score) * (1.0 - contribution)

        # Escalate when the payment is in the extreme tail of the ledger.
        extreme = (score > 0) & (percentile > 0.98)
        score = score.mask(extreme, np.minimum(1.0, score + 0.15))
        score = score.round(4).clip(upper=1.0)

        reasons = pd.Series("", index=df.index, dtype="string")

        def _add_reason(mask: pd.Series, text: str) -> None:
            """Append a reason fragment to every row in ``mask``."""
            target = mask.fillna(False)
            if target.any():
                existing = reasons.loc[target]
                reasons.loc[target] = existing.where(existing == "", existing + "; ") + text

        _add_reason(signal_new, "vendor registered less than 90 days before the payment")
        _add_reason(signal_dormant, "vendor had no activity for over 180 days before this payment")
        _add_reason(signal_low_frequency, "vendor has almost no transaction history")
        _add_reason(signal_spike, "amount is more than 8x this vendor's average payment")

        vendor_label = df["vendor_name"].fillna(df["vendor_id"]).astype("string")
        has_reason = reasons != ""
        reasons = reasons.mask(
            has_reason, "Unusual vendor (" + vendor_label + "): " + reasons
        )
        return RuleResult(self._definition("unusual_vendor"), score, reasons)

    def rule_large_round_amount(self) -> RuleResult:
        """Rule 3 - detect large, obviously round amounts.

        A simple ``amount % 1000 == 0`` test is useless on its own because plenty
        of genuine payments are round. Three conditions must hold together:

        1. the amount is above the ledger's 95th percentile, **and**
        2. the amount clears planning materiality (CNY 100,000), **and**
        3. the amount is an exact multiple of 1,000.

        The materiality floor is what makes the rule usable. Without it the test
        flags hundreds of immaterial round payments - a CNY 12,000 stationery
        order is round and nobody cares. The score then scales with how far into
        the tail the amount sits.
        """
        df = self.df
        amount = self._amount
        percentile = self._percentile(amount)

        p95 = float(amount.quantile(0.95))
        materiality = max(MATERIALITY_THRESHOLD_CNY, p95)
        is_round = (amount % 1_000 == 0) & (amount >= 10_000)
        is_very_round = (amount % 10_000 == 0) & (amount >= 10_000)

        triggered = (amount >= materiality) & is_round & (percentile > 0.95)
        # Scale 0.35 -> 0.85 by how far above P95 the amount sits.
        tail_position = ((percentile - 0.95) / 0.05).clip(lower=0.0, upper=1.0)
        score = pd.Series(0.0, index=df.index)
        score = score.mask(triggered, (0.35 + 0.50 * tail_position).round(4))
        score = score.mask(triggered & is_very_round, (score + 0.15).clip(upper=1.0).round(4))
        score = score.clip(upper=1.0)

        reason = pd.Series("", index=df.index, dtype="string")
        top_share = ((1.0 - percentile) * 100).round(2).astype("string")
        multiple = pd.Series(
            np.where(is_very_round, "10,000", "1,000"), index=df.index, dtype="string"
        )
        reason = reason.mask(
            triggered,
            "Large round amount: CNY "
            + amount.round(2).astype("string")
            + " sits in the top "
            + top_share
            + "% of all voucher values and is an exact multiple of "
            + multiple,
        )
        return RuleResult(self._definition("large_round_amount"), score, reason)

    def rule_rapid_payment(self) -> RuleResult:
        """Rule 7 - detect payments made abnormally soon after the invoice arrived."""
        df = self.df
        delay = df.get("payment_delay_hours", pd.Series(np.nan, index=df.index))
        triggered = delay.notna() & (delay < RAPID_PAYMENT_HOURS)

        # 0 hours -> 1.0, just under the threshold -> 0.55.
        urgency = (1.0 - (delay.clip(lower=0.0) / RAPID_PAYMENT_HOURS)).clip(lower=0.0, upper=1.0)
        score = pd.Series(0.0, index=df.index)
        score = score.mask(triggered, (0.55 + 0.45 * urgency).round(4))

        reason = pd.Series("", index=df.index, dtype="string")
        reason = reason.mask(
            triggered,
            "Rapid payment: settled "
            + delay.round(1).astype("string")
            + f" hours after the invoice arrived (normal supplier terms are 30-60 days)",
        )
        return RuleResult(self._definition("rapid_payment"), score, reason)

    def rule_suspicious_description(self) -> RuleResult:
        """Rule 8 - detect vague, missing or boilerplate narration.

        Keyword matching is intentionally simple. A transparent keyword list that
        an auditor can read and amend is far more useful in practice than an
        opaque language model, and it keeps every alert explainable.

        Three signals feed the score: a high-risk keyword, a missing narration, and
        a narration shorter than ten characters.

        **Repetition is measured but not scored.** Identical narration across many
        vendors *sounds* like a red flag, but on this ledger system-generated
        journals (for example the cost-of-sales transfer) legitimately repeat the
        same text thousands of times. Scoring repetition flagged 17% of the ledger
        at 2% precision, which would have buried the real findings. The repetition
        count is therefore exposed as ``description_repetition_flag`` for the
        auditor to review, and reported on the Audit Rules page, without being
        allowed to drive the risk score. See ``docs/methodology.md``.
        """
        df = self.df
        description = df["description"].astype("string")
        lowered = description.fillna("").str.lower()

        keyword_pattern = "|".join(SUSPICIOUS_KEYWORDS)
        matched = lowered.str.contains(keyword_pattern, regex=True, na=False)
        empty = description.isna() | (lowered.str.strip() == "")
        too_short = ~empty & (lowered.str.len() < 10)

        score = pd.Series(0.0, index=df.index)
        for signal, weight in (
            (matched, 0.70),
            (empty, 0.60),
            (too_short, 0.50),
        ):
            contribution = signal.fillna(False).astype(float) * weight
            score = 1.0 - (1.0 - score) * (1.0 - contribution)
        score = score.round(4)

        reasons = pd.Series("", index=df.index, dtype="string")

        def _add(mask: pd.Series, text: str) -> None:
            target = mask.fillna(False)
            if target.any():
                existing = reasons.loc[target]
                reasons.loc[target] = existing.where(existing == "", existing + "; ") + text

        _add(matched, "narration contains a high-risk keyword")
        _add(empty, "narration is missing entirely")
        _add(too_short, "narration is shorter than 10 characters")

        has_reason = reasons != ""
        reasons = reasons.mask(has_reason, "Suspicious description: " + reasons)
        return RuleResult(self._definition("suspicious_description"), score, reasons)

    def description_repetition_flag(self) -> pd.Series:
        """Flag narration that is boilerplate reused across unrelated vendors.

        Reported as an informational metric rather than a risk trigger; see
        :meth:`rule_suspicious_description` for the reasoning.
        """
        df = self.df
        description = df["description"].astype("string")
        vendor_spread = df.groupby(description, dropna=True, observed=True)["vendor_id"].nunique()
        repetition = description.map(vendor_spread).fillna(0)
        return (repetition > BOILERPLATE_VENDOR_THRESHOLD).fillna(False)

    def rule_rare_account_usage(self) -> RuleResult:
        """Rule 9 - detect postings to seldom-used or atypical accounts."""
        df = self.df
        account = df["account_code"].astype("string")
        frequency = df.get(
            "account_transaction_frequency", pd.Series(0.0, index=df.index)
        ).fillna(0.0)

        rare_account = account.isin(RARE_ACCOUNTS)
        low_frequency = frequency < 0.01

        expected_account = df.get("process", pd.Series(pd.NA, index=df.index)).map(PROCESS_DEFAULT_ACCOUNT)
        atypical = expected_account.notna() & (account != expected_account.astype("string"))

        score = pd.Series(0.0, index=df.index)
        score = score.mask(low_frequency, 0.45)
        score = score.mask(rare_account & low_frequency, 0.70)
        score = score.mask(rare_account & low_frequency & atypical, 0.80)

        reason = pd.Series("", index=df.index, dtype="string")
        triggered = score > 0
        frequency_pct = (frequency * 100).round(2).astype("string")
        atypical_note = pd.Series(
            np.where(atypical.fillna(False), " and atypical for this business process", ""),
            index=df.index,
            dtype="string",
        )
        reason = reason.mask(
            triggered,
            "Rare account usage: posted to "
            + account
            + " ("
            + df["account_name"].astype("string")
            + "), used by only "
            + frequency_pct
            + "% of vouchers"
            + atypical_note,
        )
        return RuleResult(self._definition("rare_account_usage"), score, reason)

    def rule_weekend_posting(self) -> RuleResult:
        """Rule 2 - detect postings outside the working week."""
        df = self.df
        transaction_date = df["transaction_date"]
        weekend = transaction_date.dt.dayofweek >= 5
        holiday = transaction_date.dt.normalize().isin(HOLIDAY_INDEX)
        triggered = weekend | holiday

        # Scale by amount percentile: an off-hours posting of a material amount
        # deserves more attention than a CNY 900 expense claim.
        percentile = self._percentile(self._amount)
        base = np.where(holiday, 0.65, 0.55)
        score = pd.Series(0.0, index=df.index)
        score = score.mask(triggered, np.round(base + 0.35 * percentile, 4))
        score = score.clip(upper=1.0)

        reason = pd.Series("", index=df.index, dtype="string")
        holiday_note = pd.Series(
            np.where(holiday, " (public holiday)", ""), index=df.index, dtype="string"
        )
        reason = reason.mask(
            triggered,
            "Posted on "
            + transaction_date.dt.day_name().astype("string")
            + holiday_note
            + " - "
            + transaction_date.dt.strftime("%Y-%m-%d").astype("string")
            + " is outside the normal working week",
        )
        return RuleResult(self._definition("weekend_posting"), score, reason)

    # ------------------------------------------------------------ orchestration
    @staticmethod
    def _definition(key: str) -> RuleDefinition:
        """Look up a rule definition by key."""
        for definition in RULE_DEFINITIONS:
            if definition.key == key:
                return definition
        raise KeyError(f"Unknown rule: {key}")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute every rule and combine the results.

        Returns:
            Tuple of ``(scored transactions, long-format alerts)``. The scored
            frame carries one ``<rule>_score`` and one ``<rule>_flag`` column per
            rule, plus ``rule_risk_score``, ``rule_alert_count`` and
            ``rule_reasons``.
        """
        for method in (
            self.rule_duplicate_payment,
            self.rule_split_transaction,
            self.rule_self_approval,
            self.rule_unusual_vendor,
            self.rule_large_round_amount,
            self.rule_rapid_payment,
            self.rule_suspicious_description,
            self.rule_rare_account_usage,
            self.rule_weekend_posting,
        ):
            result = method()
            self._results[result.definition.key] = result
            LOGGER.info(
                "%-24s flagged %6s vouchers (%.2f%%)",
                result.definition.label,
                result.n_flagged,
                result.n_flagged / max(1, len(self.df)) * 100,
            )

        scored = self.df.copy()
        alert_rows: list[pd.DataFrame] = []

        # Noisy-OR combination of the individual rule scores.
        combined = pd.Series(0.0, index=self.df.index)

        for key, result in self._results.items():
            definition = result.definition
            scored[definition.column] = result.scores.round(4)
            scored[f"{key}_flag"] = result.flagged

            combined = 1.0 - (1.0 - combined) * (1.0 - definition.weight * result.scores)

            flagged_index = result.scores[result.flagged].index
            if len(flagged_index):
                alert_rows.append(
                    pd.DataFrame(
                        {
                            "transaction_id": scored.loc[flagged_index, "transaction_id"].to_numpy(),
                            "rule_key": key,
                            "rule_label": definition.label,
                            "rule_score": result.scores.loc[flagged_index].round(4).to_numpy(),
                            "risk_reason": result.reasons.loc[flagged_index].fillna("").to_numpy(),
                        }
                    )
                )

        scored["rule_risk_score"] = combined.round(4).clip(upper=1.0)
        scored["rule_alert_count"] = scored[[f"{key}_flag" for key in self._results]].sum(axis=1).astype(int)

        # Informational metric - measured, reported, but deliberately not scored.
        scored["description_repetition_flag"] = self.description_repetition_flag()

        # Collect every triggered reason into a single ordered list per voucher.
        scored["rule_reasons"] = self._collect_reasons()
        scored["rule_reason_text"] = scored["rule_reasons"].apply(lambda items: "; ".join(items))

        alerts = (
            pd.concat(alert_rows, ignore_index=True)
            if alert_rows
            else pd.DataFrame(columns=["transaction_id", "rule_key", "rule_label", "rule_score", "risk_reason"])
        )

        LOGGER.info(
            "Rule engine complete: %s alerts across %s vouchers (%.2f%% of the ledger flagged by at least one rule)",
            len(alerts),
            int((scored["rule_alert_count"] > 0).sum()),
            float((scored["rule_alert_count"] > 0).mean()) * 100,
        )
        return scored, alerts

    def _collect_reasons(self) -> pd.Series:
        """Return, for each voucher, the ordered list of triggered rule reasons."""
        # Order by rule weight descending so the strongest reason reads first.
        ordered = sorted(
            self._results.items(),
            key=lambda item: item[1].definition.weight,
            reverse=True,
        )
        reasons: list[list[str]] = [[] for _ in range(len(self.df))]
        for _key, result in ordered:
            for position in result.scores[result.flagged].index:
                text = result.reasons.loc[position]
                if isinstance(text, str) and text:
                    reasons[position].append(text)
        return pd.Series(reasons, index=self.df.index)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Score each rule against the labelled synthetic benchmark.

        For a given rule the positive class is ``anomaly_type == rule_key``, which
        answers the question an auditor actually cares about: *when this rule
        fires, how often is it pointing at the pattern it claims to detect?*

        Args:
            scored: Output of :meth:`run`.

        Returns:
            Dataframe with precision, recall and F1 per rule.
        """
        if "anomaly_label" not in scored.columns:
            raise KeyError("Ground-truth column 'anomaly_label' is required for evaluation.")

        rows: list[RuleEvaluation] = []
        for key, result in self._results.items():
            flagged = result.flagged
            if "anomaly_type" in scored.columns:
                truth = scored["anomaly_type"].astype("string") == key
            else:
                # The label survives a parquet round-trip as text ("0"/"1"), so a
                # bare ``== 1`` would silently match nothing and report a perfect
                # zero. ``as_flag_series`` is the one correct reading for every
                # dtype the label takes.
                truth = as_flag_series(scored["anomaly_label"])
            truth = truth.fillna(False)

            true_positives = int((flagged & truth).sum())
            false_positives = int((flagged & ~truth).sum())
            false_negatives = int((~flagged & truth).sum())

            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)

            rows.append(
                RuleEvaluation(
                    rule_key=key,
                    rule_label=result.definition.label,
                    flagged=int(flagged.sum()),
                    true_positives=true_positives,
                    false_positives=false_positives,
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    f1=round(f1, 4),
                )
            )

        return pd.DataFrame([row.to_dict() for row in rows])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_rule_engine(
    transactions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the rule engine over the feature-enriched transaction table.

    Args:
        transactions: Optional pre-loaded feature table. When omitted the table is
            loaded from ``data/processed/transactions_features.parquet`` (written by
            :mod:`src.feature_engineering`), falling back to the clean table - in
            which case :class:`AuditRuleEngine` rebuilds the features it needs.

    Returns:
        Mapping with ``scored``, ``alerts``, ``evaluation`` and ``engine``.
    """
    ensure_directories()
    if transactions is None:
        if TRANSACTIONS_FEATURES.exists():
            transactions = load_dataframe(TRANSACTIONS_FEATURES)
        else:
            LOGGER.warning(
                "%s not found - falling back to the clean table and rebuilding features.",
                TRANSACTIONS_FEATURES,
            )
            transactions = load_dataframe(TRANSACTIONS_CLEAN)

    engine = AuditRuleEngine(transactions)
    scored, alerts = engine.run()
    evaluation = engine.evaluate(scored)

    save_dataframe(alerts, RULE_ALERTS_CSV)
    LOGGER.info("Wrote %s", RULE_ALERTS_CSV)

    # The per-rule evaluation is persisted as well as logged: the dashboard's
    # Audit Rules page reports precision and recall alongside each procedure, and
    # recomputing it there would mean shipping the ground-truth labels into the UI.
    save_dataframe(evaluation, RULE_EVALUATION_CSV)
    LOGGER.info("Wrote %s", RULE_EVALUATION_CSV)

    return {
        "scored": scored,
        "alerts": alerts,
        "evaluation": evaluation,
        "engine": engine,
    }


def main() -> int:
    """CLI entry point."""
    with Timer("audit rule engine"):
        result = run_rule_engine()

    evaluation: pd.DataFrame = result["evaluation"]
    LOGGER.info("--- Rule performance against the labelled benchmark ---")
    for row in evaluation.itertuples(index=False):
        LOGGER.info(
            "  %-24s flagged=%6s  precision=%.3f  recall=%.3f  f1=%.3f",
            row.rule_label,
            row.flagged,
            row.precision,
            row.recall,
            row.f1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
