"""Benford's Law analysis for AuditLens.

Benford's Law describes the expected frequency of leading digits in naturally
occurring numerical data: roughly 30.1% of values start with a 1, 17.6% with a 2,
and so on down to 4.6% for a 9. It holds for numbers that span several orders of
magnitude and are not artificially constrained - which is exactly what a general
ledger looks like.

**What it is for.** Benford analysis is an *analytical procedure*. It tells the
auditor where to look. A population that departs from the expected digit
distribution is worth disaggregating, because human-generated numbers - invented
invoices, manually keyed journals, threshold-dodging splits - do not follow the
law the way organic transaction data does.

**What it is not.** A deviation from Benford's Law is **not evidence of fraud**.
Perfectly legitimate ledgers deviate for entirely innocent reasons:

* the population is too small for the test to be meaningful;
* amounts are constrained by policy (a fixed monthly rent, a standard per-diem);
* the data contains assigned or sequential identifiers;
* a genuinely unusual business event moved the mix.

This module therefore reports a *compliance indicator* and a *risk indicator*, and
never a finding. Every result carries an explicit caveat. See
``docs/methodology.md`` for the full discussion.

Usage::

    python src/benford.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/benford.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import (
    BENFORD_EXPECTED,
    BENFORD_MAD_THRESHOLDS,
    BENFORD_RESULTS_JSON,
    TRANSACTIONS_CLEAN,
    TRANSACTIONS_FEATURES,
    Timer,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_json,
)

LOGGER = get_logger(__name__)

#: Amounts below this value are excluded from the first-digit test. Small values
#: are constrained by pricing conventions (everything costs 9.99) and would bias
#: the distribution. Nigrini's guidance is to test the digit of the *amount*, and
#: practitioners commonly set a floor of 10.
MIN_AMOUNT_FOR_TEST: float = 10.0

#: Minimum observations before a Benford test is considered meaningful. Below
#: roughly 300 values the confidence intervals are so wide that the test is not
#: worth running; the result is reported as "insufficient data" rather than
#: pretending to a conclusion.
MIN_OBSERVATIONS: int = 300

#: MAD threshold used for the first-two-digits test. It is an order of magnitude
#: tighter than the first-digit test because the expected frequencies are smaller.
MAD_THRESHOLD_FIRST_TWO: float = 0.0012

#: MAD value that maps to a risk indicator of 1.0. Chosen so that the Nigrini
#: conformity bands land on sensible risk values: 0.006 -> 0.20, 0.012 -> 0.40,
#: 0.015 -> 0.50, 0.030 -> 1.00.
MAD_AT_MAXIMUM_RISK: float = 0.030


# --------------------------------------------------------------------------- #
# Digit extraction
# --------------------------------------------------------------------------- #
def extract_first_digit(amounts: pd.Series, min_amount: float = MIN_AMOUNT_FOR_TEST) -> pd.Series:
    """Return the leading (most significant) digit of each amount.

    The digit is derived arithmetically rather than from string formatting, so it
    is unaffected by locale, thousands separators or rounding.

    Args:
        amounts: Monetary values. Signs are ignored - a credit of 1,200 starts
            with a 1 just as a debit of 1,200 does.
        min_amount: Values whose absolute amount is below this are dropped
            (returned as ``NA``).

    Returns:
        Series of integers in 1-9, with ``NA`` for values that were filtered out.
    """
    absolute = amounts.abs().astype("float64")
    usable = absolute >= min_amount

    digits = pd.Series(pd.NA, index=amounts.index, dtype="Int64")
    if not usable.any():
        return digits

    values = absolute[usable].to_numpy()
    # A tiny epsilon guards against floating-point error at exact powers of ten
    # (for example log10(1000) evaluating to 2.9999999999999996).
    exponent = np.floor(np.log10(values) + 1e-9)
    leading = values / np.power(10.0, exponent)
    leading_digit = np.floor(leading + 1e-9).astype(int)
    digits.loc[usable] = np.clip(leading_digit, 1, 9)

    return digits


def extract_first_two_digits(
    amounts: pd.Series, min_amount: float = MIN_AMOUNT_FOR_TEST
) -> pd.Series:
    """Return the leading two digits of each amount (10-99).

    The first-two-digits test is more sensitive than the first-digit test because
    it produces 90 buckets instead of 9, so a pattern hidden inside a single
    leading digit becomes visible.
    """
    absolute = amounts.abs().astype("float64")
    usable = absolute >= min_amount

    result = pd.Series(pd.NA, index=amounts.index, dtype="Int64")
    if not usable.any():
        return result

    values = absolute[usable].to_numpy()
    exponent = np.floor(np.log10(values) + 1e-9) - 1.0
    scaled = values / np.power(10.0, exponent)
    first_two = np.floor(scaled + 1e-9).astype(int)
    result.loc[usable] = np.clip(first_two, 10, 99)

    return result


# --------------------------------------------------------------------------- #
# Test results
# --------------------------------------------------------------------------- #
@dataclass
class BenfordResult:
    """Outcome of one Benford test.

    Attributes:
        test_name: ``"first_digit"`` or ``"first_two_digits"``.
        label: Human-readable test name.
        n_observations: Number of values actually tested.
        min_amount: Floor applied before the test.
        table: Per-digit observed vs expected frequencies, deviations and z-scores.
        mad: Mean absolute deviation between observed and expected frequencies.
        chi_square: Pearson chi-square statistic.
        degrees_of_freedom: Degrees of freedom for the chi-square test.
        p_value: p-value of the chi-square statistic.
        conformity: Nigrini conformity band.
        risk_indicator: 0-1 risk indicator derived from the MAD.
        interpretation: Plain-language reading of the result.
        caveat: The standing reminder that this is not evidence of fraud.
    """

    test_name: str
    label: str
    n_observations: int
    min_amount: float
    table: pd.DataFrame
    mad: float
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    conformity: str
    risk_indicator: float
    interpretation: str
    caveat: str = (
        "Benford's Law is an analytical procedure, not a test for fraud. A deviation "
        "identifies a population that deserves disaggregation and enquiry; it does not "
        "establish that anything is wrong."
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (excluding the full digit table)."""
        return {
            "test_name": self.test_name,
            "label": self.label,
            "n_observations": self.n_observations,
            "min_amount": self.min_amount,
            "mad": round(self.mad, 6),
            "chi_square": round(self.chi_square, 4),
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": float(f"{self.p_value:.6g}"),
            "conformity": self.conformity,
            "risk_indicator": round(self.risk_indicator, 4),
            "interpretation": self.interpretation,
            "caveat": self.caveat,
        }


def _conformity_band(mad: float, test_name: str = "first_digit") -> str:
    """Classify a MAD value into a Nigrini conformity band."""
    if test_name == "first_two_digits":
        if mad < MAD_THRESHOLD_FIRST_TWO:
            return "Close conformity"
        if mad < MAD_THRESHOLD_FIRST_TWO * 2:
            return "Acceptable conformity"
        if mad < MAD_THRESHOLD_FIRST_TWO * 4:
            return "Marginal conformity"
        return "Nonconformity"

    if mad < BENFORD_MAD_THRESHOLDS["close_conformity"]:
        return "Close conformity"
    if mad < BENFORD_MAD_THRESHOLDS["acceptable_conformity"]:
        return "Acceptable conformity"
    if mad < BENFORD_MAD_THRESHOLDS["marginal_conformity"]:
        return "Marginal conformity"
    return "Nonconformity"


def _risk_indicator(mad: float) -> float:
    """Map a MAD value onto a 0-1 risk indicator."""
    return float(np.clip(mad / MAD_AT_MAXIMUM_RISK, 0.0, 1.0))


def _interpretation(conformity: str, n: int, worst_digits: list[int]) -> str:
    """Build a plain-language reading of the result."""
    digits_text = ", ".join(str(digit) for digit in worst_digits) if worst_digits else "none"
    if conformity == "Close conformity":
        return (
            f"The digit distribution is consistent with Benford's Law across {n:,} values. "
            "No further digit-level testing is indicated on this population."
        )
    if conformity == "Acceptable conformity":
        return (
            f"The distribution sits within acceptable tolerance across {n:,} values. "
            f"Digits {digits_text} deviate most; consider whether a specific account, "
            "entity or process drives the difference."
        )
    if conformity == "Marginal conformity":
        return (
            f"The distribution is marginally outside tolerance across {n:,} values, with "
            f"digits {digits_text} showing the largest deviations. Disaggregate the "
            "population by account, vendor and preparer before drawing any conclusion."
        )
    return (
        f"The distribution departs materially from Benford's Law across {n:,} values, with "
        f"digits {digits_text} deviating most. This warrants disaggregation and enquiry, but "
        "common innocent explanations - policy-constrained amounts, assigned numbers or a "
        "small population - should be excluded first."
    )


def first_digit_test(
    amounts: pd.Series,
    min_amount: float = MIN_AMOUNT_FOR_TEST,
    min_observations: int = MIN_OBSERVATIONS,
) -> BenfordResult:
    """Run the first-digit Benford test.

    Args:
        amounts: Monetary values to test.
        min_amount: Values below this are excluded.
        min_observations: Below this the test is reported as inconclusive.

    Returns:
        A :class:`BenfordResult`. When there are too few observations the MAD and
        chi-square are returned as ``nan`` and the conformity band reads
        "Insufficient data" - the honest answer rather than a misleading number.
    """
    digits = extract_first_digit(amounts, min_amount).dropna()
    n = int(len(digits))

    expected = np.array([BENFORD_EXPECTED[digit] for digit in range(1, 10)], dtype=float)
    observed_count = digits.value_counts().reindex(range(1, 10), fill_value=0).to_numpy(dtype=float)

    if n < min_observations:
        table = pd.DataFrame(
            {
                "digit": range(1, 10),
                "observed_count": observed_count.astype(int),
                "observed_frequency": (observed_count / max(n, 1)).round(6),
                "expected_frequency": expected.round(6),
            }
        )
        table["deviation"] = (table["observed_frequency"] - table["expected_frequency"]).round(6)
        # The z-score column is kept, as nan, so that the table's schema does not
        # depend on whether the test was conclusive. Callers can then format the
        # table without branching on the conformity band.
        table["z_score"] = np.nan
        table["significant"] = False
        return BenfordResult(
            test_name="first_digit",
            label="First-digit test",
            n_observations=n,
            min_amount=min_amount,
            table=table,
            mad=float("nan"),
            chi_square=float("nan"),
            degrees_of_freedom=8,
            p_value=float("nan"),
            conformity="Insufficient data",
            risk_indicator=0.0,
            interpretation=(
                f"Only {n:,} values meet the test criteria, below the {min_observations:,} "
                "needed for a meaningful first-digit test. No conclusion is drawn."
            ),
        )

    observed_frequency = observed_count / n
    deviation = observed_frequency - expected
    mad = float(np.mean(np.abs(deviation)))

    expected_count = expected * n
    chi_square = float(np.sum((observed_count - expected_count) ** 2 / expected_count))
    degrees_of_freedom = 8
    p_value = float(stats.chi2.sf(chi_square, degrees_of_freedom))

    # Per-digit z-score: is this digit's deviation larger than sampling noise?
    standard_error = np.sqrt(expected * (1.0 - expected) / n)
    z_score = deviation / standard_error

    table = pd.DataFrame(
        {
            "digit": range(1, 10),
            "observed_count": observed_count.astype(int),
            "observed_frequency": observed_frequency.round(6),
            "expected_frequency": expected.round(6),
            "deviation": deviation.round(6),
            "z_score": z_score.round(3),
        }
    )
    table["significant"] = table["z_score"].abs() > 1.96

    conformity = _conformity_band(mad, "first_digit")
    worst_digits = (
        table.reindex(table["z_score"].abs().sort_values(ascending=False).index)["digit"]
        .head(3)
        .tolist()
    )

    return BenfordResult(
        test_name="first_digit",
        label="First-digit test",
        n_observations=n,
        min_amount=min_amount,
        table=table,
        mad=mad,
        chi_square=chi_square,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        conformity=conformity,
        risk_indicator=_risk_indicator(mad),
        interpretation=_interpretation(conformity, n, [int(d) for d in worst_digits]),
    )


def first_two_digit_test(
    amounts: pd.Series,
    min_amount: float = MIN_AMOUNT_FOR_TEST,
    min_observations: int = MIN_OBSERVATIONS,
) -> BenfordResult:
    """Run the first-two-digits Benford test (90 buckets, more sensitive)."""
    first_two = extract_first_two_digits(amounts, min_amount).dropna()
    n = int(len(first_two))

    digits = np.arange(10, 100)
    expected = np.log10(1.0 + 1.0 / digits)
    expected = expected / expected.sum()

    observed_count = first_two.value_counts().reindex(digits, fill_value=0).to_numpy(dtype=float)

    if n < min_observations:
        table = pd.DataFrame(
            {
                "digit": digits,
                "observed_count": observed_count.astype(int),
                "observed_frequency": (observed_count / max(n, 1)).round(6),
                "expected_frequency": expected.round(6),
            }
        )
        table["deviation"] = (table["observed_frequency"] - table["expected_frequency"]).round(6)
        # Same stable-schema contract as the first-digit test.
        table["z_score"] = np.nan
        table["significant"] = False
        return BenfordResult(
            test_name="first_two_digits",
            label="First-two-digits test",
            n_observations=n,
            min_amount=min_amount,
            table=table,
            mad=float("nan"),
            chi_square=float("nan"),
            degrees_of_freedom=89,
            p_value=float("nan"),
            conformity="Insufficient data",
            risk_indicator=0.0,
            interpretation=(
                f"Only {n:,} values meet the test criteria, below the {min_observations:,} "
                "needed for a meaningful first-two-digits test."
            ),
        )

    observed_frequency = observed_count / n
    deviation = observed_frequency - expected
    mad = float(np.mean(np.abs(deviation)))

    expected_count = expected * n
    # Buckets with an expected count below 1 make the chi-square statistic unstable,
    # so those buckets are excluded from the statistic (not from the MAD).
    usable = expected_count >= 1.0
    chi_square = float(np.sum((observed_count[usable] - expected_count[usable]) ** 2 / expected_count[usable]))
    degrees_of_freedom = int(usable.sum()) - 1
    p_value = float(stats.chi2.sf(chi_square, degrees_of_freedom)) if degrees_of_freedom > 0 else float("nan")

    table = pd.DataFrame(
        {
            "digit": digits,
            "observed_count": observed_count.astype(int),
            "observed_frequency": observed_frequency.round(6),
            "expected_frequency": expected.round(6),
            "deviation": deviation.round(6),
            # Same per-bucket z-score as the first-digit test: the observed
            # frequency against the expected one, in units of the binomial
            # standard error. The earlier version flagged a bucket when
            # ``|deviation| > 2 / sqrt(n)``, which is a fixed absolute threshold.
            # At n=30,128 that threshold is 0.0115, while the expected frequency
            # of most buckets is far smaller than that - digit 99's is 0.0044 - so
            # the flag could essentially never fire and the column was inert. The
            # z-score scales each bucket by its own expected frequency, which is
            # the comparison the test actually intends.
            "z_score": (deviation / np.sqrt(expected * (1.0 - expected) / n)).round(3),
        }
    )
    # 1.96 is the two-sided 5% critical value. With 90 buckets roughly 4-5
    # exceedances are expected by chance alone, so this column identifies where to
    # look, not what is wrong.
    table["significant"] = table["z_score"].abs() > 1.96

    conformity = _conformity_band(mad, "first_two_digits")
    worst = table.reindex(table["deviation"].abs().sort_values(ascending=False).index)["digit"].head(3)

    return BenfordResult(
        test_name="first_two_digits",
        label="First-two-digits test",
        n_observations=n,
        min_amount=min_amount,
        table=table,
        mad=mad,
        chi_square=chi_square,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        conformity=conformity,
        risk_indicator=_risk_indicator(mad),
        interpretation=_interpretation(conformity, n, [int(d) for d in worst.tolist()]),
    )


# --------------------------------------------------------------------------- #
# Disaggregated analysis
# --------------------------------------------------------------------------- #
def benford_by_group(
    transactions: pd.DataFrame,
    group_column: str,
    amount_column: str = "debit_amount",
    min_observations: int = MIN_OBSERVATIONS,
    min_amount: float = MIN_AMOUNT_FOR_TEST,
) -> pd.DataFrame:
    """Run the first-digit test separately for each value of ``group_column``.

    Disaggregation is the whole point of Benford analysis in audit. A group whose
    digits depart from the expected distribution - while the population as a whole
    looks fine - is far more informative than a single ledger-wide statistic,
    because deviations in different directions can cancel each other out.

    Args:
        transactions: Transaction table.
        group_column: Column to group by (``account_code``, ``vendor_id``,
            ``process``, ``department``, ...).
        amount_column: Amount column to test.
        min_observations: Groups with fewer observations are skipped.
        min_amount: Floor applied to the amounts.

    Returns:
        One row per group, sorted by descending MAD.
    """
    rows: list[dict[str, Any]] = []

    for group_value, frame in transactions.groupby(group_column, dropna=True, observed=True):
        result = first_digit_test(
            frame[amount_column], min_amount=min_amount, min_observations=min_observations
        )
        if result.n_observations < min_observations:
            continue
        rows.append(
            {
                group_column: group_value,
                "n_observations": result.n_observations,
                "total_amount": round(float(frame[amount_column].sum()), 2),
                "mad": round(result.mad, 6),
                "chi_square": round(result.chi_square, 4),
                "p_value": float(f"{result.p_value:.6g}"),
                "conformity": result.conformity,
                "risk_indicator": round(result.risk_indicator, 4),
            }
        )

    if not rows:
        LOGGER.warning("No group in '%s' had enough observations for a Benford test.", group_column)
        return pd.DataFrame(
            columns=[group_column, "n_observations", "total_amount", "mad", "chi_square", "p_value", "conformity", "risk_indicator"]
        )

    result_frame = pd.DataFrame(rows).sort_values("mad", ascending=False).reset_index(drop=True)
    LOGGER.info(
        "Benford by %s: tested %s groups (%s non-conforming)",
        group_column,
        len(result_frame),
        int((result_frame["conformity"] == "Nonconformity").sum()),
    )
    return result_frame


def account_benford_risk(
    transactions: pd.DataFrame,
    account_column: str = "account_code",
    amount_column: str = "debit_amount",
    min_observations: int = MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Return a per-account Benford risk score for transaction-level scoring.

    The composite Audit Risk Score needs a *statistical* component at voucher
    level. Benford's Law is a population test, so the honest way to translate it
    into a transaction-level signal is to score each voucher by how far the
    digit distribution of **its own account** departs from the expected
    distribution. Accounts with too few observations inherit the ledger-wide
    value, which keeps the signal defined for every row.

    Returns:
        Dataframe with ``account_column``, ``benford_mad``, ``benford_risk`` and
        ``benford_conformity``.
    """
    ledger_result = first_digit_test(transactions[amount_column])
    ledger_risk = ledger_result.risk_indicator if np.isfinite(ledger_result.mad) else 0.0
    ledger_mad = ledger_result.mad if np.isfinite(ledger_result.mad) else 0.0

    grouped = benford_by_group(
        transactions, account_column, amount_column, min_observations=min_observations
    )

    if grouped.empty:
        accounts = transactions[account_column].dropna().unique()
        return pd.DataFrame(
            {
                account_column: accounts,
                "benford_mad": ledger_mad,
                "benford_risk": ledger_risk,
                "benford_conformity": ledger_result.conformity,
                "benford_n_observations": 0,
            }
        )

    grouped = grouped.rename(
        columns={"mad": "benford_mad", "risk_indicator": "benford_risk", "conformity": "benford_conformity"}
    )
    grouped["benford_n_observations"] = grouped["n_observations"]

    # Accounts below the observation floor cannot be tested on their own. They are
    # given the ledger-wide result rather than being dropped: dropping them would
    # leave those vouchers with a statistical component of zero, which reads as
    # "no statistical risk" when the honest statement is "not assessable". An
    # inherited value is the better of the two errors, and
    # ``benford_n_observations`` stays at 0 so the inheritance is visible.
    tested = set(grouped[account_column].astype("string"))
    untested = [
        account
        for account in transactions[account_column].dropna().astype("string").unique()
        if account not in tested
    ]
    if untested:
        LOGGER.info(
            "%s account(s) below the %s-observation floor inherit the ledger-wide Benford result.",
            len(untested),
            min_observations,
        )
        grouped = pd.concat(
            [
                grouped,
                pd.DataFrame(
                    {
                        account_column: untested,
                        "benford_mad": ledger_mad,
                        "benford_risk": ledger_risk,
                        "benford_conformity": ledger_result.conformity,
                        "benford_n_observations": 0,
                    }
                ),
            ],
            ignore_index=True,
        )

    return grouped[
        [account_column, "benford_mad", "benford_risk", "benford_conformity", "benford_n_observations"]
    ]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_benford_analysis(transactions: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the complete Benford analysis and persist the results.

    Args:
        transactions: Optional transaction table. Loaded from disk when omitted.

    Returns:
        Mapping with the ledger-wide results, the per-account breakdown and the
        per-account risk table.
    """
    ensure_directories()
    if transactions is None:
        source = TRANSACTIONS_FEATURES if TRANSACTIONS_FEATURES.exists() else TRANSACTIONS_CLEAN
        transactions = load_dataframe(source)

    # Only positive amounts are tested: a credit of 1,200 and a debit of 1,200
    # both start with a 1, so the sign is irrelevant to the leading digit, but
    # zero and negative-correction rows carry no digit information.
    amounts = transactions["debit_amount"]

    first_digit = first_digit_test(amounts)
    first_two = first_two_digit_test(amounts)
    by_account = benford_by_group(transactions, "account_code")
    by_process = benford_by_group(transactions, "process")
    account_risk = account_benford_risk(transactions)

    payload = {
        "first_digit_test": first_digit.to_dict(),
        "first_two_digits_test": first_two.to_dict(),
        "first_digit_table": first_digit.table.to_dict(orient="records"),
        "first_two_digits_table": first_two.table.to_dict(orient="records"),
        "by_account": by_account.to_dict(orient="records"),
        "by_process": by_process.to_dict(orient="records"),
        "scope_note": (
            "Only the debit leg is tested. Each voucher is stored once, with "
            "debit_amount equal to credit_amount, so the debit leg is the voucher value."
        ),
    }
    save_json(payload, BENFORD_RESULTS_JSON)
    LOGGER.info("Wrote %s", BENFORD_RESULTS_JSON)

    return {
        "first_digit": first_digit,
        "first_two_digits": first_two,
        "by_account": by_account,
        "by_process": by_process,
        "account_risk": account_risk,
        "payload": payload,
    }


def main() -> int:
    """CLI entry point."""
    with Timer("Benford analysis"):
        result = run_benford_analysis()

    for key in ("first_digit", "first_two_digits"):
        outcome: BenfordResult = result[key]
        LOGGER.info("--- %s ---", outcome.label)
        LOGGER.info("  observations   %s", outcome.n_observations)
        LOGGER.info("  MAD            %.6f", outcome.mad)
        LOGGER.info("  chi-square     %.2f (df=%s, p=%s)", outcome.chi_square, outcome.degrees_of_freedom, f"{outcome.p_value:.4g}")
        LOGGER.info("  conformity     %s", outcome.conformity)
        LOGGER.info("  risk indicator %.4f", outcome.risk_indicator)
        LOGGER.info("  %s", outcome.interpretation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
