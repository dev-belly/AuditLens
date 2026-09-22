"""Tests for the Benford's Law analysis.

The digit-extraction tests are exact and cheap. The distributional tests use
log-uniform samples, which is the one family of random data that satisfies
Benford's Law by construction - so they verify the implementation rather than
tuning against whatever the synthetic ledger happens to look like.

One test exists purely to pin a piece of judgement: the result must always carry
the caveat that Benford's Law is not a fraud test. That caveat is a deliverable,
not decoration, and a refactor should not be able to drop it silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benford import (
    MAD_AT_MAXIMUM_RISK,
    MIN_AMOUNT_FOR_TEST,
    MIN_OBSERVATIONS,
    account_benford_risk,
    benford_by_group,
    extract_first_digit,
    extract_first_two_digits,
    first_digit_test,
    first_two_digit_test,
)
from src.utils import BENFORD_EXPECTED


def _log_uniform(size: int, low: float = 10.0, high: float = 1_000_000.0, seed: int = 7) -> pd.Series:
    """Draw values that are log-uniform on ``[low, high]``.

    A log-uniform draw satisfies Benford's Law by construction, which makes it the
    right fixture for testing the implementation rather than the data.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(10.0 ** rng.uniform(np.log10(low), np.log10(high), size))


# --------------------------------------------------------------------------- #
# First-digit extraction
# --------------------------------------------------------------------------- #
class TestExtractFirstDigit:
    """The leading digit, derived arithmetically."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (10.0, 1),
            (12.34, 1),
            (19.99, 1),
            (20.0, 2),
            (99.0, 9),
            (100.0, 1),
            (123.45, 1),
            (1_234.56, 1),
            (5_678.90, 5),
            (99_999.99, 9),
            (1_000_000.0, 1),
            (8_765_432.10, 8),
        ],
    )
    def test_known_values(self, amount: float, expected: int) -> None:
        result = extract_first_digit(pd.Series([amount]))
        assert int(result.iloc[0]) == expected

    def test_sign_is_ignored(self) -> None:
        """A credit of 1,200 starts with a 1, just as a debit of 1,200 does."""
        result = extract_first_digit(pd.Series([-1_234.0, 1_234.0]))
        assert result.iloc[0] == result.iloc[1] == 1

    def test_amounts_below_the_floor_are_dropped(self) -> None:
        result = extract_first_digit(pd.Series([9.99, 0.5, 12.0]))
        assert result.isna().tolist() == [True, True, False]
        assert int(result.iloc[2]) == 1

    def test_floor_is_configurable(self) -> None:
        result = extract_first_digit(pd.Series([9.99]), min_amount=1.0)
        assert int(result.iloc[0]) == 9

    def test_zero_and_nan_produce_no_digit(self) -> None:
        result = extract_first_digit(pd.Series([0.0, np.nan, 100.0]))
        assert result.isna().tolist() == [True, True, False]

    def test_index_is_preserved(self) -> None:
        amounts = pd.Series([100.0, 200.0], index=["a", "b"])
        result = extract_first_digit(amounts)
        assert result.index.tolist() == ["a", "b"]

    def test_all_outputs_are_valid_digits(self) -> None:
        result = extract_first_digit(_log_uniform(5_000)).dropna()
        assert result.between(1, 9).all()

    def test_floating_point_boundaries_are_not_misread(self) -> None:
        """1,000 must read as a 1, not a 9 from a log10 rounding artefact."""
        values = pd.Series([1_000.0, 10_000.0, 100_000.0, 1_000_000.0])
        assert extract_first_digit(values).tolist() == [1, 1, 1, 1]


# --------------------------------------------------------------------------- #
# First-two-digit extraction
# --------------------------------------------------------------------------- #
class TestExtractFirstTwoDigits:
    """The leading two digits, 10-99."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [(10.0, 10), (12.34, 12), (99.0, 99), (100.0, 10), (123.45, 12),
         (9_876.54, 98), (10_000.0, 10), (55_555.0, 55)],
    )
    def test_known_values(self, amount: float, expected: int) -> None:
        result = extract_first_two_digits(pd.Series([amount]))
        assert int(result.iloc[0]) == expected

    def test_outputs_stay_within_range(self) -> None:
        result = extract_first_two_digits(_log_uniform(5_000)).dropna()
        assert result.between(10, 99).all()

    def test_more_sensitive_than_the_first_digit_test(self) -> None:
        """90 buckets means patterns hidden inside one leading digit become visible."""
        amounts = _log_uniform(4_000)
        assert extract_first_two_digits(amounts).dropna().nunique() > extract_first_digit(amounts).dropna().nunique()


# --------------------------------------------------------------------------- #
# The first-digit test
# --------------------------------------------------------------------------- #
class TestFirstDigitTest:
    """Conformity statistics on the first digit."""

    def test_log_uniform_data_conforms(self) -> None:
        result = first_digit_test(_log_uniform(20_000))

        assert result.n_observations == 20_000
        assert result.conformity == "Close conformity"
        assert result.mad < 0.006
        assert len(result.table) == 9

    def test_expected_frequencies_match_benford(self) -> None:
        result = first_digit_test(_log_uniform(5_000))

        for row in result.table.itertuples(index=False):
            assert row.expected_frequency == pytest.approx(BENFORD_EXPECTED[row.digit], abs=1e-6)

    def test_observed_frequencies_sum_to_one(self) -> None:
        result = first_digit_test(_log_uniform(5_000))
        assert result.table["observed_frequency"].sum() == pytest.approx(1.0)

    def test_mad_is_the_mean_absolute_deviation_of_the_table(self) -> None:
        result = first_digit_test(_log_uniform(5_000))
        recomputed = result.table["deviation"].abs().mean()
        assert result.mad == pytest.approx(recomputed, abs=1e-9)

    def test_chi_square_is_non_negative_with_a_valid_p_value(self) -> None:
        result = first_digit_test(_log_uniform(5_000))

        assert result.chi_square >= 0.0
        assert 0.0 <= result.p_value <= 1.0
        assert result.degrees_of_freedom == 8

    def test_uniform_amounts_are_nonconforming(self) -> None:
        """Linear-uniform amounts over one decade put too much weight on digit 1.

        This is the counter-example that makes the test meaningful: data which is
        *not* log-distributed must be rejected, or the test proves nothing.
        """
        rng = np.random.default_rng(11)
        uniform = pd.Series(rng.uniform(10.0, 100.0, size=5_000))
        result = first_digit_test(uniform)

        assert result.conformity == "Nonconformity"
        assert result.mad > 0.015
        assert result.risk_indicator > 0.5

    def test_a_single_repeated_amount_is_nonconforming(self) -> None:
        """Fabricated entries that all sit on one round figure fail the test."""
        result = first_digit_test(pd.Series([50_000.0] * 1_000))

        assert result.conformity == "Nonconformity"
        assert result.risk_indicator == pytest.approx(1.0)

    def test_too_few_observations_is_reported_as_inconclusive(self) -> None:
        """The honest answer rather than a misleading number."""
        result = first_digit_test(_log_uniform(MIN_OBSERVATIONS - 50))

        assert result.conformity == "Insufficient data"
        assert np.isnan(result.mad)
        assert np.isnan(result.chi_square)
        assert result.risk_indicator == 0.0
        assert "no conclusion is drawn" in result.interpretation.lower()
        assert str(MIN_OBSERVATIONS) in result.interpretation

    def test_small_amounts_are_excluded_before_testing(self) -> None:
        amounts = pd.concat([_log_uniform(1_000), pd.Series([1.0, 2.0, 3.0] * 10)])
        result = first_digit_test(amounts)

        assert result.n_observations == 1_000
        assert result.min_amount == MIN_AMOUNT_FOR_TEST

    def test_risk_indicator_is_bounded(self) -> None:
        result = first_digit_test(_log_uniform(5_000))
        assert 0.0 <= result.risk_indicator <= 1.0

    def test_risk_indicator_saturates_at_the_configured_maximum(self) -> None:
        result = first_digit_test(pd.Series([50_000.0] * 1_000))
        assert result.risk_indicator == pytest.approx(1.0)
        assert MAD_AT_MAXIMUM_RISK == 0.03

    def test_interpretation_is_populated(self) -> None:
        result = first_digit_test(_log_uniform(5_000))
        assert len(result.interpretation) > 40

    def test_caveat_always_states_that_benford_is_not_a_fraud_test(self) -> None:
        """The caveat is a deliverable: it must survive any refactor."""
        for amounts in (_log_uniform(5_000), pd.Series([50_000.0] * 1_000)):
            caveat = first_digit_test(amounts).caveat.lower()
            assert "not a test for fraud" in caveat or "not establish" in caveat

    def test_supported_digits_are_all_nine(self) -> None:
        result = first_digit_test(_log_uniform(5_000))
        assert sorted(result.table["digit"].tolist()) == list(range(1, 10))


# --------------------------------------------------------------------------- #
# The first-two-digits test
# --------------------------------------------------------------------------- #
class TestFirstTwoDigitTest:
    """90 buckets, more sensitive to a localised pattern."""

    def test_log_uniform_data_conforms(self) -> None:
        result = first_two_digit_test(_log_uniform(20_000))

        assert result.n_observations == 20_000
        assert len(result.table) == 90
        assert result.conformity in {"Close conformity", "Acceptable conformity"}
        assert result.degrees_of_freedom == 89

    def test_expected_frequencies_are_benford_derived(self) -> None:
        result = first_two_digit_test(_log_uniform(5_000))
        expected = result.table["expected_frequency"].to_numpy()
        assert expected.sum() == pytest.approx(1.0, abs=1e-4)
        assert (expected > 0).all()

    def test_empty_buckets_are_excluded_from_the_chi_square(self) -> None:
        """A bucket with an expected count below one would make chi-square unstable."""
        result = first_two_digit_test(_log_uniform(2_000, low=10.0, high=200.0))

        assert result.chi_square >= 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_too_few_observations_is_inconclusive(self) -> None:
        result = first_two_digit_test(_log_uniform(100))
        assert result.conformity == "Insufficient data"

    def test_per_bucket_z_score_is_a_standardised_deviation(self) -> None:
        """The z-score must standardise each bucket by its own sampling error.

        The tolerance is not arbitrary: ``deviation`` is stored rounded to six
        decimals and ``z_score`` to three, and the buckets with the smallest
        expected frequency have a standard error near 4e-4, so a half-unit in the
        sixth decimal of the deviation moves the recomputed z by ~1.3e-3. The
        tolerance covers both roundings and nothing more - the z-scores in this
        population span roughly -2 to +2.
        """
        result = first_two_digit_test(_log_uniform(30_128))
        table = result.table

        expected = table["expected_frequency"].to_numpy()
        standard_error = np.sqrt(expected * (1.0 - expected) / result.n_observations)

        assert np.allclose(table["z_score"], table["deviation"] / standard_error, atol=5e-3)

    def test_significant_flag_follows_the_z_score(self) -> None:
        """Regression: the flag used to be ``|deviation| > 2 / sqrt(n)``.

        That is a fixed *absolute* threshold, applied to buckets whose expected
        frequencies differ by an order of magnitude. At n=30,128 the threshold is
        0.0115, which is larger than the expected frequency of most of the 90
        buckets - digit 99's is 0.0044 - so the flag was mathematically almost
        impossible to trigger and the column was inert. The test asserts the two
        properties that rule violated.
        """
        n = 30_128
        result = first_two_digit_test(_log_uniform(n))
        table = result.table

        assert bool((table["significant"] == (table["z_score"].abs() > 1.96)).all())

        # The threshold the old implementation used, against the distribution of
        # expected bucket sizes it was being applied to. It sits above the median
        # bucket and above roughly two thirds of them, which is the sense in which
        # the rule was inert.
        old_threshold = 2.0 / np.sqrt(n)
        assert old_threshold > float(table["expected_frequency"].median())
        assert (table["expected_frequency"] < old_threshold).mean() > 0.6

    def test_table_schema_is_stable_when_data_is_insufficient(self) -> None:
        """Callers can format the table without branching on the conformity band."""
        columns = {
            "digit", "observed_count", "observed_frequency", "expected_frequency",
            "deviation", "z_score", "significant",
        }

        assert columns.issubset(first_two_digit_test(_log_uniform(100)).table.columns)
        assert columns.issubset(first_digit_test(_log_uniform(100)).table.columns)


# --------------------------------------------------------------------------- #
# Disaggregation
# --------------------------------------------------------------------------- #
class TestDisaggregation:
    """Per-group tests, which is where Benford analysis becomes useful."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        rng = np.random.default_rng(3)
        conforming = pd.DataFrame(
            {
                "account_code": "6401",
                "debit_amount": 10.0 ** rng.uniform(1.5, 6.0, size=1_500),
            }
        )
        # A second account whose amounts cluster on one round figure - the classic
        # fabricated population.
        fabricated = pd.DataFrame(
            {
                "account_code": "1602",
                "debit_amount": np.full(400, 50_000.0),
            }
        )
        return pd.concat([conforming, fabricated], ignore_index=True)

    def test_by_group_returns_one_row_per_group(self) -> None:
        result = benford_by_group(self._frame(), "account_code")

        assert set(result["account_code"]) == {"6401", "1602"}
        assert {"mad", "chi_square", "p_value", "conformity", "risk_indicator"}.issubset(result.columns)

    def test_a_fabricated_group_is_separated_from_a_conforming_one(self) -> None:
        result = benford_by_group(self._frame(), "account_code").set_index("account_code")

        assert result.loc["1602", "conformity"] == "Nonconformity"
        assert result.loc["1602", "risk_indicator"] > result.loc["6401", "risk_indicator"]

    def test_groups_below_the_observation_floor_are_skipped_by_the_group_test(self) -> None:
        """The group test only reports accounts it can actually assess."""
        frame = self._frame()
        tiny = pd.DataFrame({"account_code": "9999", "debit_amount": [100.0, 200.0]})
        result = benford_by_group(pd.concat([frame, tiny], ignore_index=True), "account_code")

        assert "9999" not in set(result["account_code"])

    def test_account_benford_risk_scores_every_row(self) -> None:
        result = account_benford_risk(self._frame())

        assert len(result) == 2
        assert {"benford_mad", "benford_risk", "benford_conformity"}.issubset(result.columns)
        assert result["benford_risk"].between(0.0, 1.0).all()

    def test_account_risk_assigns_the_worst_group_the_highest_score(self) -> None:
        result = account_benford_risk(self._frame()).set_index("account_code")

        assert result.loc["1602", "benford_risk"] > result.loc["6401", "benford_risk"]

    def test_untestable_accounts_inherit_the_ledger_wide_value(self) -> None:
        """Keeps the transaction-level statistical signal defined for every row.

        Dropping these accounts instead would leave their vouchers with a
        statistical component of zero, which reads as "no statistical risk" when
        the honest statement is "not assessable".
        """
        frame = pd.concat(
            [
                self._frame(),
                pd.DataFrame({"account_code": "1221", "debit_amount": [500.0, 700.0]}),
            ],
            ignore_index=True,
        )
        result = account_benford_risk(frame).set_index("account_code")

        assert "1221" in result.index
        assert result.loc["1221", "benford_n_observations"] == 0
        assert result.loc["1221", "benford_risk"] > 0.0
        # The inherited value is the ledger-wide one, not a fabricated per-account one.
        assert result.loc["1221", "benford_risk"] == pytest.approx(result["benford_risk"].max(), abs=1e-9)
