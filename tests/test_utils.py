"""Tests for the shared helpers in :mod:`src.utils`.

``as_flag_series`` carries most of the weight here. The ground-truth label column
round-trips through parquet as text, and both obvious ways of reading it are wrong in
opposite directions - ``== 1`` marks nothing, ``astype(bool)`` marks everything. Both
bugs were live in this codebase at different times, so the coercion is tested against
every dtype the label actually takes.
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

from src.utils import REASON_SEPARATOR, as_flag_series, as_list, as_reason_list


class TestAsFlagSeries:
    """A 0/1 column read correctly, whatever dtype it arrived in."""

    @pytest.mark.parametrize(
        "values",
        [
            [0, 1, 1, 0],
            [0.0, 1.0, 1.0, 0.0],
            [False, True, True, False],
            ["0", "1", "1", "0"],
            pd.array([0, 1, 1, 0], dtype="Int64"),
            pd.array([0, 1, 1, 0], dtype="boolean"),
        ],
    )
    def test_every_dtype_reads_the_same_way(self, values: list) -> None:
        result = as_flag_series(pd.Series(values, dtype=object if isinstance(values[0], str) else None))

        assert result.tolist() == [False, True, True, False]

    def test_the_string_zero_is_not_truthy(self) -> None:
        """The dangerous case: ``pd.Series(["0"]).astype(bool)`` is ``True``."""
        series = pd.Series(["0", "1"], dtype="string")

        assert as_flag_series(series).tolist() == [False, True]
        # Document the trap this helper exists to avoid.
        assert series.astype(bool).tolist() == [True, True]

    def test_the_string_one_is_not_equal_to_the_integer_one(self) -> None:
        """The other half of the trap, which fails the opposite way."""
        series = pd.Series(["0", "1"], dtype="string")

        assert (series == 1).tolist() == [False, False]
        assert as_flag_series(series).sum() == 1

    def test_missing_values_are_treated_as_not_flagged(self) -> None:
        series = pd.Series([1, None, np.nan, 0], dtype="float64")

        assert as_flag_series(series).tolist() == [True, False, False, False]

    def test_the_result_is_aligned_to_the_input_index(self) -> None:
        series = pd.Series([1, 0], index=[10, 20])

        result = as_flag_series(series)

        assert result.index.tolist() == [10, 20]
        assert result.dtype == bool

    def test_an_empty_series_is_handled(self) -> None:
        result = as_flag_series(pd.Series([], dtype="object"))

        assert result.empty


class TestAsList:
    """Explanation cells arrive in three shapes and must normalise to one."""

    def test_none_becomes_an_empty_list(self) -> None:
        assert as_list(None) == []

    def test_a_bare_string_becomes_a_single_item_list(self) -> None:
        assert as_list("urgent payment") == ["urgent payment"]

    def test_a_delimited_string_is_split_when_a_separator_is_given(self) -> None:
        assert as_list("one; two; three", separator="; ") == ["one", "two", "three"]

    def test_a_numpy_array_becomes_a_list(self) -> None:
        """A parquet round-trip turns the stored list into an ndarray."""
        assert as_list(np.array(["a", "b"])) == ["a", "b"]

    def test_blank_entries_are_dropped_when_splitting(self) -> None:
        assert as_list("a; ; b", separator="; ") == ["a", "b"]


class TestAsReasonList:
    """Reasons are always strings, so the UI never has to check."""

    def test_reasons_round_trip_through_the_delimiter(self) -> None:
        reasons = ["Weekend posting", "Large round amount"]

        assert as_reason_list(REASON_SEPARATOR.join(reasons)) == reasons

    def test_non_string_items_are_coerced(self) -> None:
        assert as_reason_list([1, 2]) == ["1", "2"]

    def test_none_becomes_an_empty_list(self) -> None:
        assert as_reason_list(None) == []
