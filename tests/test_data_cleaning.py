"""Tests for the data cleaning and data quality module.

The cleaning stage is where an audit analytics project either earns trust or loses
it. Two properties are asserted throughout:

* **Nothing is invented.** The cleaner repairs what can be repaired from
  authoritative sources (a posting date from the transaction date, an account name
  from the chart of accounts) and flags the rest. It never imputes a missing
  narration, because a fabricated description would destroy the very signal the
  suspicious-description rule depends on.
* **Every issue is counted.** A repair that is not reported is a silent
  restatement of the population, so the quality report is tested as carefully as
  the transformation itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import data_cleaning as cleaning
from src.data_cleaning import (
    CURRENCY_ALIASES,
    DataQualityReport,
    clean_master_data,
    clean_transactions,
    coerce_numeric,
    flag_orphan_vendors,
    flag_unbalanced,
    normalise_currency,
    normalise_text,
    parse_dates,
    remove_duplicate_rows,
    repair_account_names,
)
from src.utils import ACCOUNTS, BASE_CURRENCY, FX_RATES_TO_CNY
from tests.conftest import BASE_TRANSACTION, WEEKDAY


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build a raw, ERP-like frame from a list of field overrides.

    Deliberately does *not* reuse the ``make_transactions`` helper: that helper
    parses the date columns strictly, which would raise on the malformed dates
    this module exists to handle. Here the date columns are serialised to text and
    anything already supplied as a string is passed through untouched.
    """
    records: list[dict] = []
    for index, overrides in enumerate(rows, start=1):
        record = dict(BASE_TRANSACTION)
        record["transaction_id"] = f"TX{index:04d}"
        record.update(overrides)
        records.append(record)

    frame = pd.DataFrame(records)
    for column in ("transaction_date", "posting_date", "approval_time", "invoice_time", "payment_time"):
        frame[column] = frame[column].map(
            lambda value: value.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(value, pd.Timestamp)
            else value
        )
    return frame


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
class TestNormaliseText:
    """Whitespace hygiene, which is what breaks joins in practice."""

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        frame = normalise_text(_frame([{"vendor_name": "  Acme Ltd  ", "department": " Procurement "}]))

        assert frame["vendor_name"].iloc[0] == "Acme Ltd"
        assert frame["department"].iloc[0] == "Procurement"

    def test_repeated_internal_spaces_are_collapsed(self) -> None:
        frame = normalise_text(_frame([{"vendor_name": "Acme    Trading     Ltd"}]))
        assert frame["vendor_name"].iloc[0] == "Acme Trading Ltd"

    def test_empty_strings_become_missing(self) -> None:
        """An empty narration is a missing narration, not valid text."""
        frame = normalise_text(_frame([{"description": "   "}]))
        assert pd.isna(frame["description"].iloc[0])

    def test_original_frame_is_not_mutated(self) -> None:
        source = _frame([{"vendor_name": "  Acme Ltd  "}])
        normalise_text(source)
        assert source["vendor_name"].iloc[0] == "  Acme Ltd  "


# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #
class TestNormaliseCurrency:
    """ERP currency codes are messy; the amount is only meaningful once fixed."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("CNY", "CNY"), ("cny", "CNY"), (" RMB ", "CNY"), ("RMB", "CNY"),
            ("YUAN", "CNY"), ("USD", "USD"), ("usd", "USD"), ("US$", "USD"),
            ("HKD", "HKD"), ("eur", "EUR"), ("JPY", "JPY"), ("SGD", "SGD"),
        ],
    )
    def test_aliases_and_case_are_normalised(self, raw: str, expected: str) -> None:
        frame, _ = normalise_currency(_frame([{"currency": raw}]))
        assert frame["currency"].iloc[0] == expected

    def test_every_supported_currency_survives_normalisation(self) -> None:
        """Nothing the FX table supports should be rewritten."""
        frame, changed = normalise_currency(
            _frame([{"currency": code} for code in FX_RATES_TO_CNY])
        )

        assert changed == 0
        assert set(frame["currency"]) == set(FX_RATES_TO_CNY)

    def test_unknown_codes_fall_back_to_the_reporting_currency(self) -> None:
        frame, _ = normalise_currency(_frame([{"currency": "XYZ"}]))
        assert frame["currency"].iloc[0] == BASE_CURRENCY

    def test_changed_row_count_is_reported(self) -> None:
        """Only genuinely changed rows are counted, or the report inflates."""
        frame, changed = normalise_currency(
            _frame([{"currency": "CNY"}, {"currency": "RMB"}, {"currency": "xyz"}])
        )

        assert changed == 2
        assert frame["currency"].tolist() == ["CNY", "CNY", "CNY"]

    def test_missing_currency_is_treated_as_unknown(self) -> None:
        frame, _ = normalise_currency(_frame([{"currency": None}]))
        assert frame["currency"].iloc[0] == BASE_CURRENCY

    def test_every_alias_target_is_a_supported_currency(self) -> None:
        assert set(CURRENCY_ALIASES.values()).issubset(set(FX_RATES_TO_CNY))


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
class TestParseDates:
    """Invalid dates, and the one repair that is safe to make."""

    def test_valid_dates_are_parsed(self) -> None:
        frame, invalid_tx, invalid_posting, repaired = parse_dates(_frame([{}]))

        assert pd.api.types.is_datetime64_any_dtype(frame["transaction_date"])
        assert (invalid_tx, invalid_posting, repaired) == (0, 0, 0)

    def test_unparseable_transaction_date_is_counted_not_repaired(self) -> None:
        """A voucher with no date cannot be placed in a period, so it is not guessed."""
        frame, invalid_tx, _, _ = parse_dates(_frame([{"transaction_date": "not-a-date"}]))

        assert invalid_tx == 1
        assert pd.isna(frame["transaction_date"].iloc[0])

    def test_posting_date_is_repaired_from_the_transaction_date(self) -> None:
        """The safe repair: a posting date is almost always within days of the voucher."""
        frame, _, invalid_posting, repaired = parse_dates(
            _frame([{"transaction_date": WEEKDAY, "posting_date": "??"}])
        )

        assert invalid_posting == 1
        assert repaired == 1
        assert frame["posting_date"].iloc[0] == pd.Timestamp(WEEKDAY)

    def test_invalid_counts_are_measured_before_repair(self) -> None:
        """The quality report must reflect what arrived, not what was fixed."""
        frame, _, invalid_posting, repaired = parse_dates(
            _frame([{"transaction_date": WEEKDAY, "posting_date": None}])
        )

        assert invalid_posting == repaired == 1

    def test_timestamps_are_parsed(self) -> None:
        frame, _, _, _ = parse_dates(_frame([{}]))
        for column in ("approval_time", "invoice_time", "payment_time"):
            assert pd.api.types.is_datetime64_any_dtype(frame[column]), column


# --------------------------------------------------------------------------- #
# Numeric coercion
# --------------------------------------------------------------------------- #
class TestCoerceNumeric:
    """Amounts arrive as text when the export is CSV."""

    def test_numeric_strings_are_converted(self) -> None:
        frame = pd.DataFrame({"debit_amount": ["1234.56", "2000", "3000.00"]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert result["debit_amount"].iloc[0] == pytest.approx(1_234.56)
        assert result["debit_amount"].iloc[2] == pytest.approx(3_000.0)

    def test_thousands_separators_and_symbols_are_stripped(self) -> None:
        """A bare to_numeric would turn every one of these into a missing amount."""
        frame = pd.DataFrame({"debit_amount": ["1,234.56", "¥ 1 234.56", "CNY1,234.56", "1\u00a0234.56"]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert result["debit_amount"].notna().all()
        assert result["debit_amount"].tolist() == pytest.approx([1_234.56] * 4)

    def test_accounting_negatives_in_brackets_are_recognised(self) -> None:
        frame = pd.DataFrame({"debit_amount": ["(1,234.56)", "1,234.56"]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert result["debit_amount"].iloc[0] == pytest.approx(-1_234.56)
        assert result["debit_amount"].iloc[1] == pytest.approx(1_234.56)

    def test_unparseable_values_become_missing(self) -> None:
        frame = pd.DataFrame({"debit_amount": ["abc", "100"]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert pd.isna(result["debit_amount"].iloc[0])
        assert result["debit_amount"].iloc[1] == pytest.approx(100.0)

    def test_blank_strings_become_missing(self) -> None:
        frame = pd.DataFrame({"debit_amount": ["", "   ", "100"]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert result["debit_amount"].isna().tolist() == [True, True, False]

    def test_already_numeric_columns_are_untouched(self) -> None:
        """The pipeline's own files are numeric; the string path must not alter them."""
        frame = pd.DataFrame({"debit_amount": [1_234.567890123, 0.1, 1e15]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert result["debit_amount"].tolist() == pytest.approx(frame["debit_amount"].tolist())

    def test_missing_column_is_ignored(self) -> None:
        frame = pd.DataFrame({"other": [1, 2]})
        result = coerce_numeric(frame, ("debit_amount",))

        assert list(result.columns) == ["other"]

    def test_nonfinite_numeric_values_become_missing(self) -> None:
        frame = pd.DataFrame({"amount": [float("inf"), float("-inf"), 100.0]})
        result = coerce_numeric(frame, ("amount",))

        assert result["amount"].isna().tolist() == [True, True, False]
        assert result["amount"].iloc[2] == 100.0


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
class TestRemoveDuplicates:
    """Exact duplicate vouchers, which are an extract artefact."""

    def test_exact_duplicates_are_removed_and_counted(self) -> None:
        rows = [
            {"transaction_id": "TX0001", "invoice_id": "INV1"},
            {"transaction_id": "TX0001", "invoice_id": "INV1"},
            {"transaction_id": "TX0002", "invoice_id": "INV2"},
        ]
        frame, removed = remove_duplicate_rows(pd.DataFrame(rows))

        assert removed == 1
        assert len(frame) == 2

    def test_vouchers_sharing_an_invoice_are_not_duplicates(self) -> None:
        """Two payments on one invoice is a finding, not an extract error."""
        rows = [
            {"transaction_id": "TX0001", "invoice_id": "INV1", "debit_amount": 1_000.0},
            {"transaction_id": "TX0002", "invoice_id": "INV1", "debit_amount": 2_000.0},
        ]
        frame, removed = remove_duplicate_rows(pd.DataFrame(rows))

        assert removed == 0
        assert len(frame) == 2

    @pytest.mark.parametrize(
        ("field", "changed"),
        [
            ("currency", "USD"),
            ("fx_rate", 7.0),
            ("invoice_id", "INV-DIFFERENT"),
            ("approved_by", "E9999"),
            ("credit_account_code", "1002"),
            ("posting_date", WEEKDAY + pd.Timedelta(days=1)),
        ],
    )
    def test_conflicting_source_evidence_is_not_silently_deduplicated(
        self, field: str, changed: object
    ) -> None:
        original = dict(BASE_TRANSACTION)
        conflicting = {**original, field: changed}
        frame, removed = remove_duplicate_rows(pd.DataFrame([original, conflicting, original]))

        assert removed == 1
        assert len(frame) == 2
        assert frame[field].tolist() == [original[field], changed]


# --------------------------------------------------------------------------- #
# Account names
# --------------------------------------------------------------------------- #
class TestRepairAccountNames:
    """The account code is authoritative; the name is a description."""

    def test_mismatched_name_is_restored_from_the_chart_of_accounts(self) -> None:
        frame = _frame([{"account_code": "6401", "account_name": "Some Wrong Name"}])
        result, mismatches = repair_account_names(frame)

        assert mismatches == 1
        assert result["account_name"].iloc[0] == ACCOUNTS["6401"]["name"]

    def test_a_correct_name_is_left_alone(self) -> None:
        frame = _frame([{"account_code": "6401", "account_name": ACCOUNTS["6401"]["name"]}])
        _, mismatches = repair_account_names(frame)

        assert mismatches == 0

    def test_an_unknown_account_code_is_not_invented(self) -> None:
        """A code absent from the chart of accounts has no authoritative name."""
        frame = _frame([{"account_code": "9999", "account_name": "Mystery Account"}])
        result, mismatches = repair_account_names(frame)

        assert mismatches == 0
        assert result["account_name"].iloc[0] == "Mystery Account"


# --------------------------------------------------------------------------- #
# Referential integrity and balance
# --------------------------------------------------------------------------- #
class TestFlagOrphanVendors:
    """A voucher pointing at a vendor that does not exist in the master file."""

    def test_unknown_vendor_is_flagged_but_kept(self) -> None:
        frame = _frame([{"vendor_id": "V0001"}, {"vendor_id": "V-GHOST"}])
        vendors = pd.DataFrame({"vendor_id": ["V0001"]})

        result, orphans = flag_orphan_vendors(frame, vendors)

        assert orphans == 1
        assert result["orphan_vendor_flag"].tolist() == [False, True]
        assert len(result) == 2, "orphans must be kept - dropping them understates the population"

    def test_a_missing_vendor_id_is_not_an_orphan(self) -> None:
        """Employee reimbursements have no vendor; that is not a master-data gap."""
        frame = _frame([{"vendor_id": None}])
        result, orphans = flag_orphan_vendors(frame, pd.DataFrame({"vendor_id": ["V0001"]}))

        assert orphans == 0
        assert not result["orphan_vendor_flag"].any()


class TestFlagUnbalanced:
    """The single most important control in double-entry bookkeeping."""

    def test_matching_legs_are_not_flagged(self) -> None:
        frame = _frame([{"debit_amount": 1_000.0, "credit_amount": 1_000.0}])
        _, unbalanced = flag_unbalanced(frame)

        assert unbalanced == 0

    def test_a_break_in_the_balance_is_flagged_not_repaired(self) -> None:
        frame = _frame([{"debit_amount": 1_000.0, "credit_amount": 900.0}])
        result, unbalanced = flag_unbalanced(frame)

        assert unbalanced == 1
        assert result["unbalanced_flag"].iloc[0]
        assert result["credit_amount"].iloc[0] == 900.0, "the value must not be silently corrected"

    def test_rounding_noise_below_one_cent_is_tolerated(self) -> None:
        frame = _frame([{"debit_amount": 1_000.0, "credit_amount": 1_000.004}])
        _, unbalanced = flag_unbalanced(frame)

        assert unbalanced == 0


# --------------------------------------------------------------------------- #
# Quality report
# --------------------------------------------------------------------------- #
class TestDataQualityReport:
    """The report is a deliverable in its own right."""

    def test_total_issues_sums_the_individual_categories(self) -> None:
        report = DataQualityReport(
            raw_rows=1_000,
            duplicates_removed=10,
            invalid_date_rows=5,
            unbalanced_rows=3,
            orphan_vendor_rows=2,
            account_name_mismatches=7,
            currency_normalised_rows=4,
        )

        assert report.total_issues == 31

    def test_issue_rate_is_measured_against_the_raw_population(self) -> None:
        report = DataQualityReport(raw_rows=1_000, duplicates_removed=50)
        payload = report.to_dict()

        assert payload["issue_rate_pct"] == pytest.approx(5.0)

    def test_a_zero_row_input_does_not_divide_by_zero(self) -> None:
        report = DataQualityReport(raw_rows=0)
        payload = report.to_dict()

        assert payload["issue_rate_pct"] == pytest.approx(0.0)

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        payload = DataQualityReport(raw_rows=10, clean_rows=9).to_dict()
        assert json.dumps(payload, default=str)

    def test_defaults_are_all_zero(self) -> None:
        report = DataQualityReport(raw_rows=100, clean_rows=100)
        assert report.total_issues == 0


class TestVoucherCreatorCount:
    """The population figures must be named for what they actually measure.

    The report used to expose ``unique_employees``, computed from ``created_by``.
    That name collided with the ``employees`` master table, which is a *different*
    and larger population: roles such as HR, sales and the executive team never
    raise an AP voucher, so the roster is always bigger than the set of people who
    post one. Read side by side in the same report set, the two numbers looked like
    a contradiction rather than two legitimately different populations.

    A reviewer who spots a figure that appears to contradict itself stops trusting
    the rest of the numbers, so the count is pinned to its real definition here.
    """

    @staticmethod
    def _one_creator_two_employees() -> tuple[pd.DataFrame, pd.DataFrame]:
        """Two vouchers naming two employees, but only one of them raises any."""
        rows = [
            {"transaction_id": "TX0001", "employee_id": "E0001", "created_by": "E0001"},
            {"transaction_id": "TX0002", "employee_id": "E0002", "created_by": "E0001"},
        ]
        vendors = pd.DataFrame({"vendor_id": ["V0001"], "vendor_name": ["Test Vendor One"]})
        return _frame(rows), vendors

    def test_count_follows_created_by_not_employee_id(self) -> None:
        """The narrower population is the intended one, and the name must say so."""
        transactions, vendors = self._one_creator_two_employees()
        clean, report = clean_transactions(transactions, vendors)

        assert clean["employee_id"].nunique() == 2, "fixture must reference two employees"
        assert clean["created_by"].nunique() == 1, "fixture must have one creator"
        assert report.unique_voucher_creators == 1

    def test_the_ambiguous_field_name_cannot_come_back(self) -> None:
        """``unique_employees`` is the name that caused the misreading."""
        report = DataQualityReport()

        assert not hasattr(report, "unique_employees")
        assert "unique_employees" not in report.to_dict()
        assert "unique_voucher_creators" in report.to_dict()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
class TestCleanTransactions:
    """The full stage, on a deliberately dirty extract."""

    @staticmethod
    def _dirty() -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = [
            # Clean.
            {"transaction_id": "TX0001", "invoice_id": "INV1", "vendor_id": "V0001",
             "currency": "CNY", "debit_amount": 10_000.0, "credit_amount": 10_000.0},
            # Duplicate of the first row.
            {"transaction_id": "TX0001", "invoice_id": "INV1", "vendor_id": "V0001",
             "currency": "CNY", "debit_amount": 10_000.0, "credit_amount": 10_000.0},
            # Lowercase currency alias.
            {"transaction_id": "TX0003", "invoice_id": "INV3", "vendor_id": "V0001",
             "currency": "rmb", "debit_amount": 20_000.0, "credit_amount": 20_000.0},
            # Broken account name.
            {"transaction_id": "TX0004", "invoice_id": "INV4", "vendor_id": "V0001",
             "account_code": "6401", "account_name": "WRONG", "currency": "CNY",
             "debit_amount": 30_000.0, "credit_amount": 30_000.0},
            # Vendor missing from the master file.
            {"transaction_id": "TX0005", "invoice_id": "INV5", "vendor_id": "V-GHOST",
             "currency": "CNY", "debit_amount": 40_000.0, "credit_amount": 40_000.0},
            # Unbalanced.
            {"transaction_id": "TX0006", "invoice_id": "INV6", "vendor_id": "V0001",
             "currency": "CNY", "debit_amount": 50_000.0, "credit_amount": 45_000.0},
            # Unparseable transaction date.
            {"transaction_id": "TX0007", "invoice_id": "INV7", "vendor_id": "V0001",
             "transaction_date": "31/31/2025", "currency": "CNY",
             "debit_amount": 60_000.0, "credit_amount": 60_000.0},
        ]
        vendors = pd.DataFrame({"vendor_id": ["V0001"], "vendor_name": ["Test Vendor One"]})
        return _frame(rows), vendors

    def test_every_issue_category_is_counted(self) -> None:
        transactions, vendors = self._dirty()
        clean, report = clean_transactions(transactions, vendors)

        assert report.raw_rows == 7
        assert report.duplicates_removed == 1
        assert report.currency_normalised_rows >= 1
        assert report.account_name_mismatches == 1
        assert report.orphan_vendor_rows == 1
        assert report.unbalanced_rows == 1
        assert report.invalid_date_rows == 1

    def test_rows_with_no_usable_date_are_dropped(self) -> None:
        """A voucher that cannot be placed in a period cannot be audited."""
        transactions, vendors = self._dirty()
        clean, _ = clean_transactions(transactions, vendors)

        assert "TX0007" not in set(clean["transaction_id"])

    def test_clean_rows_equals_raw_minus_duplicates_minus_undated(self) -> None:
        transactions, vendors = self._dirty()
        clean, report = clean_transactions(transactions, vendors)

        assert report.clean_rows == len(clean)
        assert report.clean_rows == report.raw_rows - report.duplicates_removed - report.invalid_date_rows

    def test_the_cleaned_frame_is_typed_for_downstream_use(self) -> None:
        transactions, vendors = self._dirty()
        clean, _ = clean_transactions(transactions, vendors)

        assert pd.api.types.is_datetime64_any_dtype(clean["transaction_date"])
        assert pd.api.types.is_numeric_dtype(clean["debit_amount"])

    def test_flagged_rows_are_kept_so_the_auditor_can_chase_them(self) -> None:
        transactions, vendors = self._dirty()
        clean, _ = clean_transactions(transactions, vendors)

        assert "TX0005" in set(clean["transaction_id"]), "orphan vendor must survive cleaning"
        assert "TX0006" in set(clean["transaction_id"]), "unbalanced voucher must survive cleaning"

    def test_missing_narration_is_not_imputed(self) -> None:
        """Imputing a description would destroy the signal the rule engine needs."""
        transactions, vendors = self._dirty()
        transactions.loc[:, "description"] = pd.NA
        clean, _ = clean_transactions(transactions, vendors)

        assert clean["description"].isna().all()

    def test_cleaning_is_idempotent(self) -> None:
        """Running the cleaner twice must not keep changing the population."""
        transactions, vendors = self._dirty()
        once, _ = clean_transactions(transactions, vendors)
        twice, report = clean_transactions(once, vendors)

        assert len(twice) == len(once)
        assert report.duplicates_removed == 0

    def test_infinite_functional_amount_is_removed_and_reported(self) -> None:
        transactions = _frame([{}, {"amount": float("inf")}])
        vendors = pd.DataFrame({"vendor_id": ["V0001"]})
        clean, report = clean_transactions(transactions, vendors)

        assert len(clean) == 1
        assert report.invalid_amount_rows == 1
        assert clean["transaction_id"].tolist() == ["TX0001"]

    @pytest.mark.parametrize(
        ("original", "rate"),
        [(float("inf"), 1.0), (100.0, float("inf")), (1e308, 7.0)],
    )
    def test_nonfinite_translation_uses_finite_functional_amount(self, original, rate) -> None:
        transactions = _frame([{"amount_original": original, "fx_rate": rate}])
        vendors = pd.DataFrame({"vendor_id": ["V0001"]})
        clean, _ = clean_transactions(transactions, vendors)

        assert clean["amount_cny"].tolist() == [BASE_TRANSACTION["amount"]]


def _masters(vendor_rows: list[dict] | None = None, employee_rows: list[dict] | None = None):
    vendor = {
        "vendor_id": "V0001", "vendor_name": "Test Vendor", "bank_account": "BANK_A",
        "registration_date": "2024-01-01", "risk_level": "low", "is_dormant": False,
    }
    employee = {"employee_id": "E0001", "employee_name": "Test Employee", "role": "creator"}
    vendors = pd.DataFrame([{**vendor, **row} for row in (vendor_rows if vendor_rows is not None else [{}])])
    employees = pd.DataFrame([{**employee, **row} for row in (employee_rows if employee_rows is not None else [{}])])
    return vendors, employees


class TestMasterDataIntegrity:
    def test_identical_records_merge_without_mutating_sources(self) -> None:
        vendors, employees = _masters([{}, {}], [{}, {}])
        originals = vendors.copy(deep=True), employees.copy(deep=True)
        clean_vendors, clean_employees = clean_master_data(vendors, employees)
        assert len(clean_vendors) == len(clean_employees) == 1
        pd.testing.assert_frame_equal(vendors, originals[0])
        pd.testing.assert_frame_equal(employees, originals[1])

    def test_known_text_fields_use_existing_whitespace_normalisation(self) -> None:
        vendors, employees = _masters(
            [{}, {"vendor_id": " V0001 ", "vendor_name": " Test    Vendor ", "bank_account": " BANK_A "}],
            [{}, {"employee_id": " E0001 ", "role": " creator "}],
        )
        cleaned = clean_master_data(vendors, employees)
        assert [len(frame) for frame in cleaned] == [1, 1]

    def test_missing_ids_still_get_removed(self) -> None:
        vendors, employees = _masters(
            [{}, {"vendor_id": None}, {"vendor_id": "   "}],
            [{}, {"employee_id": None}, {"employee_id": "   "}],
        )
        cleaned = clean_master_data(vendors, employees)
        assert [len(frame) for frame in cleaned] == [1, 1]

    def test_missing_cells_are_equivalent_including_unlisted_columns(self) -> None:
        vendors, employees = _masters([
            {"bank_account": None, "registration_date": None, "source_note": None},
            {"bank_account": " ", "registration_date": pd.NA, "source_note": float("nan")},
        ])
        clean_vendors, _ = clean_master_data(vendors, employees)
        assert len(clean_vendors) == 1
        assert pd.isna(clean_vendors.loc[0, "bank_account"])
        assert pd.isna(clean_vendors.loc[0, "registration_date"])

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("field,value", [
        ("bank_account", "BANK_B"), ("risk_level", "high"),
        ("registration_date", "2024-02-01"), ("is_dormant", True),
        ("source_revision", "revision_b"),
    ])
    def test_conflicting_vendor_records_reject_in_either_order(self, field, value, reverse) -> None:
        vendors, employees = _masters([
            {"source_revision": "revision_a"}, {"source_revision": "revision_a", field: value},
        ])
        if reverse:
            vendors = vendors.iloc[::-1]
        with pytest.raises(ValueError, match="Conflicting master data") as error:
            clean_master_data(vendors, employees)
        message = str(error.value)
        assert "vendors_raw.csv" in message and "V0001" in message and field in message
        assert "BANK_A" not in message and "BANK_B" not in message
        assert "source rows 2, 3" in message

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("field,value", [("role", "approver"), ("can_approve", True)])
    def test_conflicting_employee_attributes_reject_in_either_order(self, field, value, reverse) -> None:
        vendors, employees = _masters(employee_rows=[
            {"can_approve": False}, {"can_approve": False, field: value},
        ])
        if reverse:
            employees = employees.iloc[::-1]
        with pytest.raises(ValueError, match="employees_raw.csv") as error:
            clean_master_data(vendors, employees)
        assert "E0001" in str(error.value) and field in str(error.value)

    def test_missing_and_populated_attributes_conflict_instead_of_filling(self) -> None:
        vendors, employees = _masters([{"bank_account": None}, {}])
        with pytest.raises(ValueError, match="bank_account"):
            clean_master_data(vendors, employees)

    @pytest.mark.parametrize("dates", [
        ("invalid_a", "invalid_b"), ("2024-01-01", "01/01/2024"), (None, "invalid_a"),
    ])
    def test_date_conversion_cannot_erase_source_differences(self, dates) -> None:
        vendors, employees = _masters([{"registration_date": date} for date in dates])
        with pytest.raises(ValueError, match="registration_date"):
            clean_master_data(vendors, employees)

    def test_source_row_positions_survive_missing_ids_and_duplicate_indexes(self) -> None:
        vendors, employees = _masters([{}, {"vendor_id": None}, {"bank_account": "BANK_B"}])
        vendors.index = [9, 9, 9]
        with pytest.raises(ValueError, match="source rows 2, 4"):
            clean_master_data(vendors, employees)


@pytest.fixture
def cleaning_outputs(monkeypatch, tmp_path):
    targets = [
        tmp_path / "processed" / "transactions.parquet",
        tmp_path / "processed" / "vendors.parquet",
        tmp_path / "processed" / "employees.parquet",
        tmp_path / "reports" / "quality.json",
    ]
    for name, path in zip(
        ("TRANSACTIONS_CLEAN", "VENDORS_CLEAN", "EMPLOYEES_CLEAN", "DATA_QUALITY_REPORT"), targets,
    ):
        monkeypatch.setattr(cleaning, name, path)
    vendors, employees = _masters()
    monkeypatch.setattr(cleaning, "load_raw_data", lambda: (_frame([{}]), vendors, employees))
    monkeypatch.setattr(cleaning, "ensure_directories", lambda: None)
    return targets


def _seed_outputs(targets: list[Path]) -> dict[Path, bytes]:
    previous = {path: f"previous complete result {index}".encode() for index, path in enumerate(targets)}
    for path, payload in previous.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return previous


def _assert_no_temporary_files(targets: list[Path]) -> None:
    for parent in {path.parent for path in targets}:
        assert not list(parent.glob(".auditlens-clean-*"))


class TestCleaningPublication:
    def test_normalised_master_ids_are_used_for_orphan_detection(self, cleaning_outputs, monkeypatch) -> None:
        vendors, employees = _masters([{"vendor_id": " V0001 "}])
        monkeypatch.setattr(cleaning, "load_raw_data", lambda: (_frame([{}]), vendors, employees))
        result = cleaning.run_cleaning()
        assert result["report"].orphan_vendor_rows == 0
        assert not result["transactions"]["orphan_vendor_flag"].any()

    def test_success_publishes_matching_tables_and_report(self, cleaning_outputs) -> None:
        _seed_outputs(cleaning_outputs)
        result = cleaning.run_cleaning()
        for key, path in zip(("transactions", "vendors", "employees"), cleaning_outputs):
            pd.testing.assert_frame_equal(pd.read_parquet(path), result[key])
        assert json.loads(cleaning_outputs[3].read_text("utf-8")) == result["report"].to_dict()
        _assert_no_temporary_files(cleaning_outputs)

    def test_master_conflict_preserves_all_previous_evidence(self, cleaning_outputs, monkeypatch) -> None:
        previous = _seed_outputs(cleaning_outputs)
        vendors, employees = _masters([{}, {"bank_account": "BANK_B"}])
        monkeypatch.setattr(cleaning, "load_raw_data", lambda: (_frame([{}]), vendors, employees))
        with pytest.raises(ValueError, match="bank_account"):
            cleaning.run_cleaning()
        assert {path: path.read_bytes() for path in cleaning_outputs} == previous
        _assert_no_temporary_files(cleaning_outputs)

    @pytest.mark.parametrize("failed_stage", ["employees", "report"])
    def test_serialisation_failure_preserves_all_previous_outputs(
        self, cleaning_outputs, monkeypatch, failed_stage,
    ) -> None:
        previous = _seed_outputs(cleaning_outputs)
        if failed_stage == "employees":
            original = pd.DataFrame.to_parquet
            calls = 0

            def fail_third(frame, path, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    Path(path).write_bytes(b"partial serialisation")
                    raise OSError("serialisation failed")
                return original(frame, path, **kwargs)

            monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_third)
        else:
            def fail_report(payload, path):
                path.write_bytes(b"partial report")
                raise OSError("serialisation failed")

            monkeypatch.setattr(cleaning, "save_json", fail_report)
        with pytest.raises(OSError, match="serialisation failed"):
            cleaning.run_cleaning()
        assert {path: path.read_bytes() for path in cleaning_outputs} == previous
        _assert_no_temporary_files(cleaning_outputs)

    @pytest.mark.parametrize("has_previous", [False, True])
    @pytest.mark.parametrize("failed_replace", [1, 2, 4])
    def test_publication_failure_rolls_back_complete_result(
        self, cleaning_outputs, monkeypatch, has_previous, failed_replace,
    ) -> None:
        previous = _seed_outputs(cleaning_outputs) if has_previous else {}
        original = cleaning.os.replace
        calls = 0

        def fail_once(source, target):
            nonlocal calls
            calls += 1
            if calls == failed_replace:
                raise OSError("publication failed")
            return original(source, target)

        monkeypatch.setattr(cleaning.os, "replace", fail_once)
        with pytest.raises(OSError, match="publication failed"):
            cleaning.run_cleaning()
        assert {path: path.read_bytes() for path in cleaning_outputs if path.exists()} == previous
        _assert_no_temporary_files(cleaning_outputs)

    def test_backup_failure_never_publishes_any_output(self, cleaning_outputs, monkeypatch) -> None:
        previous = _seed_outputs(cleaning_outputs)

        def fail_copy(source, target):
            raise OSError("backup failed")

        monkeypatch.setattr(cleaning.shutil, "copyfile", fail_copy)
        with pytest.raises(OSError, match="backup failed"):
            cleaning.run_cleaning()
        assert {path: path.read_bytes() for path in cleaning_outputs} == previous
        _assert_no_temporary_files(cleaning_outputs)

    def test_failed_rollback_retains_old_bytes_and_reports_recovery_path(
        self, cleaning_outputs, monkeypatch,
    ) -> None:
        previous = _seed_outputs(cleaning_outputs)
        original = cleaning.os.replace
        calls = 0

        def fail_publication_and_rollback(source, target):
            nonlocal calls
            calls += 1
            if calls in (2, 3):
                raise OSError("blocked replacement")
            return original(source, target)

        monkeypatch.setattr(cleaning.os, "replace", fail_publication_and_rollback)
        with pytest.raises(RuntimeError, match="rollback was incomplete") as error:
            cleaning.run_cleaning()
        backups = list(cleaning_outputs[0].parent.glob(".auditlens-clean-*"))
        assert len(backups) == 1 and backups[0].read_bytes() == previous[cleaning_outputs[0]]
        assert str(backups[0]) in str(error.value)
        assert all(path.read_bytes() == previous[path] for path in cleaning_outputs[1:])
