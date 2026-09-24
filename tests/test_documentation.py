"""The README's testing table, kept honest.

The test count is restated in six places in the README plus an eleven-row table, and it
has drifted repeatedly during development (302 -> 331 -> 351). Every drift was a
document that was correct when written and silently wrong afterwards, which is the same
failure mode as the SQL header that said "Fourteen queries" while the file held fifteen.

These checks are pure text comparisons: no pytest introspection, so they cannot break
spuriously as tests are added or parametrised. They only fail when a document and the
repository genuinely disagree.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
TESTS_DIR = PROJECT_ROOT / "tests"

#: ``| `test_foo.py` | 12 | what it pins down |``
TABLE_ROW = re.compile(r"^\|\s*`(test_\w+)\.py`\s*\|\s*(\d+)\s*\|", re.MULTILINE)


def _table_rows() -> dict[str, int]:
    return {name: int(count) for name, count in TABLE_ROW.findall(README.read_text("utf-8"))}


def _test_files_on_disk() -> set[str]:
    return {path.stem for path in TESTS_DIR.glob("test_*.py")}


def _stated_total() -> int:
    """The headline test count, read from the badge."""
    match = re.search(r"Tests:\s*(\d+)", README.read_text("utf-8"))
    assert match, "the README badge no longer states a test count"
    return int(match.group(1))


class TestTestingTable:
    """The table lists every test file, and its numbers add up."""

    def test_the_table_is_present(self) -> None:
        assert _table_rows(), "the README no longer has a testing table"

    def test_every_test_file_has_a_row(self) -> None:
        missing = sorted(_test_files_on_disk() - set(_table_rows()))
        assert not missing, f"these test files are not described in the README: {missing}"

    def test_every_row_refers_to_a_real_file(self) -> None:
        """A row for a deleted file is as misleading as a missing row."""
        ghosts = sorted(set(_table_rows()) - _test_files_on_disk())
        assert not ghosts, f"the README describes test files that do not exist: {ghosts}"

    def test_every_row_states_a_positive_count(self) -> None:
        empty = sorted(name for name, count in _table_rows().items() if count <= 0)
        assert not empty, f"rows claiming no tests: {empty}"

    def test_the_badge_matches_the_sum_of_the_rows(self) -> None:
        """The one number a reader trusts must not contradict the table beneath it."""
        stated, summed = _stated_total(), sum(_table_rows().values())
        assert stated == summed, (
            f"the badge says {stated} tests but the table sums to {summed}; "
            "update whichever is stale"
        )
