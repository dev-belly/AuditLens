"""The README's structural claims, kept honest.

Two kinds of claim live here, both of the same shape: a fact restated in the README
that the repository can contradict without anyone noticing.

**The test count.** It appears in seven places in the README plus a thirteen-row
table, and it has drifted repeatedly during development (302 -> 331 -> 351 -> 374 ->
378). Every drift was a document that was correct when written and silently wrong
afterwards, which is the same failure mode as the SQL header that said "Fourteen
queries" while the file held fifteen.

The first version of this file compared the table only against the badge - a pure text
check, chosen so that adding a test could not break it. That choice is why the drift it
was written to prevent happened again: adding four tests left the table claiming 55 and
18 for files that now held 57 and 20, and every guard stayed green, because the table and
the badge agreed with each other and neither was compared to reality.

**The project-structure tree.** Every module the README lists must exist, and every
module in ``src/`` must be listed. A rename or an extraction leaves a ghost in the tree
otherwise, and the tree is the first thing a reviewer reads to understand the layout.

So the checks come in two kinds, and both are needed:

* **Text checks** - the table lists every test file, every row names a real file, and the
  rows sum to the badge. Cheap, and they cannot break spuriously.
* **Introspection checks** - the rows and the badge are compared against what pytest
  actually collects, and the tree against what is on disk. Without these the whole file
  only proves the README agrees with itself.

The test-count introspection runs ``pytest --collect-only`` in a subprocess rather than
reading ``request.session.items``. A session describes only the tests the current
invocation collected, so running this file on its own would compare the table against a
single file and fail for a reason that has nothing to do with the documentation. A fresh
collection over ``tests/`` always describes the whole suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
TESTS_DIR = PROJECT_ROOT / "tests"

#: ``| `test_foo.py` | 12 | what it pins down |``
TABLE_ROW = re.compile(r"^\|\s*`(test_\w+)\.py`\s*\|\s*(\d+)\s*\|", re.MULTILINE)

#: ``├── utils.py`` / ``└── ci_summary.py`` inside the README's structure tree.
TREE_FILE = re.compile(r"[├└]──\s+(\S+\.py)")

#: Directories the structure tree describes, searched in this order.
TREE_DIRS = ("src", "dashboard", "tools", "tests")

#: ``tests/test_foo.py: 12`` - pytest's per-file summary under ``--collect-only -q``.
#: Matched on the stem rather than the full path, because pytest echoes back whatever
#: path form it was handed.
COLLECTED_ROW = re.compile(r"^(\S*test_\w+\.py):\s*(\d+)\s*$", re.MULTILINE)


def _table_rows() -> dict[str, int]:
    return {name: int(count) for name, count in TABLE_ROW.findall(README.read_text("utf-8"))}


def _test_files_on_disk() -> set[str]:
    return {path.stem for path in TESTS_DIR.glob("test_*.py")}


def _tree_files() -> set[str]:
    """Every ``.py`` filename the README's project-structure tree lists."""
    return set(TREE_FILE.findall(README.read_text("utf-8")))


def _stated_total() -> int:
    """The headline test count, read from the badge."""
    match = re.search(r"Tests:\s*(\d+)", README.read_text("utf-8"))
    assert match, "the README badge no longer states a test count"
    return int(match.group(1))


@lru_cache(maxsize=1)
def _collected_counts() -> dict[str, int]:
    """How many tests each file really contains, straight from pytest.

    Collection only - no test body runs - so this is safe to call from inside a test.
    Cached because two tests need it and a collection pass is not free.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "pytest --collect-only failed, so the documented counts cannot be checked:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    counts = {Path(path).stem: int(count) for path, count in COLLECTED_ROW.findall(result.stdout)}
    if not counts:
        raise AssertionError(
            "could not parse any per-file counts out of `pytest --collect-only -q`; "
            "the output format may have changed:\n" + result.stdout
        )
    return counts


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


class TestTheDocumentedCountsAreTrue:
    """The table and badge must describe the suite that exists, not a previous one.

    The text checks above prove the README agrees with itself. These prove it agrees
    with pytest. Without them a new test can be added and the documentation silently
    becomes a lie while everything stays green - which is what happened at 374 -> 378.
    """

    def test_every_row_matches_the_collected_count(self) -> None:
        collected = _collected_counts()
        stale = [
            f"`{name}.py`: README says {stated}, pytest collects {collected[name]}"
            for name, stated in sorted(_table_rows().items())
            if name in collected and collected[name] != stated
        ]
        assert not stale, "the README testing table is out of date: " + "; ".join(stale)

    def test_the_badge_matches_what_pytest_collects(self) -> None:
        collected = _collected_counts()
        actual = sum(collected.values())
        assert _stated_total() == actual, (
            f"the badge says {_stated_total()} tests but pytest collects {actual}; "
            "the README is stale"
        )

    def test_the_collected_counts_cover_every_test_file(self) -> None:
        """Guards the parser: a renamed file must not silently drop out of the check."""
        missing = sorted(_test_files_on_disk() - set(_collected_counts()))
        assert not missing, (
            "pytest collected nothing from these files, so their counts are unverified: "
            f"{missing}"
        )


class TestProjectStructure:
    """The structure tree must describe the repository that exists.

    The tree is how a reviewer works out where anything lives, so a ghost entry or a
    missing module is worse than an out-of-date number: it misdirects someone who has
    no other map. ``src/anomaly_injection.py`` was extracted from ``data_generator.py``
    and had to be added here; without a guard the next extraction would not be.
    """

    def test_the_tree_lists_some_files(self) -> None:
        """Guards the parser: an empty match would make the other two vacuous."""
        assert _tree_files(), "could not parse any filenames out of the README tree"

    def test_every_file_in_the_tree_exists(self) -> None:
        """A renamed or deleted module must not leave a ghost in the tree."""
        ghosts = sorted(
            name
            for name in _tree_files()
            if not any((PROJECT_ROOT / directory / name).exists() for directory in TREE_DIRS)
        )
        assert not ghosts, f"the README structure tree names files that do not exist: {ghosts}"

    def test_every_module_appears_in_the_tree(self) -> None:
        """A new module must be documented, not merely committed."""
        documented = _tree_files()
        missing: list[str] = []
        for directory in ("src", "tools", "dashboard"):
            missing += sorted(
                path.name
                for path in (PROJECT_ROOT / directory).glob("*.py")
                if path.name != "__init__.py" and path.name not in documented
            )
        assert not missing, f"these modules are missing from the README tree: {missing}"
