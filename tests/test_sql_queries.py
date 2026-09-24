"""The SQL analyst library.

``sql/audit_queries.sql`` is a deliverable in its own right - the queries an audit team
would actually run against the warehouse. It was the one part of the project with no test
at all, which matters because :func:`src.database.run_sql_file` **swallows failures**: a
query that references a dropped column logs an error and returns an empty dataframe, so
the pipeline still reports "Executed N SQL queries" and finishes green. A silently broken
query would leave a hole in the library and nothing would say so.

The tests below therefore execute every query for real, and pin the count that the
documentation claims. That last guard exists because the count had already drifted: the
file's own header said "Fourteen queries" while it contained fifteen, and nothing
noticed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (  # noqa: E402  (needs sys.path above)
    DB_PATH,
    GROUND_TRUTH_COLUMNS,
    SQL_DIR,
    split_sql_statements,
)

QUERY_FILE = SQL_DIR / "audit_queries.sql"

#: The brief asks for 8-10 analyst queries. The library exceeds that; the floor is
#: what the test protects.
MINIMUM_QUERIES = 8

#: Words used for small counts in prose, so a stale "Fourteen" can be caught.
NUMBER_WORDS: dict[str, int] = {
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _script() -> str:
    return QUERY_FILE.read_text(encoding="utf-8")


def _statements() -> list[tuple[str, str]]:
    return split_sql_statements(_script())


def _documented_counts() -> dict[str, int]:
    """Every query count the documentation asserts, as ``{where: count}``.

    Parsing the docs is unusual, but the alternative is what already happened: a
    hand-maintained count in five places with nothing tying them to the file.
    """
    found: dict[str, int] = {}

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"\*\*(\d+)\s*\n?\s*named business queries", readme):
        found[f"README.md (**{match.group(1)} named business queries)"] = int(match.group(1))
    for match in re.finditer(r"#\s*(\d+)\s+named business queries", readme):
        found[f"README.md (tree: {match.group(1)})"] = int(match.group(1))

    architecture = (PROJECT_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for match in re.finditer(r"\((\d+)\s+named analyst queries\)", architecture):
        found[f"architecture.md ({match.group(1)} named analyst queries)"] = int(match.group(1))
    for match in re.finditer(r"\b(\d+)\s+analyst queries\b", architecture):
        found[f"architecture.md ({match.group(1)} analyst queries)"] = int(match.group(1))
    for match in re.finditer(r"\b(\d+)\s+named queries\b", architecture):
        found[f"architecture.md ({match.group(1)} named queries)"] = int(match.group(1))

    # The SQL file describes itself in words.
    header = re.search(r"^--\s+(\w+)\s+queries an audit team", _script(), re.MULTILINE)
    if header:
        word = header.group(1).lower()
        if word in NUMBER_WORDS:
            found[f"audit_queries.sql (\"{header.group(1)} queries\")"] = NUMBER_WORDS[word]

    return found


class TestSplitSqlStatements:
    """The ``-- name:`` convention that labels each query."""

    def test_splits_two_named_statements(self) -> None:
        script = "-- name: first\nSELECT 1;\n-- name: second\nSELECT 2;"
        assert split_sql_statements(script) == [("first", "SELECT 1;"), ("second", "SELECT 2;")]

    def test_ignores_comment_and_blank_lines(self) -> None:
        script = "-- a header\n\n-- name: q\n-- why it matters\nSELECT 1;\n"
        assert split_sql_statements(script) == [("q", "SELECT 1;")]

    def test_strips_whitespace_around_the_name(self) -> None:
        assert split_sql_statements("-- name:   spaced   \nSELECT 1;") == [("spaced", "SELECT 1;")]

    def test_a_script_with_no_markers_yields_nothing(self) -> None:
        assert split_sql_statements("SELECT 1;\n-- just a comment") == []

    def test_a_statement_is_kept_without_a_following_marker(self) -> None:
        """The last query in a file has nothing to terminate it."""
        assert split_sql_statements("-- name: only\nSELECT 1;\nSELECT 2;") == [
            ("only", "SELECT 1;\nSELECT 2;")
        ]

    def test_a_marker_with_no_statement_is_dropped(self) -> None:
        """A dangling ``-- name:`` would otherwise register an empty query."""
        assert split_sql_statements("-- name: empty\n-- name: real\nSELECT 1;") == [
            ("real", "SELECT 1;")
        ]


class TestQueryLibrary:
    """What the library contains, checked without a database."""

    def test_the_file_exists(self) -> None:
        assert QUERY_FILE.exists(), f"missing {QUERY_FILE}"

    def test_declares_at_least_the_required_number_of_queries(self) -> None:
        assert len(_statements()) >= MINIMUM_QUERIES

    def test_query_names_are_unique(self) -> None:
        names = [name for name, _ in _statements()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert not duplicates, f"duplicate query names: {duplicates}"

    def test_query_names_are_identifiers(self) -> None:
        """The name is used as a dict key and printed in the pipeline log."""
        bad = [name for name, _ in _statements() if not re.fullmatch(r"[a-z][a-z0-9_]*", name)]
        assert not bad, f"query names should be snake_case identifiers: {bad}"

    def test_every_statement_is_read_only(self) -> None:
        """These are auditor-facing queries. Nothing here may write."""
        offenders = [
            name
            for name, statement in _statements()
            if not statement.lstrip().upper().startswith(("SELECT", "WITH"))
        ]
        assert not offenders, f"non-SELECT statements: {offenders}"

    def test_no_statement_mutates_the_warehouse(self) -> None:
        forbidden = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE)\b", re.I)
        offenders = [name for name, statement in _statements() if forbidden.search(statement)]
        assert not offenders, f"statements that could modify the database: {offenders}"

    def test_no_query_references_a_ground_truth_column(self) -> None:
        """The warehouse carries the labels; an auditor-facing query must not read them.

        This is the same guarantee the dashboard tests enforce, applied to SQL. The
        labels travel in the parquet so the model evaluation can be reproduced - they
        are not there to be handed to the person doing the audit.
        """
        offenders: dict[str, list[str]] = {}
        for name, statement in _statements():
            hits = [column for column in GROUND_TRUTH_COLUMNS if column in statement]
            if hits:
                offenders[name] = hits
        assert not offenders, f"queries reading ground truth: {offenders}"

    def test_every_query_carries_an_explanatory_comment(self) -> None:
        """A query library with no rationale is a list of SQL, not a deliverable."""
        lines = _script().splitlines()
        markers = [i for i, line in enumerate(lines) if line.strip().lower().startswith("-- name:")]
        assert markers, "no queries found"
        for index in markers:
            following = lines[index + 1 : index + 3]
            assert any(line.strip().startswith("--") for line in following), (
                f"query at line {index + 1} has no explanatory comment"
            )


class TestDocumentedCounts:
    """The count in the prose matches the count in the file."""

    def test_the_documentation_states_a_count_at_all(self) -> None:
        assert _documented_counts(), "no query count is stated in the docs"

    def test_every_stated_count_matches_the_file(self) -> None:
        actual = len(_statements())
        wrong = {
            where: stated for where, stated in _documented_counts().items() if stated != actual
        }
        assert not wrong, f"the library holds {actual} queries, but these say otherwise: {wrong}"


@pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="Run `python src/run_pipeline.py` (or `make sql`) before the query tests.",
)
class TestQueriesExecuteAgainstTheWarehouse:
    """Every query runs, and returns rows.

    This is the guard against the swallowed failure: ``run_sql_file`` turns an
    exception into an empty frame, so without these tests a query naming a column
    that no longer exists would pass the whole suite.
    """

    @pytest.fixture(scope="class")
    def results(self) -> dict[str, object]:
        from src.database import run_sql_file

        return run_sql_file(QUERY_FILE)

    def test_every_query_was_executed(self, results: dict[str, object]) -> None:
        assert len(results) == len(_statements())

    def test_no_query_returned_an_empty_frame(self, results: dict[str, object]) -> None:
        empty = sorted(name for name, frame in results.items() if frame.empty)
        assert not empty, f"these queries returned nothing, so they probably failed: {empty}"

    def test_every_query_returns_at_least_one_column(self, results: dict[str, object]) -> None:
        shapeless = sorted(name for name, frame in results.items() if not len(frame.columns))
        assert not shapeless, f"queries with no columns: {shapeless}"

    def test_no_ground_truth_column_came_back(self, results: dict[str, object]) -> None:
        """Belt and braces: check the result sets, not just the SQL text."""
        leaked: dict[str, list[str]] = {}
        for name, frame in results.items():
            hits = [column for column in GROUND_TRUTH_COLUMNS if column in frame.columns]
            if hits:
                leaked[name] = hits
        assert not leaked, f"ground truth reached a query result: {leaked}"
