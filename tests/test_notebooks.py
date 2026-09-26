"""The committed notebooks, and the tool that builds them.

A notebook is the one artefact in this project that can be *silently* empty. The code
runs, the build reports success, and the ``.ipynb`` lands on GitHub with nothing under
each cell - which reads as unfinished work to the reviewer it was written for. Nothing
in the pipeline notices, because a notebook with no outputs is a perfectly valid file.

Two failures are guarded here.

The first is emptiness: every cell that prints, displays or plots must carry output.
The second is *instability*. Two things made the committed notebooks differ on every
build even when every number matched - pandas stamps each Styler render with a fresh
``uuid4`` table id, and the pipeline's loggers write a wall-clock timestamp to whatever
``sys.stdout`` was live when they were first built. Both are normalised by the builder,
and the tests below fail if either ever leaks back in.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_notebooks import (  # noqa: E402  (needs sys.path above)
    NOTEBOOK_DIR,
    _reset_table_ids,
    _split_trailing_expression,
    _stable_table_ids,
    execute,
)

#: ``T_`` followed by five hex digits is what pandas' random uuid produces. A stable id
#: is all digits, so this pattern matches only the random ones.
RANDOM_TABLE_ID = re.compile(r"T_[0-9a-f]{5}")

#: A logger record as ``src.utils.get_logger`` formats it, timestamp first.
LOG_RECORD = re.compile(r"\d{2}:\d{2}:\d{2} \| (?:INFO|WARNING|ERROR)")

#: A cell is expected to show something if it prints, displays, or draws.
PRODUCES_OUTPUT = re.compile(r"\b(?:print|display)\s*\(|\bplt\.")


def _load(name: str) -> dict[str, Any]:
    """Read one committed notebook."""
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _output_text(notebook: dict[str, Any]) -> str:
    """Every stream and text payload in a notebook, concatenated."""
    chunks: list[str] = []
    for cell in _code_cells(notebook):
        for output in cell.get("outputs") or []:
            text = output.get("text")
            if text is not None:
                chunks.append("".join(text) if isinstance(text, list) else str(text))
            for value in (output.get("data") or {}).values():
                if isinstance(value, str):
                    chunks.append(value)
                elif isinstance(value, list):
                    chunks.append("".join(str(part) for part in value))
    return "\n".join(chunks)


NOTEBOOK_NAMES = tuple(sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")))


class TestStableTableIds:
    """pandas' random Styler ids are rewritten to a deterministic sequence."""

    def test_a_random_id_is_replaced(self) -> None:
        assert _stable_table_ids('<table id="T_3b175">') == '<table id="T_z0000">'

    def test_every_reference_to_one_table_collapses_to_one_id(self) -> None:
        """A Styler repeats its id on the table and on every cell inside it."""
        _reset_table_ids()
        html = '<table id="T_3b175"><th id="T_3b175_level0_col0">x</th></table>'
        rendered = _stable_table_ids(html)
        assert "T_3b175" not in rendered
        assert rendered.count("T_z0000") == 2

    def test_two_tables_get_two_ids(self) -> None:
        """Distinct tables must not collide, so the sequence advances per table."""
        _reset_table_ids()
        assert _stable_table_ids('id="T_aaaaa"') == 'id="T_z0000"'
        assert _stable_table_ids('id="T_bbbbb"') == 'id="T_z0001"'

    def test_the_replacement_cannot_be_mistaken_for_a_random_id(self) -> None:
        """The whole point of the ``z``: a stable id must not match the random pattern.

        An all-digit replacement would match :data:`RANDOM_TABLE_ID` too, so the
        leak check in :class:`TestCommittedNotebooks` could never fail.
        """
        _reset_table_ids()
        assert RANDOM_TABLE_ID.search(_stable_table_ids('id="T_3b175"')) is None

    def test_html_without_a_table_id_is_untouched(self) -> None:
        assert _stable_table_ids("<table><tr><td>1</td></tr></table>") == (
            "<table><tr><td>1</td></tr></table>"
        )


class TestSplitTrailingExpression:
    """A cell's final expression is peeled off so it can be evaluated and shown."""

    def test_a_trailing_expression_is_separated_from_its_statements(self) -> None:
        statements, trailing = _split_trailing_expression("x = 1\nx + 1")
        assert statements.strip() == "x = 1"
        assert trailing == "x + 1"

    def test_a_cell_ending_in_an_assignment_has_no_trailing_expression(self) -> None:
        statements, trailing = _split_trailing_expression("x = 1\ny = x + 1")
        assert trailing is None
        assert "y = x + 1" in statements

    def test_a_call_statement_is_peeled_and_shows_nothing(self) -> None:
        """``print(x)`` is an expression, so it is peeled - and evaluates to ``None``.

        The builder shows nothing for it, matching IPython, where the text arrives on
        stdout rather than in an ``Out[]``. That is why peeling is harmless here.
        """
        statements, trailing = _split_trailing_expression("x = 1\nprint(x)")
        assert statements.strip() == "x = 1"
        assert trailing == "print(x)"

    def test_an_expression_only_cell_yields_empty_statements(self) -> None:
        statements, trailing = _split_trailing_expression("42")
        assert statements == ""
        assert trailing == "42"

    def test_a_syntax_error_leaves_the_body_alone(self) -> None:
        """The cell is still executed, so its real error is the one that surfaces."""
        body = "def broken(:\n    pass"
        statements, trailing = _split_trailing_expression(body)
        assert statements == body
        assert trailing is None


class TestDisplayShim:
    """``display()`` records the object it is handed, not its contents.

    The shim was originally ``displayed.extend``, which takes an *iterable* of things to
    display. ``display(styler)`` raised, and ``display(frame)`` recorded the frame's
    column names - a wrong table, produced without an error. Both are pinned here.
    """

    def test_a_dataframe_is_displayed_as_a_table(self) -> None:
        outputs = execute(
            [("code", "import pandas as pd"), ("code", "display(pd.DataFrame({'amount': [1, 2]}))")],
            "probe.ipynb",
        )
        assert len(outputs[1]) == 1
        html = outputs[1][0]["data"]["text/html"]
        assert "<table" in html
        assert "amount" in html

    def test_a_styler_does_not_raise(self) -> None:
        outputs = execute(
            [
                ("code", "import pandas as pd"),
                ("code", "display(pd.DataFrame({'amount': [1, 2]}).style.format('{:,.0f}'))"),
            ],
            "probe.ipynb",
        )
        assert "<table" in outputs[1][0]["data"]["text/html"]

    def test_several_values_in_one_display_are_all_recorded(self) -> None:
        outputs = execute(
            [("code", "import pandas as pd"), ("code", "display(pd.Series([1]), pd.Series([2]))")],
            "probe.ipynb",
        )
        assert len(outputs[1]) == 2


class TestCommittedNotebooks:
    """The notebooks on disk are populated, stable and structurally valid."""

    def test_there_are_notebooks_to_check(self) -> None:
        assert len(NOTEBOOK_NAMES) == 3, f"expected three notebooks, found {NOTEBOOK_NAMES}"

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_notebook_is_valid_nbformat_4(self, name: str) -> None:
        notebook = _load(name)
        assert notebook["nbformat"] == 4
        assert notebook["cells"], f"{name} has no cells"
        assert notebook["metadata"]["language_info"]["name"] == "python"

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_every_cell_that_shows_something_has_output(self, name: str) -> None:
        """The guard against a notebook committed with empty output cells."""
        empty: list[int] = []
        for index, cell in enumerate(_code_cells(_load(name))):
            source = "".join(cell["source"])
            if PRODUCES_OUTPUT.search(source) and not cell.get("outputs"):
                empty.append(index)
        assert not empty, f"{name}: code cells {empty} print or plot but produced no output"

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_each_notebook_embeds_a_chart(self, name: str) -> None:
        """Headless ``plt.show()`` discards the figure unless the builder captures it."""
        images = [
            output
            for cell in _code_cells(_load(name))
            for output in (cell.get("outputs") or [])
            if "image/png" in (output.get("data") or {})
        ]
        assert images, f"{name} embeds no chart"

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_no_random_styler_id_survives(self, name: str) -> None:
        leaked = sorted(set(RANDOM_TABLE_ID.findall(_output_text(_load(name)))))
        assert not leaked, f"{name} still carries random table ids: {leaked[:5]}"

    @pytest.mark.parametrize("name", NOTEBOOK_NAMES)
    def test_no_wall_clock_timestamp_survives(self, name: str) -> None:
        """A timestamp makes the committed file differ on every build."""
        leaked = sorted(set(LOG_RECORD.findall(_output_text(_load(name)))))
        assert not leaked, f"{name} still carries logged timestamps: {leaked[:5]}"
