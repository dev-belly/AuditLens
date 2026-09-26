"""Build the three notebooks, execute every code cell, and capture the results.

Writing notebook JSON by hand is error-prone, and a notebook that does not run is
worse than no notebook. This script holds the cell sources as plain strings, runs
all of a notebook's code cells in one namespace (exactly as a kernel would), and
only writes the ``.ipynb`` once the whole notebook has executed cleanly.

It also captures what each cell *produced* - stdout, ``display()`` values and
matplotlib figures - and embeds that in the notebook. A committed notebook with empty
output cells is a portfolio liability: on GitHub it renders as code with nothing
underneath, which reads as unfinished. Capturing the output here means the notebooks
are never hand-edited and can never drift from the pipeline that produced them.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import itertools
import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

KERNELSPEC: dict[str, Any] = {
    "display_name": "Python 3 (AuditLens)",
    "language": "python",
    "name": "python3",
}

LANGUAGE_INFO: dict[str, Any] = {
    "name": "python",
    "version": "3.11",
    "mimetype": "text/x-python",
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "codemirror_mode": {"name": "ipython", "version": 3},
}


from tools.notebook_content import NOTEBOOKS


def to_source(text: str) -> list[str]:
    """Split a cell body into the line list nbformat expects."""
    lines = text.splitlines(keepends=True)
    return lines or [""]


#: Inline figure resolution. GitHub scales notebook images down to the content
#: column, so anything above ~110 dpi is bytes nobody sees.
FIGURE_DPI = 110


def _html_for(value: Any) -> str | None:
    """Render a value as an HTML table when it has a sensible one.

    pandas DataFrames, Series and Styler objects all expose ``to_html``. Returning it
    as ``text/html`` is what makes a table render as a table on GitHub rather than as
    a wall of monospaced repr.
    """
    to_html = getattr(value, "to_html", None)
    if not callable(to_html):
        return None
    try:
        return _stable_table_ids(to_html())
    except Exception:  # noqa: BLE001 - a Styler with an unsupported dtype, say
        return None


#: pandas stamps every ``Styler`` render with a fresh ``uuid4``-derived table id, which
#: appears on the ``<table>`` and on every ``<th>``/``<td>`` inside it. Two builds of the
#: same notebook therefore differ in hundreds of bytes even when every number matches.
_TABLE_ID = re.compile(r"T_[0-9a-f]{5}")

#: Serial used to mint the replacement ids. Reset per build so the sequence depends
#: only on cell order, which is fixed.
_TABLE_SERIAL = itertools.count()


def _reset_table_ids() -> None:
    """Restart the deterministic table-id sequence. Called once per build."""
    global _TABLE_SERIAL
    _TABLE_SERIAL = itertools.count()


def _stable_table_ids(html: str) -> str:
    """Rewrite pandas' random Styler table ids to deterministic ones.

    The id is only a hook for the stylesheet pandas embeds alongside the table, so
    substituting a stable value changes nothing a reader sees - but it is the difference
    between a committed notebook that diffs on every build and one that does not.

    Ids are memoised per call, so the many references to one table's id all collapse to
    the same replacement while a cell displaying two tables still gets two distinct ids.

    The replacement deliberately contains a ``z``. A stable id of ``T_00000`` would
    itself match :data:`_TABLE_ID` - all five characters are hex - which makes "did any
    random id leak into the committed notebook?" impossible to answer by inspection.
    """
    replacements: dict[str, str] = {}

    def swap(match: re.Match[str]) -> str:
        original = match.group(0)
        if original not in replacements:
            replacements[original] = f"T_z{next(_TABLE_SERIAL):04d}"
        return replacements[original]

    return _TABLE_ID.sub(swap, html)


def _value_output(value: Any, *, output_type: str, count: int | None = None) -> dict[str, Any]:
    """Wrap one Python value as a notebook ``display_data`` or ``execute_result``."""
    html = _html_for(value)
    data: dict[str, Any] = {}
    if html is not None:
        data["text/html"] = html
    else:
        try:
            data["text/plain"] = repr(value)
        except Exception:  # noqa: BLE001 - a repr that raises is still worth recording
            data["text/plain"] = f"<unrepresentable {type(value).__name__}>"

    output: dict[str, Any] = {"output_type": output_type, "metadata": {}, "data": data}
    if count is not None:
        output["execution_count"] = count
    return output


def _figure_outputs() -> list[dict[str, Any]]:
    """Collect any matplotlib figures the cell left open, as inline PNGs.

    The notebooks run headless, so ``plt.show()`` is a no-op and the figure would
    otherwise be discarded. Capturing it here is the difference between a committed
    notebook that shows its charts and one that shows the code that drew them.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional for a given notebook
        return []

    outputs: list[dict[str, Any]] = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", bbox_inches="tight", dpi=FIGURE_DPI)
        except Exception:  # noqa: BLE001 - a figure that will not render is not fatal
            plt.close(figure)
            continue

        width, height = figure.get_size_inches()
        outputs.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": [
                        f"<Figure size {width * FIGURE_DPI:.0f}x{height * FIGURE_DPI:.0f} "
                        f"with {len(figure.axes)} Axes>"
                    ],
                },
            }
        )
        plt.close(figure)
    return outputs


def _split_trailing_expression(body: str) -> tuple[str, str | None]:
    """Split a cell into its statements and a trailing expression, if it has one.

    A kernel displays the value of a cell's final expression; ``exec`` throws it away.
    Peeling the last expression off and evaluating it separately is what makes a cell
    ending in ``df.head()`` or a bare ``summary`` show a table instead of nothing.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return body, None

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return body, None

    trailing = tree.body.pop()
    statements = ast.unparse(ast.fix_missing_locations(tree)) if tree.body else ""
    return statements, ast.unparse(trailing.value)


def build_notebook(cells: list[tuple[str, str]], outputs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Assemble a notebook document, embedding the captured cell outputs."""
    count = 0
    rendered: list[dict[str, Any]] = []

    for index, (kind, body) in enumerate(cells):
        if kind != "code":
            rendered.append({"cell_type": "markdown", "metadata": {}, "source": to_source(body)})
            continue

        count += 1
        cell_outputs = outputs[index] if index < len(outputs) else []
        # `execute_result` outputs carry the counter; normalise it to this cell's.
        for output in cell_outputs:
            if output.get("output_type") == "execute_result":
                output["execution_count"] = count

        rendered.append(
            {
                "cell_type": "code",
                "execution_count": count,
                "metadata": {},
                "outputs": cell_outputs,
                "source": to_source(body),
            }
        )

    return {
        "cells": rendered,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": LANGUAGE_INFO,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(cells: list[tuple[str, str]], name: str) -> list[list[dict[str, Any]]]:
    """Run every code cell in one namespace, as a kernel would.

    Returns one output list per cell, in notebook format. A cell that raises aborts
    the build, because a notebook that does not run should not be committed.

    INFO logging is switched off for the duration. The pipeline's loggers stamp every
    record with a wall-clock time and write to whatever ``sys.stdout`` was live when the
    logger was first built - which, during this build, is the per-cell capture buffer.
    Left alone that leaks a timestamp into the notebook (so two builds never match) and
    silently drops later records into a buffer that is no longer being read. Warnings and
    errors still surface, which is what a reader actually needs to see.
    """
    import json as _json

    displayed: list[Any] = []

    def _display(*values: Any) -> None:
        """Stand-in for IPython's ``display``: record each argument for embedding.

        Deliberately ``*values`` and not ``list.extend``. ``extend`` takes an *iterable*
        of things to display, so ``display(styler)`` would raise (a Styler is not
        iterable) and ``display(frame)`` would silently record the frame's *column
        names* instead of the frame - a wrong table with no error to show for it.
        """
        displayed.extend(values)

    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "json": _json,
        # `display()` is a builtin in a real kernel; here it just records the value.
        "display": _display,
    }

    logging.disable(logging.INFO)
    try:
        return _run_cells(cells, name, namespace, displayed)
    finally:
        logging.disable(logging.NOTSET)


def _run_cells(
    cells: list[tuple[str, str]],
    name: str,
    namespace: dict[str, Any],
    displayed: list[Any],
) -> list[list[dict[str, Any]]]:
    """Execute each code cell in ``namespace`` and collect what it produced."""
    outputs_by_cell: list[list[dict[str, Any]]] = []

    for index, (kind, body) in enumerate(cells):
        if kind != "code":
            outputs_by_cell.append([])
            continue

        displayed.clear()
        stream = io.StringIO()
        outputs: list[dict[str, Any]] = []
        result: Any = None
        has_result = False

        try:
            statements, trailing = _split_trailing_expression(body)
            with contextlib.redirect_stdout(stream):
                if statements:
                    exec(compile(statements, f"<{name}:cell{index}>", "exec"), namespace)  # noqa: S102
                if trailing:
                    result = eval(compile(trailing, f"<{name}:cell{index}>", "eval"), namespace)  # noqa: S307
                    has_result = True
        except Exception:
            print(f"\n!! {name} cell {index} raised:\n")
            traceback.print_exc()
            raise SystemExit(1)

        text = stream.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": to_source(text)})

        outputs.extend(_value_output(value, output_type="display_data") for value in displayed)
        outputs.extend(_figure_outputs())

        # A trailing expression that evaluates to None is not shown, matching IPython:
        # otherwise every cell ending in a function call would gain a stray `None`.
        if has_result and result is not None:
            outputs.append(_value_output(result, output_type="execute_result"))

        outputs_by_cell.append(outputs)

    return outputs_by_cell


def main() -> int:
    # Guarded rather than ``exist_ok=True``: some sandboxes raise on a redundant
    # mkdir instead of silently succeeding.
    if not NOTEBOOK_DIR.exists():
        NOTEBOOK_DIR.mkdir(parents=True)
    _reset_table_ids()
    for name, cells in NOTEBOOKS.items():
        print(f"executing {name} ...", flush=True)
        outputs = execute(cells, name)
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(build_notebook(cells, outputs), indent=1) + "\n", encoding="utf-8")

        code_cells = sum(1 for kind, _ in cells if kind == "code")
        populated = sum(1 for index, (kind, _) in enumerate(cells) if kind == "code" and outputs[index])
        print(
            f"  wrote {path.relative_to(PROJECT_ROOT)} "
            f"({len(cells)} cells, {populated}/{code_cells} code cells with output)"
        )
    print("\nall notebooks executed cleanly and were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
