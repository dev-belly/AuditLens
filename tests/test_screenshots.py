"""The committed dashboard captures, and the four places the six pages are named.

``docs/screenshots/README.md`` calls them "committed captures of all six dashboard
pages". For a reader who will not run the app, they *are* the dashboard - and until
this file they were the last documented artefact in the repository that nothing
checked.

The six pages are named in four places:

1. ``dashboard/app.py`` - the ``title=`` of each ``st.Page``, which is the sidebar
   label Streamlit actually renders.
2. ``dashboard/views/*.py`` - the string each view passes to ``page_header``, which
   is the heading that proves the page has painted.
3. ``tools/capture_screenshots.py`` - ``PAGES``, which says which label to click,
   which heading to wait for, and what to call the PNG.
4. ``docs/screenshots/README.md`` - the table a reader uses to tell one file from
   another.

A rename in (1) or (2) alone does make ``make screenshots`` fail - the click target
or the heading never appears, and the run times out. That is the good case, but it is
a *slow* one: it costs a browser launch and a 60-second timeout per page to learn
what these tests answer in milliseconds.

The quiet case is worse. The committed PNGs keep the old name for the old page, the
README keeps describing them, and nothing disagrees with anything until somebody
re-runs the capture by hand. This is the same shape as the ``GROUND_TRUTH_COLUMNS``
tuple the architecture notes warn about - "three copies of the same tuple is how one
of them goes stale".

Everything here is static: the sources are parsed rather than imported, because
importing ``dashboard/app.py`` calls ``main()`` and starts Streamlit, which is not
something a test should do to read a list of strings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "dashboard" / "app.py"
VIEWS_DIR = PROJECT_ROOT / "dashboard" / "views"
CAPTURE = PROJECT_ROOT / "tools" / "capture_screenshots.py"
SHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"
SHOTS_README = SHOTS_DIR / "README.md"

#: PNG magic number. A committed capture that is not a PNG is a placeholder.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: The smallest real capture is ~315 KB at the default 1440x1000@2x viewport. The
#: threshold sits well below that so halving the scale does not fail the build, and
#: far above an empty or truncated file.
MIN_CAPTURE_BYTES = 50_000


def _app_pages() -> list[tuple[str, str]]:
    """``(sidebar title, view module)`` for each ``st.Page``, in source order."""
    tree = ast.parse(APP.read_text("utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Page"
    ]
    pages: list[tuple[str, str]] = []
    for call in sorted(calls, key=lambda node: node.lineno):
        title = next(
            (ast.literal_eval(kw.value) for kw in call.keywords if kw.arg == "title"), None
        )
        # `st.Page(executive_overview.render, ...)` - the module is the outer name.
        first = call.args[0] if call.args else None
        module = first.value.id if isinstance(first, ast.Attribute) else None
        if isinstance(title, str) and module:
            pages.append((title, module))
    return pages


def _capture_pages() -> list[tuple[str, str, str]]:
    """The ``(click label, expected heading, filename stem)`` rows of ``PAGES``."""
    tree = ast.parse(CAPTURE.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "PAGES":
            return [tuple(ast.literal_eval(elt)) for elt in node.value.elts]  # type: ignore[misc]
    raise AssertionError("tools/capture_screenshots.py no longer defines a literal PAGES")


def _rendered_headings() -> dict[str, str]:
    """The ``page_header`` string each view module renders, keyed by module stem."""
    headings: dict[str, str] = {}
    for path in sorted(VIEWS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "page_header"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                headings[path.stem] = node.args[0].value
    return headings


class TestTheAppDeclaresThePages:
    """``dashboard/app.py`` is the source of truth for which pages exist."""

    def test_the_app_declares_the_six_pages(self) -> None:
        """Guards the parser: an empty match would make the other checks vacuous."""
        pages = _app_pages()
        assert len(pages) == 6, f"expected six st.Page calls, found {len(pages)}: {pages}"

    def test_every_page_points_at_a_view_module_that_exists(self) -> None:
        missing = [
            module for _, module in _app_pages() if not (VIEWS_DIR / f"{module}.py").exists()
        ]
        assert not missing, f"st.Page points at view modules that do not exist: {missing}"


class TestTheCaptureToolTracksTheApp:
    """``make screenshots`` drives the real app, so its labels must be the app's."""

    def test_the_click_labels_match_the_app_in_order(self) -> None:
        """The click target is a sidebar label; a rename makes the capture hang."""
        expected = [title for title, _ in _app_pages()]
        actual = [label for label, _, _ in _capture_pages()]
        assert actual == expected, (
            "tools/capture_screenshots.py clicks labels the app does not render; "
            f"capture tool has {actual}, app renders {expected}"
        )

    def test_the_expected_headings_match_what_the_views_render(self) -> None:
        """The heading is the render check, so it must be the string on the page."""
        headings = _rendered_headings()
        expected = [headings.get(module) for _, module in _app_pages()]
        actual = [heading for _, heading, _ in _capture_pages()]
        assert actual == expected, (
            "the capture tool waits for headings the views do not render; "
            f"capture tool expects {actual}, views render {expected}"
        )

    def test_every_row_carries_a_heading_and_a_filename(self) -> None:
        incomplete = [row for row in _capture_pages() if not all(part.strip() for part in row)]
        assert not incomplete, f"PAGES rows with a blank field: {incomplete}"

    def test_the_filenames_are_distinct_and_numbered_in_order(self) -> None:
        """The README table and the directory listing both read top-to-bottom."""
        names = [name for _, _, name in _capture_pages()]
        assert len(set(names)) == len(names), f"duplicate capture filenames: {names}"
        assert [name[:2] for name in names] == [f"{i:02d}" for i in range(1, len(names) + 1)], (
            f"capture filenames are not numbered 01..{len(names):02d} in order: {names}"
        )


class TestTheCapturesOnDisk:
    """The PNGs themselves, checked in both directions."""

    def test_every_capture_exists(self) -> None:
        missing = [
            f"{name}.png" for _, _, name in _capture_pages() if not (SHOTS_DIR / f"{name}.png").exists()
        ]
        assert not missing, f"the capture tool writes files that are not committed: {missing}"

    def test_there_are_no_orphan_captures(self) -> None:
        """A renamed page leaves the old PNG behind, and the README still shows it."""
        expected = {f"{name}.png" for _, _, name in _capture_pages()}
        actual = {path.name for path in SHOTS_DIR.glob("*.png")}
        assert actual == expected, (
            f"orphan captures: {sorted(actual - expected)}; "
            f"missing captures: {sorted(expected - actual)}"
        )

    def test_every_capture_is_a_real_png(self) -> None:
        """A truncated or placeholder file would still be committed and still display."""
        for _, _, name in _capture_pages():
            path = SHOTS_DIR / f"{name}.png"
            data = path.read_bytes()
            assert data.startswith(PNG_MAGIC), f"{path.name} is not a PNG"
            assert len(data) > MIN_CAPTURE_BYTES, (
                f"{path.name} is {len(data)} bytes, too small for a dashboard render"
            )


class TestTheScreenshotReadme:
    """The table a reader uses to tell one file from another."""

    def test_every_capture_is_named_in_the_readme(self) -> None:
        text = SHOTS_README.read_text("utf-8")
        missing = [f"{name}.png" for _, _, name in _capture_pages() if f"{name}.png" not in text]
        assert not missing, f"docs/screenshots/README.md does not mention: {missing}"

    def test_the_readme_names_no_capture_that_does_not_exist(self) -> None:
        text = SHOTS_README.read_text("utf-8")
        mentioned = set(re.findall(r"\d{2}_[a-z_]+\.png", text))
        expected = {f"{name}.png" for _, _, name in _capture_pages()}
        ghosts = sorted(mentioned - expected)
        assert not ghosts, (
            f"docs/screenshots/README.md describes captures that do not exist: {ghosts}"
        )
