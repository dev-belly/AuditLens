"""The README's structural claims, kept honest.

Three kinds of claim live here, all of the same shape: a fact restated in the README
that the repository can contradict without anyone noticing.

**The test count.** It appears in seven places in the README plus one table row per
test file, and it has drifted repeatedly during development (302 -> 331 -> 351 -> 374 ->
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

**The Makefile's entry points.** ``.PHONY`` declared fifteen targets while the file
defined fourteen. ``lint`` had no recipe, so ``make lint`` printed "Nothing to be done"
and exited **0** - a command that looks like it ran, reports success, and inspects
nothing, which is the worst shape a check can take. ``make help`` did not list it
either, so the phantom was invisible to the one command a reviewer would run, and
nothing compared the declaration against the file it describes.

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
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
TESTS_DIR = PROJECT_ROOT / "tests"
MAKEFILE = PROJECT_ROOT / "Makefile"

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


#: Markdown links and images, plus the raw ``<img src="...">`` form the README uses for
#: its screenshot grid.
MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
IMG_SRC = re.compile(r"<img[^>]*?\ssrc=[\"']([^\"']+)[\"']", re.IGNORECASE)

#: Fenced blocks and inline code spans. Documentation that *shows* link syntax - this
#: very file's sibling, ``docs/architecture.md``, writes ``<img src="docs/screenshots/...">``
#: inside backticks - would otherwise be read as a reference and fail the check. A code
#: span is an example, not a link. That example is now the regression fixture: delete the
#: stripping below and `test_every_local_link_resolves` fails on architecture.md.
FENCED_CODE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")

DOCUMENTS = (
    README,
    PROJECT_ROOT / "docs" / "architecture.md",
    PROJECT_ROOT / "docs" / "methodology.md",
    PROJECT_ROOT / "docs" / "screenshots" / "README.md",
)


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercased, punctuation dropped, spaces to hyphens."""
    stripped = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", stripped)


def _anchors(path: Path) -> set[str]:
    return {
        _slug(match.group(1))
        for match in re.finditer(r"^#{1,6}\s+(.*)$", path.read_text("utf-8"), re.MULTILINE)
    }


def _prose(path: Path) -> str:
    """The document with code removed, so syntax examples are not read as references.

    Stripping an inline span leaves the surrounding link intact: ``[`docs/x.md`](docs/x.md)``
    becomes ``[](docs/x.md)``, which still matches ``MD_LINK`` and still yields the real
    target.
    """
    text = FENCED_CODE.sub("", path.read_text("utf-8"))
    return INLINE_CODE.sub("", text)


def _relative_targets(path: Path) -> list[str]:
    """Every non-http link or image target in a document."""
    text = _prose(path)
    targets = MD_LINK.findall(text) + IMG_SRC.findall(text)
    return [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]


class TestDocumentationLinks:
    """Every relative link and image in the docs must resolve.

    The README is the first thing a reviewer opens, and its dashboard section is six raw
    ``<img src="docs/screenshots/...">`` tags. Rename a capture and those become broken
    image icons on GitHub - a *visible* defect, and one no existing test caught, because
    the screenshot guard checks the files against the capture tool rather than against
    the README's references to them. Two guards, two different mistakes.
    """

    def test_the_parser_found_some_targets(self) -> None:
        """Guards the two checks below: an empty match would make them vacuous."""
        assert any(_relative_targets(doc) for doc in DOCUMENTS), (
            "no relative links or images were found; the parser may be broken"
        )

    def test_every_local_link_resolves(self) -> None:
        broken: list[str] = []
        for document in DOCUMENTS:
            for target in _relative_targets(document):
                path_part, _, anchor = target.partition("#")
                if not path_part:
                    if anchor and anchor not in _anchors(document):
                        broken.append(f"{document.name}: #{anchor}")
                    continue
                resolved = (document.parent / path_part).resolve()
                if not resolved.exists():
                    broken.append(f"{document.name}: {target}")
                elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
                    broken.append(f"{document.name}: {target} (no such heading)")
        assert not broken, f"documentation links that do not resolve: {broken}"

    def test_every_readme_image_exists(self) -> None:
        """The six dashboard captures, checked from the README's side."""
        images = IMG_SRC.findall(README.read_text("utf-8"))
        assert images, "the README no longer embeds any images"
        missing = [src for src in images if not (PROJECT_ROOT / src).exists()]
        assert not missing, f"the README embeds images that do not exist: {missing}"


#: ``.PHONY: a b c \`` - one declaration, possibly continued over several lines.
PHONY = re.compile(r"^\.PHONY:(.*?)(?=\n\S)", re.MULTILINE | re.DOTALL)

#: ``target:`` at the start of a line. The ``(?!=)`` keeps ``FOO := bar`` out.
TARGET = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_-]*):(?!=)", re.MULTILINE)

#: ``target:  ## what it does`` - a target that ``make help`` will print.
DOCUMENTED_TARGET = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_-]*):[^=\n]*##\s+(.*)$", re.MULTILINE)

#: ``make target``, as written in a command.
MAKE_INVOCATION = re.compile(r"\bmake\s+([a-zA-Z][a-zA-Z0-9_-]*)")

#: The colour codes ``make help`` writes around each target name.
ANSI = re.compile(r"\033\[[0-9;]*m")


def _makefile() -> str:
    return MAKEFILE.read_text("utf-8")


def _phony_names() -> set[str]:
    """The names ``.PHONY`` declares, with any line continuations removed."""
    match = PHONY.search(_makefile())
    assert match, "the Makefile no longer declares .PHONY"
    return set(match.group(1).replace("\\", " ").split())


def _targets() -> set[str]:
    """Every target the Makefile defines, documented or not."""
    return set(TARGET.findall(_makefile()))


def _documented_targets() -> dict[str, str]:
    """Targets carrying a ``## `` description, mapped to that description."""
    return dict(DOCUMENTED_TARGET.findall(_makefile()))


def _code(path: Path) -> str:
    """The code in a document - fenced blocks and inline spans - where commands live.

    The inverse of ``_prose``: scanning prose for ``make x`` would match sentences
    ("make the pipeline reproducible"), so the invocation check reads only the spans a
    reader would copy and paste.
    """
    text = path.read_text("utf-8")
    return "\n".join(FENCED_CODE.findall(text) + INLINE_CODE.findall(text))


@lru_cache(maxsize=1)
def _make_help() -> dict[str, str]:
    """What ``make help`` really prints, as target name to description.

    Run rather than re-parsed from the Makefile, because the ``help`` target builds its
    own list with a ``grep`` over the file. Re-parsing would only prove the Makefile
    agrees with itself; if that grep pattern broke, a reviewer would see a short list
    while this check stayed green.
    """
    make = shutil.which("make")
    if make is None:  # pragma: no cover - make ships with every supported platform
        pytest.skip("make is not installed, so its help output cannot be checked")

    result = subprocess.run([make, "help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"`make help` failed with exit code {result.returncode}:\n"
        f"{result.stdout}{result.stderr}"
    )
    listed = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"^ {2}(\S+) {2,}(.+)$", ANSI.sub("", result.stdout), re.MULTILINE
        )
    }
    assert listed, (
        "could not parse any targets out of `make help`; its output format may have "
        f"changed:\n{result.stdout}"
    )
    return listed


class TestTheMakefile:
    """The developer entry points must be real, and the documents must name them right.

    ``make`` is the project's advertised interface - the README's quick start is three
    ``make`` commands - so a target that exists only in ``.PHONY``, or a command the
    README names and the Makefile never defined, is a broken promise either way.
    """

    def test_the_parser_found_the_targets(self) -> None:
        """Guards the checks below: an empty parse would make them all vacuous."""
        assert len(_targets()) > 5, "could not parse targets out of the Makefile"
        assert _documented_targets(), "no Makefile target carries a `## ` description"
        assert _phony_names(), "could not parse the .PHONY declaration"

    def test_every_phony_name_is_a_real_target(self) -> None:
        """The defect this class exists for.

        A name in ``.PHONY`` with no target of its own is not a harmless typo:
        ``make <name>`` prints "Nothing to be done" and exits **0**, so anything that
        runs it - a script, a CI step, a reviewer checking the tree is lint-clean - is
        told the work succeeded when no work happened. ``lint`` sat in this state, and
        ``make help`` never mentioned it, so only the declaration gave it away.
        """
        ghosts = sorted(_phony_names() - _targets())
        assert not ghosts, (
            f".PHONY declares targets the Makefile does not define: {ghosts}; "
            "`make <name>` exits 0 without running anything"
        )

    def test_every_target_is_phony(self) -> None:
        """The other direction: an unlisted target is disabled by a file of its name."""
        missing = sorted(_targets() - _phony_names())
        assert not missing, (
            f"these targets are not declared .PHONY: {missing}; a file or directory "
            "sharing the name would make make skip them"
        )

    def test_make_help_lists_every_target(self) -> None:
        """``make help`` greps the Makefile, so it can disagree with the file.

        Compared against *every* target rather than against the ones carrying a ``## ``
        line. The weaker comparison moves both sides together: delete a description and
        the target leaves ``make help`` while also leaving the set it is compared to, so
        the check stays green while a reviewer's help output silently loses a target.
        """
        targets, listed = _targets(), set(_make_help())
        assert listed == targets, (
            f"`make help` lists {sorted(listed)} but the Makefile defines "
            f"{sorted(targets)}; either a target lost its `## ` description, or the "
            "help target's grep no longer matches it"
        )

    def test_the_documents_only_name_real_targets(self) -> None:
        """Every ``make x`` in the docs must resolve, or the quick start is a dead end."""
        targets = _targets()
        unknown = [
            f"{document.name}: make {name}"
            for document in DOCUMENTS
            for name in MAKE_INVOCATION.findall(_code(document))
            if name not in targets
        ]
        assert not unknown, f"the documentation names targets that do not exist: {unknown}"

