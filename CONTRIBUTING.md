# Contributing to AuditLens

Financial audit analytics over a deterministic synthetic general ledger, with explainable review workpapers.

Bug reports, documentation fixes and focused improvements are welcome. English and Chinese are both fine. For setup and questions, start with [SUPPORT.md](SUPPORT.md) and the [README](README.md). Follow [SECURITY.md](SECURITY.md) for suspected vulnerabilities.

## Prepare a change

Check existing issues and pull requests to avoid duplicating work. For a larger design or research change, describe the problem, assumptions and acceptance criteria in an issue first. Keep each pull request focused and explain the resulting behavior for a reader who has not followed the discussion.

Use Python 3.11+.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Activate the virtual environment with the command appropriate for your shell. Run commands from the repository root unless a command changes directory.

## Validate

The [CI workflow](.github/workflows/ci.yml) is the current reference for automated checks. Run the checks relevant to your change and report the actual outcome:

```bash
python -m ruff check src dashboard tests tools
python src/run_pipeline.py
python -m pytest tests/ -q
```

Add or update a meaningful regression test when behavior changes. A documentation-only change needs review of its commands, links and claims; do not invent test outcomes or benchmark results. If a check was skipped, say so in the pull request.

## Preserve evidence

Include the generator seed, pipeline stage, a synthetic voucher example, and the affected rule or workpaper field. Distinguish rule flags, review selections, and injected labels; an alert is not a confirmed fraud finding.

Run the pipeline before opening the Streamlit dashboard or testing generated workpapers. Prefer a small generated ledger for a reproduction. Real ledger extracts, vendor bank details, employee identities, reviewer notes and audit evidence must stay out of public reports.

Keep historical results distinguishable from reruns. Record configurations, seeds, environment and source revision when adding experiments. Avoid choosing favorable seeds or changing benchmark definitions to improve a headline number. Describe limitations and label proposed features as planned until implemented and verified.

## Submit

Use a descriptive branch and commit message, fill in the pull request template, and link any related issue. Include concise validation evidence and update relevant documentation. Do not include generated local caches, credentials, private datasets or unrelated edits. Contributions are reviewed as capacity permits; no response or merge date is promised.
