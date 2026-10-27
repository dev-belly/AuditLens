# Support

Financial audit analytics over a deterministic synthetic general ledger, with explainable review workpapers.

## Start here

Read the [README](README.md) for setup, examples and method boundaries. The tested maintenance commands are defined in the [CI workflow](.github/workflows/ci.yml). Environment: Python 3.11+.

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run the pipeline before opening the Streamlit dashboard or testing generated workpapers. Prefer a small generated ledger for a reproduction. Real ledger extracts, vendor bank details, employee identities, reviewer notes and audit evidence must stay out of public reports.

## Ask a question or report a problem

Use [GitHub Issues](https://github.com/dev-belly/AuditLens/issues) for usage questions, reproducible bugs and scoped feature proposals. English and Chinese reports are welcome. Check existing issues first and use the bug or feature form when it fits. Support is best effort; no response time or delivery date is promised.

Include the generator seed, pipeline stage, a synthetic voucher example, and the affected rule or workpaper field. Distinguish rule flags, review selections, and injected labels; an alert is not a confirmed fraud finding.

Include the commit SHA, operating system, dependency versions, exact command, expected behavior and a minimal reproduction. Share only redacted logs and data you are allowed to publish. Remove credentials, personal information and confidential or licensed records. For a suspected security vulnerability, follow [SECURITY.md](SECURITY.md) before posting details.

## Verify locally

Run relevant checks before reporting a regression or opening a pull request:

```bash
python -m ruff check src dashboard tests tools
python src/run_pipeline.py
python -m pytest tests/ -q
```

These are verification commands, not a claim that a new environment has already passed them. For changes that affect results, retain input provenance, configuration and seeds alongside the output.
