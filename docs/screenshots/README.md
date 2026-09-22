# Screenshots

Dashboard captures live here. The directory is empty in the repository on purpose.

The dashboard was verified by rendering all six pages headlessly through
`streamlit.testing.v1.AppTest` (see `tests/test_dashboard.py`, which asserts on the
text each page actually emits), but a headless browser was not available in the
development environment, so no committed PNGs were produced. Charts rendered by
matplotlib *are* committed, under `outputs/charts/`, and are referenced directly from
the main README.

To add your own captures:

```bash
make dashboard          # then open http://localhost:8501
```

Take one screenshot per page and save them here as
`01_executive_overview.png`, `02_transaction_explorer.png`, `03_audit_rules.png`,
`04_benford_analysis.png`, `05_machine_learning.png`, `06_vendor_risk.png`.
