# AuditLens — Architecture

## What this document is for

It explains how the nine pipeline stages fit together, what each module is
responsible for, and which decisions were made deliberately rather than by
accident. It is written for someone reviewing the repository who wants to know why
the code is shaped the way it is before reading it.

## The shape of the system

AuditLens is a **batch pipeline with two presentation layers**. It is not a service
and has no scheduler, because the thing it models — a journal entry testing
engagement — is a batch process: the ledger is extracted once, analysed, and the
findings are worked over the following weeks.

```
                         ┌──────────────────────────────────────┐
                         │  src/data_generator.py               │
                         │  src/anomaly_injection.py            │
                         │  Synthetic ledger: vouchers, vendors,│
                         │  employees + injected anomalies      │
                         └───────────────┬──────────────────────┘
                                         │  data/raw/*.csv
                                         ▼
                         ┌──────────────────────────────────────┐
                         │  src/data_cleaning.py                │
                         │  Missing values, duplicates, invalid │
                         │  dates/amounts, FX normalisation     │
                         └───────────────┬──────────────────────┘
                                         │  *_clean.parquet
                                         │  data_quality_report.json
                                         ▼
                         ┌──────────────────────────────────────┐
                         │  src/feature_engineering.py          │
                         │  20 audit-explainable features       │
                         └───────────────┬──────────────────────┘
                                         │  transactions_features.parquet
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │ src/audit_rules   │ │ src/benford       │ │ src/anomaly_      │
        │ 9 procedures,     │ │ digit tests +     │ │ detection         │
        │ noisy-OR combine  │ │ disaggregation    │ │ Isolation Forest  │
        └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                  │  rule flags         │  account Benford    │  anomaly score
                  └─────────────────────┼─────────────────────┘
                                        ▼
                         ┌──────────────────────────────────────┐
                         │  src/risk_scoring.py                 │
                         │  0-100 composite + risk_reasons      │
                         └───────────────┬──────────────────────┘
                                         │  transactions_scored.parquet
                                         │  vendor_risk.parquet
                    ┌────────────────────┼────────────────────┐
                    ▼                                         ▼
        ┌───────────────────────┐              ┌──────────────────────────┐
        │ src/database.py       │              │ src/reporting.py         │
        │ SQLite warehouse      │              │ PNG charts + JSON/CSV    │
        └───────────┬───────────┘              └──────────┬───────────────┘
                    │  data/auditlens.db                  │  outputs/
                    ▼                                     ▼
        ┌───────────────────────┐              ┌──────────────────────────┐
        │ dashboard/ (Streamlit)│              │ README, docs, notebooks  │
        │ 6 pages               │              │                          │
        └───────────────────────┘              └──────────────────────────┘
                    ▲
                    │  sql/audit_queries.sql (15 named analyst queries)
```

`src/run_pipeline.py` orchestrates stages 1–9 in order and prints a summary. Every
stage is also independently runnable (`python src/benford.py`), which matters when
one of them fails at 2am.

## Stage-by-stage responsibilities

| Stage | Module | Input | Output | Notes |
|---|---|---|---|---|
| 1 | `data_generator` + `anomaly_injection` | `GeneratorConfig` | `data/raw/*.csv` | 30,000 vouchers, 3% injected anomalies |
| 2 | `data_cleaning` | raw CSV | `*_clean.parquet`, `data_quality_report.json` | Repairs only what is safely repairable |
| 3 | `feature_engineering` | clean parquet | `transactions_features.parquet` | 20 features, all audit-explainable |
| 4 | `audit_rules` | features | rule flags, `rule_alerts.csv`, `rule_evaluation.csv` | 9 procedures, noisy-OR combination |
| 5 | `benford` | features | `benford_results.json` | First-digit and first-two-digits, plus disaggregation |
| 6 | `anomaly_detection` | features | `anomaly_score`, `model_metrics.json` | Isolation Forest + `StandardScaler` |
| 7 | `risk_scoring` | rules + ML + Benford | `transactions_scored.parquet`, `vendor_risk.parquet` | 0–100 composite, `risk_reasons` |
| 8 | `database` | scored parquet | `data/auditlens.db` | 5 tables, 7 indexes, 15 analyst queries |
| 9 | `reporting` | scored parquet | `outputs/charts/*.png`, `audit_summary.json`, `high_risk_transactions.csv` | The artefacts that get circulated |

## Design decisions worth defending

### One source of truth for configuration

`src/utils.py` holds every constant that the rest of the project depends on:
thresholds, the chart of accounts, the working calendar, the risk weights, the
anomaly taxonomy. The alternative — a threshold written in the rule and a
different one in the README — is how an audit analytics project loses credibility.
A reviewer can check any number in the README against one file.

### The ground truth is generated, then quarantined

The generator injects anomalies and records `anomaly_label` / `anomaly_type`. Those
two columns are used for exactly one purpose: grading the detectors after the fact.
They are written to the warehouse so the evaluation is reproducible, and the
dashboard never selects them. `tests/test_dashboard.py::TestNoGroundTruthLeak`
enforces that, because "we remembered not to show it" is not a control.

Enforcement works at two levels, because they fail differently:

| Level | Test | What it catches |
|---|---|---|
| Rendered output | `test_ground_truth_columns_are_never_rendered` | A page that starts showing the labels today |
| The grid builder | `test_the_grid_ignores_columns_it_does_not_know_about` | A column added to the pipeline *later*, reaching the grid by default |

The second is the durable one. `transaction_table` lists its columns explicitly
rather than slicing `frame.columns`, so a new column cannot appear in the voucher
grid without someone deliberately adding it. The list of protected names lives once,
in `src.database.GROUND_TRUTH_COLUMNS`, and the dashboard imports it — three copies
of the same tuple is how one of them goes stale.

### Two splits, both pure moves

Two files crossed a thousand lines. Both were cut, and both cuts were pure moves: no
logic changed, no draw reordered, no cell rewritten, and every artefact byte-identical
afterwards. The verification is what makes a late refactor of the code that *produces*
those artefacts safe to attempt at all.

**`src/data_generator.py` → `src/data_generator.py` + `src/anomaly_injection.py`.**

`anomaly_injection.py` owns the nine patterns and the ground-truth columns they stamp;
`data_generator.py` owns the masters, the clean ledger and the data-quality defects.
The seam is the ground-truth boundary described above, which is easier to police when
the code that *writes* the labels is somewhere you can point at. Verified by diffing
`outputs/` before and after, and guarded by `tests/test_reproducibility.py`.

**`tools/build_notebooks.py` → `tools/build_notebooks.py` + `tools/notebook_content.py`.**

`notebook_content.py` holds what the three notebooks *say* — the prelude, the three
cell lists, the `NOTEBOOKS` registry. `build_notebooks.py` holds the machinery that
turns them into `.ipynb`: the kernel spec, the figure capture, the table-id
normalisation, the IPython-style output splitting. The seam is content versus
mechanism, and it is the same seam that keeps the notebook guard honest — the content
can change without touching the code that proves the build is deterministic. Verified
by diffing `notebooks/` before and after.

Three details are load-bearing:

- `data_generator` re-exports `ANOMALY_MIX` and `inject_anomalies`, so the import
  surface is unchanged for callers and tests.
- `anomaly_injection` imports `GeneratorConfig` only under `TYPE_CHECKING`. It needs
  the type for annotations and nothing else; importing it for real would make the two
  modules import each other.
- `notebook_content` defines no `PROJECT_ROOT` and reads no file. It is data the
  builder consumes, not a second entry point — so it cannot drift into doing work.

The structure tree in `README.md` is checked against the modules on disk by
`tests/test_documentation.py`, in both directions: a file listed but absent fails, and
a module present but unlisted fails.

### Rules and the model: measured, not assumed, to be complementary

The rules encode what auditors already know how to look for. The Isolation Forest is
supposed to find what nobody wrote down. That is the *hypothesis*, and the pipeline
measures it explicitly rather than asserting it.

The measurement has to be done on two different bases, and conflating them is the
easiest mistake to make with this project's output:

| Basis | Question | Result |
|---|---|---|
| **Flags** (`flagged_by_model_only`) | How many vouchers did the model raise that no rule raised? | 244 |
| **Anomalies** (`anomaly_overlap.caught_by_model_only`) | How many injected anomalies did the model catch that no rule caught? | **2** |

The first number looks like evidence of complementarity and is not: 242 of those 244
vouchers are false positives. The rules catch 902 of the 919 injected anomalies (98.2%),
which is itself an artefact — the anomalies were injected to match the rule
definitions, so the rules are near-exhaustive by construction.

An earlier version of this summary reported only the flag-level overlap, under the name
`detections_not_caught_by_rules`, which reads as "anomalies the rules missed". It is not
that, and the summary now reports both bases under names that cannot be confused.
`tests/test_reporting.py` asserts the two are consistent.

The honest conclusion is recorded in `docs/methodology.md`: this benchmark cannot
measure the value of a model layered on a rule set, and the project does not claim it
does.

### Explainability is structural, not bolted on

Every rule returns a `RuleResult` carrying a `reasons` Series, not just a score. The
risk scoring stage assembles `risk_reasons` from those, ordered by the weight of the
component that produced each one, and caps the list so the output stays readable.
A flagged voucher with an empty explanation is treated as a bug and is asserted
against in `tests/test_risk_scoring.py`.

### The warehouse is a real deliverable

Audit analytics is delivered to a team, not to one analyst. A single-file SQLite
database with documented tables and 15 named queries is what lets an engagement
manager answer their own question without waiting for the data scientist. The
dashboard reads from it by preference, so the app and the SQL path cannot diverge.

### Data cleaning flags rather than fixes

The cleaner repairs two things, both from an authoritative source: a missing
`posting_date` from the `transaction_date`, and a wrong `account_name` from the
chart of accounts. Everything else — a missing narration, an orphan vendor, an
unbalanced voucher — is flagged and kept. Dropping an unbalanced voucher would
understate the population, and imputing a description would destroy the signal the
suspicious-description rule depends on.

## Data contracts

Each stage's output is the next stage's input, so the column contracts matter.

**`transactions_features.parquet`** must carry the clean ledger plus the engineered
features in `REQUIRED_FEATURE_COLUMNS`. If a required feature is absent, the rule
engine rebuilds the whole feature set rather than defaulting it to zero — silently
defaulting produced a rule that flagged 100% of the ledger, which is documented in
`docs/methodology.md`.

**`transactions_scored.parquet`** carries 117 columns: the ledger, the nine
`<rule>_score` / `<rule>_flag` pairs, `rule_risk_score`, `anomaly_score`, the five
`risk_component_*` columns, `audit_risk_score`, `risk_level`, and `risk_reasons`.

**List-valued columns do not survive a parquet round-trip as lists.** They come back
as `numpy.ndarray`, and SQLite cannot bind them at all. `src/database.py` detects
non-scalar columns by inspection and drops them before the load, and
`src/utils.py::as_reason_list` normalises whichever of the three shapes (list,
ndarray, delimited string) it is handed. This is the kind of detail that only shows
up in production, so it is handled in one place.

## Testing strategy

`tests/` runs in about 20 seconds without the dashboard and a few minutes with it.

- **Hermetic unit tests** (`test_audit_rules`, `test_benford`, `test_risk_scoring`,
  `test_data_cleaning`) build the smallest ledger that can exercise the behaviour
  and assert on it. A change to the synthetic generator cannot silently invalidate
  them.
- **Render tests** (`test_dashboard`) execute all six pages in-process against the
  real pipeline output via `streamlit.testing.v1.AppTest`. A Streamlit page fails in
  a way unit tests never catch: it imports cleanly and then raises inside a callback
  because a filter returned an empty frame.
- **Constraint tests** assert the properties the project promises: no ground truth
  in the UI, a reason for every flag, weights that sum to one, scores that stay in
  range.
- **Artefact tests** (`test_notebooks`) check the committed notebooks rather than the
  code that builds them: that every cell which prints or plots carries output, that each
  notebook embeds a chart, and that no random table id or logged timestamp survives. A
  notebook is the one artefact here that can be silently *empty* — the build succeeds, the
  file is valid, and it renders as code with nothing underneath. `test_ci_summary` is the
  same idea for the CI job summary, and `test_reproducibility` for the generator.
- **Prose tests** (`test_readme_claims`, `test_documentation`). For a portfolio
  repository the numbers *are* the credibility: a reviewer who checks one headline figure
  and finds it stale stops trusting the rest. Nothing else in the suite reads the README,
  so before these files the figures were correct only for as long as nobody re-ran the
  pipeline with a different seed or threshold. Every headline figure, all nine rules'
  flagged/precision/recall, the Benford table, the risk-band counts and values, the
  component weights (which must sum to one) and the chart count are now compared against
  `outputs/reports/`. Counts are exact; money allows 1% because a figure quoted to two
  decimals in billions is only good to about that. `test_documentation` additionally runs
  `pytest --collect-only -q` in a subprocess and compares the testing table and the badge
  against the real collection, per file and in total. The purely textual version of that
  check could only prove the README agreed with itself, and let four new tests pass while
  the documented counts went stale.
- **Query tests** (`test_sql_queries`) execute all 15 queries in `sql/audit_queries.sql`
  against the built warehouse and assert each returns rows. `run_sql_file` deliberately
  surfaces a failure rather than raising — it logs and returns an empty frame — so without
  these tests a query naming a column that no longer exists would leave the library
  quietly short and the pipeline green. The same file pins the count the documentation
  states, because that count had already drifted once (the SQL header said "Fourteen
  queries" while the file held fifteen).
- **Artefact stability.** The notebooks are built twice in development and diffed. They
  are deliberately *not* diffed in CI: the embedded charts are rendered with whatever CJK
  font the runner has, so their PNG bytes legitimately differ from the macOS ones. Only
  `outputs/reports/` is held to the byte-identical standard across operating systems.

## What the architecture deliberately does not include

- **No LLM, no agent, no RAG, no deep learning.** The detection methods are ones an
  auditor can re-perform and challenge. A transformer that flags 3% of the ledger
  with no explanation would be useless in an audit file however accurate it was.
- **No streaming or scheduling.** The problem is a batch process; adding Airflow
  would be resume-driven design.
- **No authentication or multi-tenancy.** The dashboard runs on the analyst's
  machine against a local file.
- **No database server.** SQLite is the right choice for a single-file deliverable
  and keeps the repository self-contained.
