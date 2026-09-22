# AuditLens

**Financial anomaly detection and audit analytics over a 30,000-voucher general ledger.**

AuditLens is a working model of how an audit data analytics engagement actually runs:
generate a realistic ledger, clean it the way an ERP extract actually arrives, run nine
independent audit procedures over it, add an unsupervised model, combine the evidence
into a single explainable risk score, and hand an auditor a triage list they can work.

Every number in this README was produced by the code in this repository. Run
`python src/run_pipeline.py` and you will get the same ones — the pipeline is
deterministic and `tests/test_reproducibility.py` enforces that across processes.

```
30,128 vouchers · CNY 4.44B · 409 vendors · 2024-01-01 → 2026-01-03
```

---

## What this is, and what it is not

**It is** an end-to-end analytics pipeline that treats audit analytics as an
*evidence-ranking* problem: 30,128 vouchers is far more than any team can read, so the
job is not "find the fraud" — it is "decide what to read first, and be able to justify
that decision to a partner."

**It is not** a bookkeeping app, a CRUD demo, or a fraud detector. There is no LLM, no
agent framework, no deep learning, and no accuracy claim. The most useful thing this
project does is state precisely what it cannot do (see [Limitations](#limitations)).

Three design commitments drive everything else:

| Commitment | Why |
|---|---|
| **Every flag must answer "why"** | A score an auditor cannot interrogate is a score they will not use. Each voucher carries an ordered, plain-English reason list naming the value and the threshold it breached. |
| **The dashboard never shows the answer key** | The injected `anomaly_label` / `anomaly_type` columns exist so the model can be graded, not so an auditor can be handed the answers. A test fails the build if either ever renders. |
| **Honest numbers, including the ugly ones** | Two of the nine rules run at precision 0.15–0.17. That is reported here, on the dashboard, and in the methodology — not tuned away. |

---

## Headline results

| | |
|---|---|
| Population | **30,128** clean vouchers, CNY **4,443,451,722** |
| Raw extract | 30,223 rows → 285 data-quality issues (**0.94%**) |
| Injected anomalies | **919** (3.05%) across 9 patterns — ground truth, used only for grading |
| Flagged by rules | **2,208** vouchers (**7.33%**) — the reviewable population |
| High or Critical risk | **255** vouchers (**0.85%**), carrying **CNY 267.0M** |
| Anomalies caught by the nine rules | **902 of 919 (98.2%)** |
| Anomalies caught by the model and no rule | **2** |
| Anomalies caught by neither | **15** |
| Isolation Forest | ROC-AUC **0.816**, precision 0.291, recall 0.286 |
| Benford first-digit MAD | **0.00218** — close conformity |
| Test suite | **293 tests**, all passing — including in-process render tests for all six dashboard pages |

### The most important number here is 98.2%, and it is a warning

The nine rules catch **902 of the 919 injected anomalies**. That sounds like a triumph.
It is not — it is an artefact of how the benchmark was built. The anomalies were
injected to match the rule definitions, so of course the rules find them. Several rules
show recall of exactly 1.000 for the same reason.

The honest reading: **this benchmark cannot demonstrate that the model adds value.**
The model catches 263 anomalies, but 261 of those were already caught by a rule, and
only **2** were caught by the model alone. It raised 244 vouchers that no rule raised —
and 242 of those are false positives.

What the model *does* demonstrate is that it **ranks** well without labels: ROC-AUC
0.816, and 47% precision on the top 0.5% of the ranking (a 15× lift over random). Its
real argument is that it would generalise to patterns nobody wrote a rule for — and
that is precisely the thing a synthetic benchmark with known answer keys cannot test.
Saying so is more useful than claiming the model found 244 hidden anomalies, which is
what an earlier version of this project's own summary implied.

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │  src/data_generator.py       │   seeded, reproducible
                        │  30,000 vouchers + 420       │   log-normal amounts,
                        │  vendors + 140 employees     │   working calendar,
                        └──────────────┬───────────────┘   vendor lifecycle
                                       │
                        ┌──────────────▼───────────────┐
                        │  data/raw/*.csv              │   deliberately dirty:
                        │  30,223 rows                 │   dupes, bad dates,
                        └──────────────┬───────────────┘   unbalanced, orphans
                                       │
   ┌───────────────────────────────────▼───────────────────────────────────┐
   │  src/data_cleaning.py                                                 │
   │  repair what is safe to repair, flag what is itself the finding       │
   │  → DataQualityReport: 285 issues, every one of them reported          │
   └───────────────────────────────────┬───────────────────────────────────┘
                                       │  30,128 clean vouchers
                        ┌──────────────▼───────────────┐
                        │ src/feature_engineering.py   │  20 features, each one
                        │                              │  expressible as an audit
                        └──────────────┬───────────────┘  question
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐          ┌──────────▼─────────┐          ┌─────────▼────────┐
│ audit_rules.py │          │ anomaly_detection  │          │    benford.py    │
│  9 independent │          │  IsolationForest   │          │  first digit +   │
│  procedures    │          │  + StandardScaler  │          │  first two, MAD, │
│  noisy-OR       │          │  unsupervised      │          │  χ², per-bucket z│
└───────┬────────┘          └──────────┬─────────┘          └─────────┬────────┘
        │  rule_risk 0.40              │  ml_risk 0.25                │ statistical 0.10
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │  src/risk_scoring.py         │  + vendor_risk 0.15
                        │  Audit Risk Score 0-100      │  + amount_risk 0.10
                        │  + risk_reasons for every    │
                        │    single flagged voucher    │
                        └──────────────┬───────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐          ┌──────────▼─────────┐          ┌─────────▼────────┐
│ database.py    │          │   reporting.py     │          │  dashboard/      │
│ SQLite         │          │   10 matplotlib    │          │  6-page          │
│ auditlens.db   │          │   charts           │          │  Streamlit app   │
│ 15 SQL queries │          │                    │          │                  │
└────────────────┘          └────────────────────┘          └──────────────────┘
```

`src/run_pipeline.py` runs all nine stages in ~30 seconds. Each stage is independently
runnable (`python src/audit_rules.py`, `python src/benford.py`, …), which is how the
project was developed and how it is debugged.

---

## Quickstart

```bash
git clone <this-repo> && cd AuditLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make pipeline      # generate → clean → feature → rules → model → benford → score → sql → report
make dashboard     # streamlit run dashboard/app.py
make test          # full suite
```

Or with the CLI directly:

```bash
python src/run_pipeline.py                 # the whole thing, ~30s
python src/data_generator.py --transactions 50000 --seed 7   # a bigger, different ledger
streamlit run dashboard/app.py
```

Requires Python 3.11+. Everything is configured through `src/utils.py` and
`GeneratorConfig` — no absolute paths, no environment variables.

---

## The nine audit procedures

Each procedure is independent, returns a score in `[0, 1]` plus a human-readable
reason, and carries a weight. Weights are combined with **noisy-OR**, not a vote:

```
combined = 1 − Π (1 − weight_i × score_i)
```

Two weak indicators pointing the same way beat either alone; a single strong indicator
is not diluted by the eight rules that stayed silent. A majority vote would discard
exactly the evidence an auditor cares about most.

| Procedure | Trigger | Weight | Flagged | Precision | Recall |
|---|---|---|---|---|---|
| Duplicate Payment | same vendor+invoice+amount, or same vendor+amount within 5 days | 0.18 | 108 | **1.000** | 1.000 |
| Split Transaction | 2+ payments, same vendor, same day, each 90–99.5% of the CNY 50k threshold | 0.16 | 179 | 0.603 | 0.991 |
| Self Approval | `created_by == approved_by` | 0.14 | 126 | **1.000** | 1.000 |
| Unusual Vendor | new / dormant / shared bank account / amount spike, gated at ≥ P90 | 0.12 | 501 | **0.148** | 0.822 |
| Large Round Amount | multiple of 1,000 **and** ≥ P95 **and** ≥ CNY 100,000 | 0.10 | 505 | **0.214** | 1.000 |
| Rapid Payment | settled < 6 hours after invoice receipt | 0.08 | 90 | **1.000** | 1.000 |
| Suspicious Description | fraud-indicative keyword, or narration < 10 characters | 0.08 | 147 | 0.735 | 1.000 |
| Rare Account Usage | account frequency < 1%, escalated for designated rare accounts | 0.08 | 54 | **1.000** | 1.000 |
| Weekend / Holiday Posting | transaction date on a weekend or PRC public holiday | 0.06 | 765 | **0.165** | 1.000 |

**2,475 alerts across 2,208 distinct vouchers.**

### Reading that table honestly

Three things have to be said out loud, because the table is easy to misread:

1. **Precision and recall are per pattern, not against all 919 anomalies.** For a
   given rule the positive class is `anomaly_type == rule_key` — *when this rule
   fires, how often is it pointing at the pattern it claims to detect?* So `recall =
   1.000` means "every voucher injected with this pattern was caught", not "every
   anomaly was caught". **No rule catches every anomaly.**
2. **Recall of 1.000 on several rules is a benchmark artefact.** The injected
   duplicates follow exactly the pattern the duplicate rule tests for. A rule tuned on
   a known pattern finds all of it. That says nothing about a real ledger.
3. **The two low-precision rules are scoping tools, not detectors.** Unusual Vendor
   (0.148) and Weekend Posting (0.165) select populations worth looking at. Firing on
   one weekend voucher is not evidence about that voucher; it is a reason to look at
   the weekend population.

The Large Round Amount rule deliberately requires three conditions, not one. A bare
`amount % 1000 == 0` fires on 505 of 30,128 vouchers and is nearly meaningless —
plenty of legitimate payments are round because rent and retainers are round.

---

## Benford's Law

| Test | n | MAD | χ² | df | p | Conformity |
|---|---|---|---|---|---|---|
| First digit | 30,128 | **0.00218** | 16.00 | 8 | 0.042 | Close conformity |
| First two digits | 30,128 | **0.00071** | 196.09 | 89 | < 0.001 | Close conformity |

MAD is the deciding measure, banded per Nigrini: <0.006 close, 0.006–0.012 acceptable,
0.012–0.015 marginal, >0.015 nonconformity. The **debit leg only** is tested — each
voucher is stored once with `debit_amount == credit_amount`, so testing both legs
would double-count every value.

**The two tests disagree, and that is the interesting part.** The first-two-digits χ²
of 196.09 on 89 df reads as a strong rejection, while the MAD sits deep inside close
conformity. MAD measures the average *magnitude* of the deviation, which is
negligible; χ² accumulates 90 individually tiny deviations into a large statistic.
With 30,000 observations χ² will reject distributions whose departures are far too
small to matter. Per-bucket z-scores tell the same story: 13 of 90 buckets exceed
|z| > 1.96 against ~4.5 expected by chance, largest ≈ 4 standard errors. Worth a file
note; not a finding.

### What Benford's Law cannot do

**It cannot prove fraud, and it cannot clear anyone.**

- A **conforming** distribution is entirely compatible with fabrication. A competent
  fraudster who understands the test passes it.
- A **nonconforming** distribution is usually a legitimately skewed account. Payroll,
  fixed-asset additions, and anything with a narrow range of plausible values fail
  Benford for innocent reasons.
- It operates on a **population**. It cannot say "this voucher is odd".

So it is used as a **disaggregation tool** — the per-account and per-process
breakdowns are the deliverable, because that is where an enquiry starts. The
statistical component carries a weight of 0.10, the joint-smallest in the risk score.

---

## Isolation Forest

**Why unsupervised:** on a real engagement nobody hands you a column marked "fraud".
Training on injected labels would produce a model that had learned the generator's
patterns, not irregularity. Isolation Forest needs no labels and emits a continuous
score that can be cut at any point.

`IsolationForest(n_estimators=300, contamination=0.03, random_state=42)` over a
`StandardScaler` matrix of **20 features**, every one of which is expressible as an
audit question:

```
amount · log_amount · amount_percentile · vendor_transaction_count
vendor_average_amount · vendor_total_amount · vendor_amount_ratio · vendor_max_amount
vendor_age_days · days_since_last_transaction · account_transaction_frequency
is_weekend · is_round_amount · days_since_vendor_registration · approval_time_hours
payment_delay_hours · same_vendor_daily_transactions · description_length
is_self_approval · near_approval_threshold
```

**Deliberately excluded:** `vendor_master_risk_score` (a static attribute — including
it makes the model flag the same vendors every run regardless of behaviour),
all identifiers (a model that memorises vendor IDs does not generalise, and "this ID
looks unusual" is not a finding), and any ground-truth-derived column.

`contamination = 0.03` is a **planning assumption** about how much of the ledger is
worth reviewing, not a leak from the answer key. A real engagement has no answer key
and the model would be run identically.

### Results

| Metric | Value |
|---|---|
| Precision / Recall / F1 | 0.291 / 0.286 / 0.289 |
| **ROC-AUC** | **0.816** |
| Average precision | 0.201 (random baseline 0.031) |
| True positives / false negatives | 263 / 656 |
| Vouchers raised that no rule raised | 244 — of which **2** are real anomalies |

### What the model does and does not add

The nine rules catch **902 of 919** injected anomalies. The model catches 263, of which
**261 were already caught by a rule** — leaving **2** anomalies that only the model
found. Fifteen are caught by neither.

So on this benchmark the model is close to redundant, and it matters to be clear about
why: the anomalies were generated to match the rule definitions. That makes the rules
look exhaustive and any incremental model look useless. A benchmark built from a rule
set cannot measure what a model adds beyond that rule set — the measurement is
circular by construction.

What the model does establish is that it **ranks** well with no labels at all — ROC-AUC
0.816 and 47% precision at the top 0.5%. The argument for keeping it is generalisation
to patterns nobody wrote a rule for, and that argument is untestable here. Claiming
otherwise would be exactly the overclaim this project exists to avoid.

### The operating point is a resourcing decision

The metrics above are at a fixed cut. What a team actually chooses is how many
vouchers it can work:

| Vouchers reviewed | Precision | Recall |
|---|---|---|
| 151 (top 0.5%) | **47.0%** | 7.7% |
| 301 (top 1%) | 38.2% | 12.5% |
| 603 (top 2%) | 33.3% | 21.9% |
| 1,506 (top 5%) | 20.7% | 34.0% |
| 3,013 (top 10%) | 14.5% | 47.4% |

The top 0.5% gives **15× lift** over random selection.

**Recall matters more than precision in audit** — a missed misstatement is an audit
failure; a false positive is a wasted hour. But recall is not free, and pretending
otherwise is how analytics projects lose the confidence of the people who have to work
the output. The defensible position is to state the budget, state the recall it buys,
and let the engagement partner decide.

---

## The Audit Risk Score

`score = 100 × Σ (weight × component)`, clipped to `[0, 100]`.

| Component | Weight | Source |
|---|---|---|
| Rule indicators | **0.40** | noisy-OR of the nine procedures |
| Isolation Forest | **0.25** | normalised anomaly score |
| Vendor risk | **0.15** | counterparty score, 7 weighted features |
| Amount context | **0.10** | size percentile, roundness, materiality |
| Statistical (Benford) | **0.10** | account-level digit departure |

The specification suggested 0.45 / 0.30 / 0.15 / 0.10 and explicitly said not to copy
it mechanically. The final allocation follows three arguments:

1. **Rules outweigh the model (0.40 vs 0.25).** A rule is a control an auditor can
   read, re-perform and challenge; a model is a ranking. When they disagree, the
   auditable artefact should carry more weight — but the model keeps enough weight to
   move a voucher no rule caught, which is its entire purpose.
2. **Vendor risk is real but indirect (0.15).** A supplier's pattern matters to
   whether a specific payment deserves attention, but it is one step removed.
3. **Amount and statistical context are weak evidence (0.10 each).** Being large is not
   misconduct, and Benford cannot speak to an individual voucher. Note the size term is
   the percentile rank **above the median**, so a median-sized voucher scores zero
   rather than 0.5 — otherwise every voucher starts halfway up the scale for no reason.

| Band | Score | Meaning | Vouchers | Value |
|---|---|---|---|---|
| Low | 0–30 | no indicator fired; reviewed in aggregate | 28,261 | ¥3.45B |
| Medium | 30–60 | one indicator, or a moderate model score | 1,612 | ¥723.0M |
| High | 60–80 | multiple indicators, or one on a material amount | 254 | ¥265.8M |
| Critical | 80–100 | several independent indicators on a material amount | 1 | ¥1.25M |

Average score **14.64**; High + Critical = **255 vouchers (0.85%)**, ¥267.0M.

A well-calibrated triage pushes almost everything into Low. If half the ledger were
High, the score would not be discriminating — it would just be relabelling.

### Explainability

Every flagged voucher carries `risk_reasons`: an ordered list naming the indicator, the
value that breached it, and the threshold. Ordered by component weight, so the
strongest evidence reads first.

**A machine-learning flag is never emitted on its own.** It always arrives with the
features that drove it, because "the model said so" is not an audit explanation.
`tests/test_risk_scoring.py` asserts every High or Critical voucher has at least one
reason — a score with no explanation is treated as a bug, and currently zero of the
255 High/Critical vouchers violate it.

---

## Dashboard

Six pages, built with `st.navigation` / `st.Page`:

| Page | Answers |
|---|---|
| **Executive Overview** | What is the population, what did we flag, what does the triage cost? |
| **Transaction Explorer** | Why is *this* voucher flagged? Full drill-down: reasons, control timeline, score composition with weights, procedure-by-procedure result, peer comparison. |
| **Audit Rules** | What does each procedure test, at what threshold, and how well did it perform? |
| **Benford Analysis** | Observed vs expected digits, MAD, χ², per-bucket z, and the account/process disaggregation that is the actual deliverable. |
| **Machine Learning** | Confusion matrix, the precision/recall-vs-budget curve, the feature list, and what was deliberately excluded. |
| **Vendor Risk** | Counterparty ranking, shared bank accounts, dormancy reactivation, spend concentration. |

The dashboard reads the SQLite warehouse, **never the parquet with the labels** — the
answer key is structurally absent from the UI, not merely hidden. `tests/test_dashboard.py`
renders all six pages in-process with `AppTest` and fails if any page raises, produces no
content, or renders a ground-truth column.

---

## SQL

`data/auditlens.db` (SQLite) holds `transactions` (30,128), `vendors` (409),
`employees` (140) and `audit_alerts` (2,475). `sql/audit_queries.sql` contains **15
named business queries**, each with a `-- name:` marker so it can be run
individually or all together:

| Query | Demonstrates |
|---|---|
| `top_vendors_by_payment_amount` | `GROUP BY` + `ORDER BY` over the ledger |
| `duplicate_invoice_payments` | duplicate detection as a self-join |
| `weekend_and_holiday_postings` | date arithmetic against a holiday list |
| `monthly_transaction_totals` | period bucketing and trend |
| `high_risk_vendors` | joining vendor risk onto the master file |
| `accounts_with_unusual_activity` | account-level outlier ratio |
| `transactions_near_approval_threshold` | the population a split-payment scheme hides in |
| `self_approved_transactions` | segregation-of-duties violation |
| `vendor_spend_concentration` | `SUM() OVER ()` for share of total |
| `vendors_sharing_bank_accounts` | shell-company indicator via a self-join |
| `dormant_vendor_reactivation` | registration-to-first-payment gap |
| `payment_speed_by_vendor_category` | `julianday()` arithmetic on timestamps |
| `month_end_posting_concentration` | month-end clustering, a classic journal-entry risk |
| `rule_alert_summary` | alert counts and value by procedure |
| `risk_level_summary` | risk-band exposure in vouchers and value |

Run them all with `make sql`, or point any SQL client at the file.

---

## Project structure

```
AuditLens/
├── src/
│   ├── utils.py                 # single source of truth: paths, thresholds, chart of accounts
│   ├── data_generator.py        # seeded synthetic ledger + 9 anomaly injectors + data-quality defects
│   ├── data_cleaning.py         # repair / flag / report; never imputes a missing narration
│   ├── feature_engineering.py   # 20 audit-explainable features
│   ├── audit_rules.py           # 9 procedures, noisy-OR combination, self-evaluation
│   ├── benford.py               # first digit, first two digits, MAD, χ², z, disaggregation
│   ├── anomaly_detection.py     # Isolation Forest, budget-aware evaluation
│   ├── risk_scoring.py          # composite score, bands, reasons, vendor risk
│   ├── database.py              # SQLite warehouse + query runner
│   ├── reporting.py             # 10 charts, CJK-aware
│   └── run_pipeline.py          # 9 stages, ~30s
├── dashboard/
│   ├── app.py                   # st.navigation router
│   ├── common.py                # cached data access, theme, filters, formatters
│   ├── components.py            # KPI cards, risk badges, charts, tables
│   └── views/                   # the six pages
├── sql/audit_queries.sql        # 15 named business queries
├── tests/                       # 200+ tests, incl. cross-process reproducibility
├── docs/
│   ├── architecture.md          # design decisions, data contracts, what is deliberately excluded
│   └── methodology.md           # every threshold, every weight, every mistake
├── outputs/
│   ├── reports/                 # audit_summary.json, rule_evaluation.csv, model_metrics.json, …
│   └── charts/                  # 10 committed PNGs
├── notebooks/                   # 01 EDA · 02 audit analysis · 03 anomaly detection
├── tools/build_notebooks.py     # regenerates the notebooks; executes every cell before writing
└── Makefile
```

`outputs/` is committed on purpose — those reports are the evidence behind this README.
The ledger itself is not: it regenerates deterministically from `src/data_generator.py`.

---

## Business insights

Things a reviewer would actually take to an engagement:

1. **Rules alone catch 98.2% of the injected anomalies — and that number is a
   warning, not a result.** The benchmark's answer key was generated from the rule
   definitions, so the rules look exhaustive by construction. The model catches only
   **2** anomalies that no rule caught. Anyone quoting the 98.2% as evidence that
   rules are sufficient would be quoting the benchmark's design, not a finding about
   audit analytics. The honest conclusion is that this benchmark cannot measure the
   value of a model layered on top of a rule set.
2. **Two of nine procedures carry almost all the noise.** Weekend Posting (765 alerts,
   precision 0.165) and Unusual Vendor (501 alerts, 0.148) together produce 1,266 of
   2,475 alerts — 51% of the workload — for a small share of the detections. A real
   engagement should keep them as *scoping* procedures and stop treating their output
   as individual findings. That single change halves the review list.
3. **The triage collapses 30,128 vouchers into 255.** High/Critical is 0.85% of the
   population but 6.0% of its value (¥267.0M of ¥4.44B) — the score is finding risk
   where the money is, not just where the oddities are.
4. **Split-transaction detection is the highest-value rule that is not trivially
   perfect.** Precision 0.603 at recall 0.991: 71 false positives to catch 108 real
   splits. That is a good trade for a control that exists specifically to defeat an
   approval threshold, and it is the rule most worth investing engineering time in.
5. **Data quality is a finding, not a chore.** 285 issues on 30,223 rows (0.94%),
   including 45 unbalanced vouchers and 36 payments to vendors that do not exist in
   the master file. Those 36 are a master-data control failure. Dropping them silently
   — the instinctive "cleaning" move — would have destroyed the finding.
6. **The Benford result is a non-event, and reporting it as such is the point.** Close
   conformity at MAD 0.00218 tells the auditor to spend their time elsewhere. An
   analytics function that reports "no exception" is more credible than one that
   always finds something.

---

## Limitations

Stated plainly. A portfolio project that overclaims is worse than one that does less.

1. **The data is synthetic.** Anomalies were injected by a generator whose patterns are
   known to the detectors. **Every precision, recall and lift figure here is an upper
   bound** on real-world performance, and probably a generous one.
2. **Thresholds are conventions, not calibrations.** CNY 50,000 approval, CNY 100,000
   materiality, 6 hours for rapid payment, 90 days for a new vendor. A real engagement
   sets these from the client's control environment and its own materiality. They live
   in `src/utils.py` and are meant to be changed.
3. **Low-precision rules are scope, not detection.** Unusual Vendor (0.148) and Weekend
   Posting (0.165) define populations. They are not evidence about any one voucher.
4. **Several rules have recall 1.000, which is a benchmark artefact.** See above.
5. **Benford's Law cannot detect fraud.** It identifies populations that deserve
   disaggregation.
6. **The model is not tuned, and should not be.** Hyperparameters are defaults chosen
   for reproducibility, not optimised against the injected labels — tuning them would
   make the reported metrics even more of an upper bound.
7. **No temporal validation.** The ledger is split into periods for reporting, but the
   model is not trained on one period and tested on another. A real deployment would
   have to check that behaviour is stable over time, and would have to monitor drift.
8. **Single currency, single entity.** Multi-entity consolidation, intercompany
   elimination and transfer pricing are out of scope. FX is normalised to CNY with
   static rates, so no FX gain/loss analysis is possible.
9. **The model explains itself at the feature level, not the voucher level.** It reports
   which features were extreme; it does not claim to say why a specific entry was made.
   That remains the auditor's job — which is the point.

---

## Interview talking points

If you are reading this as a portfolio project, these are the things worth being able
to defend. Each is a real decision in the code.

**"Why noisy-OR instead of a weighted sum or a vote?"**
A vote throws away intensity and needs a majority. A weighted sum lets two weak
indicators cancel a strong one, which is wrong. Noisy-OR treats each indicator as
independent evidence and compounds it, so two weak signals agreeing beat either alone,
and one strong signal is never diluted by silence.

**"Why is the model only 0.25 of the score when it's the most sophisticated part?"**
Because a rule is an auditable control and a model is a ranking. An auditor can
re-perform a rule in front of a client; they cannot re-perform a forest. The score
should weight the evidence an auditor can defend.

**"Precision 0.29 sounds bad. Why ship it?"**
Because the alternative is not precision 1.0, it is precision 0. And precision at a
fixed cut is the wrong question — at the top 0.5% the model is 47% precise at 15× lift,
which is a very usable list.

**"Does the model actually add anything over the rules?"**
On this benchmark, barely — 2 anomalies that no rule caught. And I would not defend the
model on that number, because the benchmark cannot support the claim: the anomalies were
injected to match the rule definitions, so the rules are near-exhaustive by
construction and any incremental detector looks useless. The model's measurable
property is that it ranks well without labels (ROC-AUC 0.816, 15× lift at the top 0.5%).
Whether that ranking would surface an irregularity nobody wrote a rule for is exactly
what a benchmark with a known answer key cannot tell you. I would want a real labelled
engagement, or a held-out anomaly family the rules were never written against, before
making that claim.

**"How do you know the model isn't just cheating?"**
It never sees a label. `contamination = 0.03` is a review-budget assumption, not the
true anomaly rate, and the model is trained without labels. The labels are used once,
afterwards, to grade — and `dashboard/views/machine_learning.py` discloses exactly
that on the page itself. There is also a test that fails if the dashboard ever renders
a ground-truth column.

**"What was the hardest bug?"**
The generator was silently irreproducible. `rng.choice` samples by index, and one call
site sampled from `list(some_set)` — a set of vendor-id strings, whose iteration order
depends on `PYTHONHASHSEED`. Same seed, different ledger, every process. The full test
suite passed the whole time because an in-process test shares one hash seed and cannot
see it. It surfaced by diffing two pipeline runs by hand. The fix is a sorted list; the
guard is a test that runs the generator in two subprocesses with different hash seeds —
and that guard was verified to fail when the fix is reverted.

**"You report a Benford χ² p-value of 5e-10 next to 'close conformity'. Which is it?"**
Both, and they answer different questions. χ² tests whether the distribution is
*exactly* Benford, which with 30,000 observations it never is. MAD measures whether the
departure is *large enough to matter*. The MAD is 0.00071. An auditor who acts on the
p-value alone widens testing for no reason; the MAD is the measure that maps to the
audit question.

**"What would you do differently in production?"**
Temporal validation and drift monitoring first — the current evaluation is a single
snapshot. Then tune the thresholds from the client's actual control environment rather
than conventions. Then instrument the review workflow: the only real measure of an
analytics function is whether the flagged items lead to findings, and that feedback
loop is what would let you calibrate precision and recall against reality instead of
against a benchmark.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/methodology.md`](docs/methodology.md) | Every threshold, every weight, every rule's rationale, and the six places the first attempt was wrong |
| [`docs/architecture.md`](docs/architecture.md) | Stage contracts, data contracts, design decisions, testing strategy |
| `notebooks/01_eda.ipynb` | Population shape, amount distribution, seasonality, vendor concentration, data quality |
| `notebooks/02_audit_analysis.ipynb` | The nine procedures, their precision/recall trade-off, and the full Benford analysis |
| `notebooks/03_anomaly_detection.ipynb` | Isolation Forest, confusion matrix, the budget curve, and the honest answer to "does the model add anything?" |
| `outputs/reports/` | `audit_summary.json`, `rule_evaluation.csv`, `model_metrics.json`, `benford_results.json`, `data_quality_report.json`, `high_risk_transactions.csv` |
| `outputs/charts/` | 10 charts: risk distribution, component contributions, Benford observed-vs-expected, confusion matrix, precision@k, top risk vendors and accounts |

The notebooks are generated by `tools/build_notebooks.py`, which **executes every code
cell before writing the file** — so a notebook that does not run cannot be committed.

## Reproducing every number

```bash
pip install -r requirements.txt
python src/run_pipeline.py     # ~30 seconds, deterministic
python -m pytest tests/ -q     # 293 tests
```

Two consecutive runs produce byte-identical reports under `outputs/reports/`. This is
enforced by `tests/test_reproducibility.py`, not assumed.

## Testing

293 tests, all passing. `make test` runs the lot; `make test-fast` skips the dashboard
render suite.

| File | Tests | What it pins down |
|---|---|---|
| `test_data_cleaning.py` | 55 | Each repair path: missing values, duplicates, invalid dates and amounts, debit/credit balance, orphan vendor IDs, inconsistent account names, currency normalisation |
| `test_audit_rules.py` | 36 | One class per procedure, plus the text-encoded-label fallback in `evaluate()` |
| `test_benford.py` | 56 | Expected frequencies, MAD bands, χ², the per-bucket z-score, and the schema stability of the insufficient-data branch |
| `test_risk_scoring.py` | 55 | Component scores, the noisy-OR combination, band boundaries, `risk_reasons` wording |
| `test_dashboard.py` | 62 | Renders all six pages in-process via `AppTest` and asserts on the text each one emits; also pins the no-ground-truth guard at both the rendered-output and grid-builder level |
| `test_utils.py` | 19 | `as_flag_series` across every dtype the label takes, including the two string traps |
| `test_reporting.py` | 6 | The flag-vs-anomaly distinction in `detector_overlap` |
| `test_reproducibility.py` | 4 | Runs the generator in two subprocesses with different `PYTHONHASHSEED` values and compares hashes |

Four of these are regression guards for bugs that were actually shipped during
development — the reproducibility defect, the inert Benford flag, the string-encoded
label, and the overstated model contribution. `docs/methodology.md` documents each one.

## License

MIT — see [LICENSE](LICENSE).
