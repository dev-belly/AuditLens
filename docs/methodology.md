# AuditLens — Methodology

## Purpose

This document explains how each analytical procedure works, which threshold it uses
and why, and — more importantly — what it cannot tell you. It also records the
places where the first attempt was wrong, because a methodology note that only
describes the final version hides the reasoning that makes the final version
defensible.

Every number quoted below comes from the committed artefacts in `outputs/reports/`,
produced by `python src/run_pipeline.py` with `RANDOM_SEED = 42`.

---

## 1. The synthetic ledger

### Why synthetic

Client data cannot be published. A public portfolio project therefore has two
options: use a small real dataset and be unable to demonstrate anything at scale, or
generate a ledger whose statistical properties match a real one. AuditLens does the
second, and is explicit that this is what it is.

### What is generated

| Entity | Rows | Notes |
|---|---|---|
| Transactions | 30,000 → 30,223 raw → **30,128 clean** | Jan 2024 – Jan 2026 |
| Vendors | 420 → 409 in the ledger | 50 registered recently (11.9%), 42 dormant (10.0%), 7 sharing a bank account |
| Employees | 140 | 13 roles with explicit `can_create` / `can_approve` rights |

Each voucher is a balanced double entry against the 25-account chart of accounts
(企业会计准则), with a coherent control timeline: **invoice received → approved →
paid → posted**. The timeline is built backwards from the posting date so that the
approval and payment timestamps can never precede the document that justifies them.

### What is realistic

- **Amounts are log-normally distributed**, which is what makes the Benford tests
  meaningful rather than decorative.
- **Seasonality and month-end clustering.** Posting volume peaks in the last days of
  each month, which is where real journal entry risk concentrates.
- **A working calendar.** Non-working days are shifted forward, leaving a small
  residual of genuine weekend postings (~2.3% of the ledger) rather than the 28% a
  naive uniform date sampler produces.
- **Vendor lifecycle.** A vendor can never be paid before it was registered. This
  sounds obvious and is exactly the kind of constraint that makes a synthetic ledger
  usable — without it, the "unusual vendor" rule fires on impossible data and its
  measured precision is meaningless.
- **Natural roundness.** Roughly one payment in eight is a round figure by
  coincidence, at a rate that depends on magnitude. A ledger where roundness only
  ever means "manual entry" would let a naive rule look far better than it is.

### What is not realistic

The injected anomalies follow clean, self-consistent patterns. A duplicate payment
is a duplicate of a real voucher; a split transaction sits in a tidy band just below
the approval threshold. Real irregularities do not announce themselves this way.
**Every detection metric in this project is therefore an upper bound**, and the
README says so rather than quoting the numbers as if they described live
performance.

---

## 2. Data quality

`src/data_cleaning.py` profiles the extract before touching it and reports every
issue it finds, whether or not it repairs it.

| Issue | Count | Action |
|---|---|---|
| Duplicate rows | 59 | Removed |
| Invalid transaction dates | 45 | Row dropped — a voucher with no date cannot be placed in a period |
| Invalid posting dates | 45 | Repaired from `transaction_date` |
| Invalid amounts (missing / ≤ 0) | 36 | Row dropped |
| Unbalanced vouchers (debit ≠ credit) | 45 | **Flagged, kept** |
| Orphan vendors (not in master file) | 36 | **Flagged, kept** |
| Account name / code mismatches | 30 | Repaired from the chart of accounts |
| Currency codes normalised | 34 | Aliases (`RMB`, `US$`) mapped to ISO 4217 |
| **Total** | **285** | 0.94% of the raw population |

### The two repairs, and why only two

A missing `posting_date` is rebuilt from `transaction_date` because the posting date
is, in practice, the transaction date or within a few days of it. An `account_name`
that disagrees with its `account_code` is restored from the chart of accounts
because the code is the key and the name is a description.

Everything else is flagged rather than fixed:

- **Unbalanced vouchers are kept.** Dropping them would understate the population,
  and the imbalance is itself the finding.
- **Orphan vendors are kept.** A vendor that exists in the ledger but not in the
  master file is a master-data control failure worth reporting, not a row to delete.
- **Missing narrations are never imputed.** This is the one that matters most. The
  suspicious-description rule scores a missing narration at 0.60 confidence. If the
  cleaner imputed a plausible description, it would erase the signal the rule exists
  to detect. A "helpful" imputation would silently disable a control.

---

## 3. The nine audit procedures

Each rule is independent, returns a score in `[0, 1]` and a human-readable reason for
every triggered voucher, and carries a weight used in the noisy-OR combination.
Measured performance is against the injected ground truth.

| Procedure | Threshold | Weight | Flagged | Precision | Recall |
|---|---|---|---|---|---|
| Duplicate Payment | same vendor + invoice + amount; or same vendor + amount within 5 days | 0.18 | 108 | 1.000 | 1.000 |
| Split Transaction | 2+ payments, same vendor, same day, each 90–99.5% of CNY 50,000 | 0.16 | 179 | 0.603 | 0.991 |
| Self Approval | `created_by == approved_by` | 0.14 | 126 | 1.000 | 1.000 |
| Unusual Vendor | new / dormant / shared bank account / amount spike, gated on ≥ P90 | 0.12 | 501 | 0.148 | 0.822 |
| Large Round Amount | multiple of 1,000 **and** ≥ P95 **and** ≥ CNY 100,000 | 0.10 | 505 | 0.214 | 1.000 |
| Rapid Payment | settled < 6 hours after invoice receipt | 0.08 | 90 | 1.000 | 1.000 |
| Suspicious Description | fraud-indicative keyword, or narration < 10 characters | 0.08 | 147 | 0.735 | 1.000 |
| Rare Account Usage | account frequency < 1%, escalated for the designated rare accounts | 0.08 | 54 | 1.000 | 1.000 |
| Weekend / Holiday Posting | transaction date on a weekend or a PRC public holiday | 0.06 | 765 | 0.165 | 1.000 |

**Totals:** 2,475 alerts across 2,208 distinct vouchers — **7.33% of the ledger**.
That is a reviewable population: large enough that nothing material is likely to be
missed, small enough that a team can actually work it.

Precision and recall here are measured **per pattern**, not against the 919 injected
anomalies as a whole: for a given rule the positive class is
`anomaly_type == rule_key`, which answers the question an auditor actually asks —
*when this rule fires, how often is it pointing at the pattern it claims to detect?*
Recall of 1.000 therefore means "every voucher injected with this pattern was
caught", not "every anomaly was caught". No rule does the latter.

### Combining the rules: noisy-OR, not a vote

Independent indicators are combined as

```
combined = 1 − Π (1 − weight_i × score_i)
```

Two weak indicators pointing the same way produce a stronger signal than either
alone; a single strong indicator is not diluted by the rules that stayed silent.
A majority vote would discard exactly the evidence an auditor cares about most.

### Where the first attempt was wrong

These are recorded because they are the substance of the work.

**The unusual-vendor rule flagged 61% of the ledger.** The root cause was not the
rule: the engine was loading the *clean* transaction table rather than the
feature-enriched one, so every engineered feature silently defaulted to `0.0`, and
"days since the vendor registered" was zero for everyone. The fix was structural —
the engine now detects missing required features and rebuilds the whole feature set
rather than defaulting them. A second cause was that `vendor_master_risk_score` was
being used as a trigger; it is a static attribute, so it flagged the same vendors on
every run regardless of behaviour. It was removed as a trigger and the signals were
gated on amounts above the 90th percentile. Precision is still 0.148, and that is
the honest answer: **the rule is scoped as a population selector, not a detector**,
and the README reads it that way.

**The rapid-payment rule flagged 48% of the ledger.** `payment_delay_hours` was
being imputed with `0.0` inside feature engineering, so every voucher without an
invoice looked like it had been paid instantly. Imputation now happens at model-matrix
time with the median, and the rule requires the feature to be present.

**The suspicious-description rule ran at precision 0.021.** The culprit was the
boilerplate signal: system-generated journals legitimately repeat identical narration
across thousands of vouchers, covering 17% of the ledger. Repetition is a real
signal in some contexts and useless here, so it is now reported as
`description_repetition_flag` for the auditor to look at and is **not allowed to
drive the score**. Precision rose to 0.735.

**The Benford test showed nonconformity.** The ledger's amounts were drawn from a
log-normal with too narrow a spread, which concentrates leading digits and is a
genuine nonconformity rather than a bug. Widening the distribution to
`sigma = (log(high) − log(low)) / 3` and reducing the rate of naturally round amounts
brought the first-digit MAD to 0.0022 — close conformity. The lesson is that
Benford conformity is a property of the data, not something to be tuned away.

**The first-two-digits test had an inert flag column.** It marked a bucket as
significant when `|deviation| > 2 / sqrt(n)`. That is a fixed *absolute* threshold
applied to 90 buckets whose expected frequencies differ by an order of magnitude:
at n = 30,128 the threshold is 0.0115, while digit 99's expected frequency is 0.0044
and the median bucket's is 0.0078. Two thirds of the buckets could not reach the
threshold even if they were empty, so the column was dead. It now uses the same
per-bucket z-score as the first-digit test, which is the comparison the test
intended. Thirteen of the 90 buckets exceed |z| > 1.96 — more than the ~4.5 expected
by chance, which is worth knowing, but it is a statement about multiple testing
across 90 buckets, not evidence of irregularity. The MAD stays inside close
conformity, and the two measures answer different questions.

**The generator was not reproducible.** The seeded generator produced a different
ledger on every run. `rng.choice` samples by *index*, and one call site was sampling
from `list(some_set)` — a set of vendor-id strings. String set iteration order
depends on `PYTHONHASHSEED`, which differs between processes, so the same random
index landed on a different vendor each time. Two consecutive pipeline runs
disagreed on the vendor count (408 vs 409), the alert count, and the Isolation
Forest's decision threshold. The whole test suite passed throughout, because an
in-process test shares one hash seed and cannot see it; it surfaced only by diffing
two pipeline runs by hand. The pool is now a sorted list, and
`tests/test_reproducibility.py` runs the generator in two subprocesses with
different hash seeds and compares ledger hashes — a guard that was verified to fail
when the fix is reverted.

**The notebooks were committed with no outputs.** All 23 code cells across the three
notebooks contained zero output. `tools/build_notebooks.py` executed every cell to prove
the notebook ran, then wrote the `.ipynb` with the results discarded — so the build
reported success and the committed file rendered on GitHub as code with nothing
underneath it. Nothing failed, because a notebook with empty outputs is a perfectly
valid file. The builder now captures and embeds what each cell produced: stdout, the
`display()` values as HTML tables, and the matplotlib figures as inline PNGs.

Embedding the output then exposed two sources of instability, which mattered because
they made the committed notebooks differ on *every* build:

* **pandas mints a random table id per `Styler` render.** `Styler.to_html()` stamps its
  `<table>` and every `<th>`/`<td>` inside it with a `uuid4`-derived id, so the same table
  rendered as `T_3b175` in one process and `T_9f01c` in the next. The builder rewrites
  those ids to a deterministic sequence. The replacement deliberately contains a `z`
  (`T_z0000`), because an all-digit id such as `T_00000` is itself valid hex and would
  match the pattern used to detect leaked random ids — the check could never have failed.
* **The loggers wrote a wall-clock timestamp into the capture buffer.** `get_logger`
  builds its `StreamHandler` against whatever `sys.stdout` is live when the logger is
  first constructed. During a notebook build that is the per-cell capture buffer, so a
  `13:16:49 | INFO | src.reporting | Chart font: ...` record landed in the notebook and
  changed on every run. Worse, records from later cells were written into a buffer that
  was no longer being read and were silently lost. INFO logging is now disabled for the
  duration of the build; warnings and errors still surface.

`tests/test_notebooks.py` guards all of this: every cell that prints or plots must carry
output, every notebook must embed at least one chart, and no random table id or logged
timestamp may survive into a committed notebook. The `display()` shim is pinned too — it
was originally `list.extend`, which takes an *iterable* of things to display, so
`display(styler)` raised and `display(frame)` silently recorded the frame's **column
names** in place of the frame. A wrong table, produced with no error.

**The guard against a stale test count could not detect one.** `tests/test_documentation.py`
was written because the README's test count had drifted repeatedly (302 → 331 → 351). It
compared the testing table against the badge, and deliberately did *no* pytest
introspection so that adding a test could not break it. That choice removed its ability
to do the job it was written for. Four new tests took two files from 55 to 57 and 18 to
20 while the badge still read 374, and every check stayed green — the table and the badge
agreed with each other, and neither was ever compared to the number of tests pytest
actually collects. It was caught by hand, in the same session, by reading the collection
output.

The file now runs `pytest --collect-only -q` in a subprocess and compares every row and
the total against it, keeping the text checks as well. The subprocess is the load-bearing
detail: reading `request.session.items` would make the answer depend on how the file was
invoked, so running it on its own would compare the table against a single file and fail
for a reason unrelated to the documentation. The new check immediately caught the next
drift — its own three tests.

---

## 4. Benford's Law

### What is tested

The **debit leg only**. Each voucher is stored once with `debit_amount ==
credit_amount`, so testing both legs would double-count every value. Values below
CNY 10 are excluded, because a small number's leading digit is not distributed the
way Benford's Law describes.

### Results

| Test | Observations | MAD | χ² | df | p | Conformity |
|---|---|---|---|---|---|---|
| First digit | 30,128 | **0.00218** | 16.00 | 8 | 0.042 | Close conformity |
| First two digits | 30,128 | **0.00071** | 196.09 | 89 | < 0.001 | Close conformity |

MAD (mean absolute deviation between observed and expected frequency) is the primary
measure, classified using the bands in Nigrini's work on digital analysis: below
0.006 is close conformity, 0.006–0.012 acceptable, 0.012–0.015 marginal, above 0.015
nonconformity. The χ² test is reported alongside it but is not the deciding measure:
with 30,000 observations, χ² is powerful enough to detect deviations of no audit
significance, which is why the two tests can disagree and MAD wins.

The two tests disagree here, and the disagreement is instructive rather than a
defect. The first-two-digits χ² is 196.09 on 89 degrees of freedom (p ≈ 5e-10),
which reads as a strong rejection, while the MAD is 0.00071 — deep inside close
conformity. The reason is that MAD measures the *average magnitude* of the
deviation, which is negligible, whereas χ² accumulates 90 individually tiny
deviations into a large statistic. With this many observations, χ² will flag
distributions whose departures are far too small to matter. The per-bucket z-scores
show the same thing: 13 buckets exceed |z| > 1.96 against roughly 4.5 expected by
chance, with a largest deviation of about four standard errors. That is worth a
sentence in a file note; it is not a finding, and it is not a reason to widen
testing. The MAD is the measure that answers the audit question.

### Disaggregation is the point

A population-level MAD of 0.002 tells you nothing actionable. The useful output is
the per-account breakdown, which is where an enquiry actually starts. On this ledger
several accounts are nonconforming at account level while the population as a whole
looks fine — deviations in opposite directions cancel out in the aggregate.

### What Benford's Law cannot do

**It cannot prove fraud, and it cannot clear anyone.** This is stated on the
dashboard, in the report, and here:

- A **conforming** distribution is entirely compatible with fabricated entries. A
  competent fraudster who understands the test can pass it.
- A **nonconforming** distribution is usually a legitimately skewed account. Payroll,
  fixed asset additions, and any account with a narrow range of plausible values
  fail Benford for innocent reasons.
- The test operates on a **population**. It cannot say "this voucher is odd"; it can
  only say "this account's digits depart from expectation".

The correct use is as a **disaggregation tool**: it tells you which sub-population
deserves a closer look. That is how the composite risk score treats it — the
statistical component carries a weight of 0.10, the smallest of the five.

---

## 5. Isolation Forest

### Why unsupervised

On a real engagement nobody hands you a column marked "fraud". A supervised model
would require labels that do not exist, and training on injected labels would produce
a model that had learned the generator's patterns rather than anything about
irregularity. Isolation Forest is the right family: it needs no labels, it scales,
and its output is a continuous score that can be cut at any point.

### The 20 features

All of them are expressible as an audit question, which is the test for inclusion:

`amount`, `log_amount`, `amount_percentile`, `vendor_transaction_count`,
`vendor_average_amount`, `vendor_total_amount`, `vendor_amount_ratio`,
`vendor_max_amount`, `vendor_age_days`, `days_since_last_transaction`,
`account_transaction_frequency`, `is_weekend`, `is_round_amount`,
`days_since_vendor_registration`, `approval_time_hours`, `payment_delay_hours`,
`same_vendor_daily_transactions`, `description_length`, `is_self_approval`,
`near_approval_threshold`.

**Deliberately excluded:**

- **`vendor_master_risk_score`** — a static attribute. Including it makes the model
  flag the same vendors every run regardless of behaviour. Behaviour belongs in a
  model; a risk rating belongs in a rule.
- **All identifiers** — vendor ID, employee ID, account code, transaction ID. A model
  that memorises IDs does not generalise, and "this vendor ID looks unusual" is not
  an actionable finding.
- **Any ground-truth-derived column.** `anomaly_label` and `anomaly_type` are never
  inputs.

### Configuration

`IsolationForest(n_estimators=300, contamination=0.03, random_state=42)` over a
`StandardScaler`-transformed matrix. Isolation Forest does not require scaling, but
the scaler is applied anyway so the model matrix is reproducible and portable.

`contamination = 0.03` is a **planning assumption** about how much of the ledger is
worth reviewing, not a leak from the answer key. A real engagement has no answer key
and the model would be run identically.

Four features are undefined for some vouchers — a vendor's first payment has no
"days since last transaction". They are imputed with the population median, and the
medians used are recorded in `model_metrics.json` so the matrix can be reproduced
exactly.

### Evaluation

| Metric | Value |
|---|---|
| Precision | 0.291 |
| Recall | 0.286 |
| F1 | 0.289 |
| ROC-AUC | **0.816** |
| Average precision | 0.201 (vs 0.031 random baseline) |
| True positives / false negatives | 263 / 656 |
| Vouchers raised that no rule raised | 244 (of which **2** are real anomalies) |

### The model is close to redundant on this benchmark, and that is the finding

Measured at the **anomaly** level rather than the flag level:

| | Anomalies | Share of 919 |
|---|---|---|
| Caught by a rule | **902** | 98.2% |
| Caught by the model | 263 | 28.6% |
| Caught by both | 261 | 28.4% |
| **Caught by the model only** | **2** | 0.2% |
| Caught by a rule only | 641 | 69.7% |
| Caught by neither | 15 | 1.6% |

The rules are near-exhaustive because the anomalies were **injected to match the rule
definitions**. That makes the comparison circular: a benchmark generated from a rule set
cannot measure what a model adds beyond that rule set, and any incremental detector will
look useless on it.

This is worth stating in the strongest terms because an earlier version of this
project's own summary reported `detections_not_caught_by_rules = 244`, which reads as
"244 anomalies the rules missed" and is wrong: 242 of those 244 are false positives.
The summary now reports the overlap on both bases, and the flag-level and anomaly-level
numbers are named so they cannot be confused.

What the model does establish is that it ranks well with no labels: ROC-AUC 0.816 and
47% precision at the top 0.5% of the ranking. Its real argument — generalisation to
patterns nobody wrote a rule for — is untestable here and is not claimed.

### Precision and recall against the review budget

The metrics above are at a fixed cut point. What an audit team actually chooses is
how many vouchers it can work:

| Vouchers reviewed | Precision | Recall |
|---|---|---|
| 151 (top 0.5%) | 47.0% | 7.7% |
| 301 (top 1%) | 38.2% | 12.5% |
| 603 (top 2%) | 33.3% | 21.9% |
| 1,506 (top 5%) | 20.7% | 34.0% |
| 3,013 (top 10%) | 14.5% | 47.4% |

The top 0.5% achieves **15× lift** over random selection. Read the table from left to
right and the trade-off is explicit: a lower threshold lifts recall and buries the
team in false positives; a higher one cleans the list and quietly drops real
findings.

**Recall matters more than precision in audit** — a missed misstatement is an audit
failure, whereas a false positive is a wasted hour. But recall is not free, and
pretending otherwise is how analytics projects lose the confidence of the people who
have to work the output. The defensible position is to state the budget, state the
recall it buys, and let the engagement partner decide.

---

## 6. The Audit Risk Score

### Components and weights

| Component | Weight | Source |
|---|---|---|
| Rule indicators | **0.40** | Noisy-OR of the nine procedures |
| Isolation Forest | **0.25** | Normalised anomaly score |
| Vendor risk | **0.15** | Counterparty score, rescaled from 0–100 |
| Amount context | **0.10** | Size percentile, roundness, materiality |
| Statistical (Benford) | **0.10** | Account-level digit departure |

The score is `100 × Σ (weight × component)`, clipped to `[0, 100]`.

### Why these weights

The specification suggested 0.45 / 0.30 / 0.15 / 0.10 and explicitly said not to copy
it mechanically. The final allocation follows three arguments:

1. **Rules outweigh the model (0.40 vs 0.25).** A rule is a control an auditor can
   read, re-perform and challenge. A model is a ranking. When the two disagree, the
   auditable artefact should carry more weight — and the model still has enough
   weight to move a voucher that no rule caught, which is the entire reason it is
   there.
2. **Vendor risk is real but indirect (0.15).** A supplier's overall pattern is
   relevant to whether a specific payment deserves attention, but it is one step
   removed from the voucher itself.
3. **Amount and statistical context are weak evidence (0.10 each).** Being large is
   not misconduct, and Benford's Law cannot say anything about an individual voucher.
   They adjust the score; they should not drive it. Note that the *size* term inside
   the amount component is the percentile rank **above the median**, so a
   median-sized voucher scores zero rather than 0.5 — otherwise every voucher would
   start halfway up the scale for no reason.

The amount component is 70% size / 20% roundness / 10% materiality, deliberately
restrained for the same reason.

### Bands

| Band | Score | Meaning | Vouchers | Value |
|---|---|---|---|---|
| Low | 0–30 | No indicator fired; reviewed in aggregate | 28,261 | ¥3.45B |
| Medium | 30–60 | One indicator, or a moderate model score | 1,612 | ¥723.0M |
| High | 60–80 | Multiple indicators, or one on a material amount | 254 | ¥265.8M |
| Critical | 80–100 | Several independent indicators on a material amount | 1 | ¥1.25M |

Average score across the population: **14.64**. High and Critical together are
**255 vouchers (0.85%)**, carrying ¥267.0M.

A well-calibrated triage pushes almost everything into Low. If half the ledger were
High, the score would not be discriminating — it would just be relabelling the
population.

### Explainability

Every voucher carries `risk_reasons`: an ordered list of plain-English statements
naming the indicator, the value that breached it, and the threshold it breached.
The order follows the weight of the component that produced each reason, so the
strongest evidence reads first, and the list is capped so the output stays readable.

A machine-learning flag is **never emitted on its own**. It is always accompanied by
the features that drove it, because "the model said so" is not an audit explanation.
`tests/test_risk_scoring.py` asserts that every High or Critical voucher has at least
one reason — a score with no explanation is treated as a bug.

---

## 7. Audit review selection

The default workpaper budgets 300 vouchers. Selection is made only from scored
transactions and rule alerts; `anomaly_label` and `anomaly_type` are absent from
every decision field and from the auditor-facing CSV.

1. Reserve 20% of the budget for random controls (60 vouchers).
2. Select the case covering the most as-yet-unrepresented rule types until all
   nine procedures are represented or the targeted budget is exhausted. Ties go
   to higher risk score, then amount, then voucher ID. Six cases cover all nine
   procedures in the committed sample.
3. Fill the remaining 234 targeted slots by risk score, amount and voucher ID.
4. Uniformly draw 60 without replacement from the 29,888 *remaining* vouchers
   using seed 42. Sorting IDs before drawing makes input row order irrelevant.

The random-route inclusion probability is $p=60/29{,}888=0.0020075$ and its
sampling weight is $1/p$. These describe only the non-targeted sampling frame.
No population exception estimate exists until an auditor determines outcomes
from source documents. The targeted route has no design-based sampling weight.

`review_plan_summary.json` records the policy, rule coverage, selection-input
fingerprint and CSV checksum. `review_plan_benchmark.json` is a separate
**synthetic-only** diagnostic computed after selection: 104 injected anomalies
are in the workpaper, compared with 137 under pure risk ranking at the same
budget. The 60 controls cost some immediate benchmark yield in exchange for a
way to inspect the remainder. This does not measure live audit performance.

---

## 8. Limitations

Stated plainly, because a portfolio project that overclaims is worse than one that
does less.

1. **The data is synthetic.** The anomalies were injected by a generator whose
   patterns are known to the detectors. Every precision, recall and lift figure in
   this document is an upper bound on real-world performance, and probably a generous
   one.
2. **The thresholds are conventions, not calibrations.** CNY 50,000 for approval,
   CNY 100,000 for materiality, 6 hours for rapid payment, 90 days for a new vendor.
   A real engagement sets these from the client's own control environment and
   materiality. They are documented in `src/utils.py` and are meant to be changed.
3. **Low-precision rules are scope, not detection.** Unusual Vendor (0.148) and
   Weekend / Holiday Posting (0.165) define populations to look at. They are useful
   as scoping procedures; they are not evidence about any individual voucher.
4. **Recall of 1.000 on several rules is a benchmark artefact.** The injected
   anomalies follow exactly the pattern the rule tests for, so a rule tuned on that
   pattern finds all of them. It says nothing about a real ledger.
5. **The benchmark cannot measure what the model adds.** The rules catch 902 of 919
   injected anomalies (98.2%) and the model contributes **2** that no rule caught. That
   is not evidence the model is useless; it is evidence that a benchmark generated from
   a rule set is circular with respect to that rule set. Measuring the model's
   incremental value would need either a real labelled engagement or a held-out anomaly
   family the rules were never written against. The project does not claim otherwise,
   and the flag-level overlap (244) is reported separately so it cannot be mistaken for
   a detection count.
6. **Benford's Law cannot detect fraud.** See §4. It identifies populations that
   deserve disaggregation.
7. **The model is not tuned, and should not be.** Hyperparameters are defaults chosen
   for reproducibility, not optimised against the injected labels — tuning them would
   make the reported metrics even more of an upper bound.
8. **No temporal validation.** The ledger is split into periods for reporting, but
   the model is not trained on one period and tested on another. A real deployment
   would need to check that behaviour is stable over time.
9. **Single currency, single entity.** Multi-entity consolidation, intercompany
   elimination and transfer pricing are out of scope.

---

## 9. Reproducing every number

```bash
pip install -r requirements.txt
python src/run_pipeline.py     # ~30 seconds
python -m pytest tests/ -q
```

The run is deterministic: `RANDOM_SEED = 42` governs the generator, the Isolation
Forest, and the sampling, and two consecutive runs produce byte-identical reports
under `outputs/reports/`. That property is enforced by
`tests/test_reproducibility.py` rather than assumed. The numbers in this document are
the numbers in `outputs/reports/audit_summary.json` from that run.
