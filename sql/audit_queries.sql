-- ============================================================================
-- AuditLens - analyst query library
-- ============================================================================
-- Fourteen queries an audit team would actually run against the warehouse.
-- Each statement is named with a `-- name:` marker so that `src/database.py`
-- can execute the file and label the results.
--
-- Conventions
--   * Every monetary figure uses `debit_amount`. Each voucher is stored once,
--     with debit_amount equal to credit_amount, so the debit leg is the voucher
--     value. Summing `amount` as well would double count.
--   * Ground-truth columns (anomaly_label, anomaly_type) are deliberately not
--     referenced. These are auditor-facing queries; the benchmark exists only to
--     evaluate the model.
--   * Dates are stored as ISO text, so strftime() and string comparison both work.
-- ============================================================================


-- name: top_vendors_by_payment_amount
-- Where the money went: the twenty largest suppliers by total value paid.
SELECT
    v.vendor_id,
    v.vendor_name,
    v.vendor_category,
    v.risk_level                                    AS master_risk_rating,
    COUNT(*)                                        AS transaction_count,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    ROUND(AVG(t.debit_amount), 2)                   AS average_amount,
    ROUND(MAX(t.debit_amount), 2)                   AS largest_amount,
    ROUND(v.vendor_risk_score, 2)                   AS vendor_risk_score
FROM transactions AS t
JOIN vendors AS v ON v.vendor_id = t.vendor_id
GROUP BY v.vendor_id, v.vendor_name, v.vendor_category, v.risk_level, v.vendor_risk_score
ORDER BY total_amount DESC
LIMIT 20;


-- name: duplicate_invoice_payments
-- The same invoice, the same amount, the same supplier, paid more than once.
-- This is the highest-value finding in accounts payable because the money is
-- recoverable if it is caught.
SELECT
    t.vendor_id,
    t.vendor_name,
    t.invoice_id,
    ROUND(t.debit_amount, 2)                        AS amount,
    COUNT(*)                                        AS payment_count,
    ROUND(SUM(t.debit_amount), 2)                   AS total_paid,
    ROUND(SUM(t.debit_amount) - MIN(t.debit_amount), 2) AS overpayment,
    GROUP_CONCAT(t.transaction_id, ', ')            AS transaction_ids,
    MIN(t.transaction_date)                         AS first_payment,
    MAX(t.transaction_date)                         AS last_payment
FROM transactions AS t
WHERE t.invoice_id IS NOT NULL
  AND t.vendor_id IS NOT NULL
GROUP BY t.vendor_id, t.vendor_name, t.invoice_id, ROUND(t.debit_amount, 2)
HAVING COUNT(*) > 1
ORDER BY overpayment DESC;


-- name: weekend_and_holiday_postings
-- Journal activity outside the working week. A weak signal on its own, which is
-- exactly why it is worth combining with the amount.
SELECT
    t.transaction_date,
    CASE CAST(strftime('%w', t.transaction_date) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 6 THEN 'Saturday'
        ELSE 'Weekday'
    END                                             AS day_type,
    COUNT(*)                                        AS transaction_count,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    ROUND(AVG(t.audit_risk_score), 2)               AS average_risk_score,
    SUM(CASE WHEN t.risk_level IN ('High', 'Critical') THEN 1 ELSE 0 END) AS high_risk_count
FROM transactions AS t
WHERE CAST(strftime('%w', t.transaction_date) AS INTEGER) IN (0, 6)
GROUP BY t.transaction_date
ORDER BY total_amount DESC
LIMIT 25;


-- name: monthly_transaction_totals
-- Volume and value by month, with the share of that month flagged for review.
SELECT
    strftime('%Y-%m', t.transaction_date)           AS year_month,
    COUNT(*)                                        AS transaction_count,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    ROUND(AVG(t.debit_amount), 2)                   AS average_amount,
    SUM(CASE WHEN t.rule_alert_count > 0 THEN 1 ELSE 0 END) AS flagged_count,
    ROUND(100.0 * SUM(CASE WHEN t.rule_alert_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS flagged_pct,
    SUM(CASE WHEN t.risk_level IN ('High', 'Critical') THEN 1 ELSE 0 END) AS high_risk_count
FROM transactions AS t
GROUP BY year_month
ORDER BY year_month;


-- name: high_risk_vendors
-- Suppliers whose risk score places them in the enhanced-monitoring tier.
SELECT
    v.vendor_id,
    v.vendor_name,
    v.vendor_category,
    v.country,
    v.registration_date,
    v.risk_level                                  AS master_risk_rating,
    v.vendor_transaction_count                    AS transaction_count,
    ROUND(v.vendor_total_amount, 2)               AS total_amount,
    v.alert_count                                 AS rule_alerts,
    ROUND(v.vendor_risk_score, 2)                 AS vendor_risk_score,
    v.vendor_risk_level                           AS vendor_risk_level,
    CASE WHEN v.shared_bank_account = 1 THEN 'YES' ELSE 'no' END AS shared_bank_account
FROM vendors AS v
WHERE v.vendor_risk_score >= 45
ORDER BY v.vendor_risk_score DESC, v.vendor_total_amount DESC;


-- name: accounts_with_unusual_activity
-- Accounts whose activity profile departs from the rest of the ledger: unusual
-- volume, unusual value, or an elevated concentration of high-risk vouchers.
SELECT
    t.account_code,
    t.account_name,
    COUNT(*)                                        AS transaction_count,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    ROUND(AVG(t.debit_amount), 2)                   AS average_amount,
    ROUND(MAX(t.debit_amount), 2)                   AS largest_amount,
    SUM(CASE WHEN t.rule_alert_count > 0 THEN 1 ELSE 0 END) AS flagged_count,
    ROUND(100.0 * SUM(CASE WHEN t.rule_alert_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS flagged_pct,
    ROUND(AVG(t.audit_risk_score), 2)               AS average_risk_score,
    ROUND(MAX(t.benford_mad), 4)                    AS account_benford_mad
FROM transactions AS t
GROUP BY t.account_code, t.account_name
HAVING flagged_pct > 5 OR account_benford_mad > 0.015
ORDER BY average_risk_score DESC;


-- name: transactions_near_approval_threshold
-- Payments sitting in the 10% band below the CNY 50,000 approval threshold.
-- A cluster of these on the same vendor and day is a split-payment indicator.
SELECT
    t.vendor_id,
    t.vendor_name,
    t.transaction_date,
    COUNT(*)                                        AS payments_in_band,
    ROUND(SUM(t.debit_amount), 2)                   AS combined_amount,
    ROUND(MIN(t.debit_amount), 2)                   AS smallest_payment,
    ROUND(MAX(t.debit_amount), 2)                   AS largest_payment,
    CASE WHEN COUNT(*) >= 2 AND SUM(t.debit_amount) > 50000
         THEN 'SPLIT PATTERN' ELSE 'single payment' END AS assessment,
    GROUP_CONCAT(t.transaction_id, ', ')            AS transaction_ids
FROM transactions AS t
WHERE t.debit_amount BETWEEN 45000 AND 49750
GROUP BY t.vendor_id, t.vendor_name, t.transaction_date
ORDER BY payments_in_band DESC, combined_amount DESC
LIMIT 30;


-- name: self_approved_transactions
-- Segregation-of-duties breaches: the employee who raised the voucher approved it.
SELECT
    t.created_by,
    e.employee_name,
    e.department,
    e.role,
    COUNT(*)                                        AS transaction_count,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    ROUND(AVG(t.audit_risk_score), 2)               AS average_risk_score,
    SUM(CASE WHEN t.risk_level IN ('High', 'Critical') THEN 1 ELSE 0 END) AS high_risk_count,
    MIN(t.transaction_date)                         AS first_transaction,
    MAX(t.transaction_date)                         AS last_transaction
FROM transactions AS t
LEFT JOIN employees AS e ON e.employee_id = t.created_by
WHERE t.created_by = t.approved_by
GROUP BY t.created_by, e.employee_name, e.department, e.role
ORDER BY total_amount DESC;


-- name: vendor_spend_concentration
-- How concentrated the supplier base is. High concentration is a going-concern
-- and dependency question as much as a fraud question.
SELECT
    v.vendor_id,
    v.vendor_name,
    ROUND(v.vendor_total_amount, 2)                 AS total_amount,
    ROUND(100.0 * v.vendor_total_amount / (SELECT SUM(debit_amount) FROM transactions), 3) AS pct_of_total_spend,
    v.vendor_transaction_count                      AS transaction_count
FROM vendors AS v
WHERE v.vendor_total_amount > 0
ORDER BY total_amount DESC
LIMIT 15;


-- name: vendors_sharing_bank_accounts
-- Two suppliers on the same bank account is a classic shell-company indicator.
-- Legitimate explanations exist (group treasury, rebranding) - which is why this
-- is a question for the auditor, not a conclusion.
SELECT
    v.bank_account,
    COUNT(*)                                        AS vendor_count,
    GROUP_CONCAT(v.vendor_name, ' | ')              AS vendors,
    ROUND(SUM(v.vendor_total_amount), 2)            AS combined_amount,
    ROUND(MAX(v.vendor_risk_score), 2)              AS highest_vendor_risk_score
FROM vendors AS v
WHERE v.bank_account IS NOT NULL
GROUP BY v.bank_account
HAVING COUNT(*) > 1
ORDER BY combined_amount DESC;


-- name: dormant_vendor_reactivation
-- Suppliers registered long ago that received their first payment of the period
-- late on, or that were paid only once. Both shapes are used to move money
-- through an entity nobody is watching.
SELECT
    v.vendor_id,
    v.vendor_name,
    v.registration_date,
    v.vendor_transaction_count                      AS lifetime_transactions,
    ROUND(v.vendor_total_amount, 2)                 AS lifetime_amount,
    v.risk_level                                    AS master_risk_rating,
    ROUND(v.vendor_risk_score, 2)                   AS vendor_risk_score
FROM vendors AS v
WHERE v.vendor_transaction_count <= 3
  AND v.vendor_total_amount > 100000
ORDER BY v.vendor_total_amount DESC;


-- name: payment_speed_by_vendor_category
-- Average invoice-to-payment cycle by supplier category. Paying far faster than
-- terms is how a payment bypasses the normal matching and review cycle.
SELECT
    v.vendor_category,
    COUNT(*)                                        AS payment_count,
    ROUND(AVG(t.payment_delay_hours), 1)            AS average_hours_to_pay,
    ROUND(MIN(t.payment_delay_hours), 1)            AS fastest_hours,
    SUM(CASE WHEN t.payment_delay_hours < 6 THEN 1 ELSE 0 END) AS paid_within_6_hours,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount
FROM transactions AS t
JOIN vendors AS v ON v.vendor_id = t.vendor_id
WHERE t.payment_delay_hours IS NOT NULL
GROUP BY v.vendor_category
ORDER BY average_hours_to_pay;


-- name: month_end_posting_concentration
-- The share of each month's value posted in its final three days. A spike at the
-- period end is where manual adjustments and cut-off errors tend to hide.
SELECT
    strftime('%Y-%m', t.transaction_date)           AS year_month,
    COUNT(*)                                        AS total_transactions,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    SUM(CASE WHEN CAST(strftime('%d', t.transaction_date) AS INTEGER)
                  > CAST(strftime('%d', date(t.transaction_date, 'start of month', '+1 month', '-1 day')) AS INTEGER) - 3
             THEN 1 ELSE 0 END)                     AS last_three_days_count,
    ROUND(SUM(CASE WHEN CAST(strftime('%d', t.transaction_date) AS INTEGER)
                       > CAST(strftime('%d', date(t.transaction_date, 'start of month', '+1 month', '-1 day')) AS INTEGER) - 3
                   THEN t.debit_amount ELSE 0 END), 2) AS last_three_days_amount,
    ROUND(100.0 * SUM(CASE WHEN CAST(strftime('%d', t.transaction_date) AS INTEGER)
                               > CAST(strftime('%d', date(t.transaction_date, 'start of month', '+1 month', '-1 day')) AS INTEGER) - 3
                           THEN t.debit_amount ELSE 0 END) / SUM(t.debit_amount), 2) AS last_three_days_pct
FROM transactions AS t
GROUP BY year_month
ORDER BY year_month;


-- name: rule_alert_summary
-- How many alerts each rule produced, the value they touch, and how much of the
-- ledger each rule sweeps up. This is the auditor's workload estimate.
SELECT
    a.rule_key,
    a.rule_label,
    COUNT(*)                                        AS alert_count,
    COUNT(DISTINCT a.transaction_id)                AS distinct_transactions,
    ROUND(SUM(t.debit_amount), 2)                   AS amount_affected,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 3) AS pct_of_ledger,
    ROUND(AVG(a.rule_score), 4)                     AS average_rule_score,
    ROUND(AVG(t.audit_risk_score), 2)               AS average_risk_score
FROM audit_alerts AS a
JOIN transactions AS t ON t.transaction_id = a.transaction_id
GROUP BY a.rule_key, a.rule_label
ORDER BY amount_affected DESC;


-- name: risk_level_summary
-- The triage outcome: how many vouchers fall into each risk band and what value
-- they represent. This is the first table on the Executive Overview page.
SELECT
    t.risk_level,
    COUNT(*)                                        AS transaction_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 3) AS pct_of_transactions,
    ROUND(SUM(t.debit_amount), 2)                   AS total_amount,
    ROUND(100.0 * SUM(t.debit_amount) / (SELECT SUM(debit_amount) FROM transactions), 3) AS pct_of_value,
    ROUND(AVG(t.audit_risk_score), 2)               AS average_risk_score,
    ROUND(MAX(t.audit_risk_score), 2)               AS maximum_risk_score
FROM transactions AS t
GROUP BY t.risk_level
ORDER BY CASE t.risk_level
    WHEN 'Critical' THEN 1
    WHEN 'High'     THEN 2
    WHEN 'Medium'   THEN 3
    ELSE 4
END;
