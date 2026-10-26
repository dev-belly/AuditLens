# Security reporting

## Scope

Report reproducible security concerns against the current default branch, including the commit SHA and affected component. Historical experiments and old snapshots are retained for reference; this document does not promise security support for any release or response time.

## Report privately when a channel is available

Open the repository's [Security tab](https://github.com/dev-belly/AuditLens/security) and check whether **Report a vulnerability** is available. If it is available, use that private reporting channel. This policy does not claim that private vulnerability reporting is enabled.

If no private reporting option is available, do not publish exploit steps, credentials, confidential data or a working exploit in a public issue or pull request. You may open a general [contact request](https://github.com/dev-belly/AuditLens/issues/new) asking for a private reporting channel, without including sensitive technical details. Wait for an agreed private channel before sharing them.

A useful private report contains the affected commit and environment, minimal reproduction, impact and any proposed mitigation. Use synthetic or public fixtures wherever possible; redact tokens and identifying records. Do not test against third-party services or accounts without authorization.

## Project-specific data handling

Run the pipeline before opening the Streamlit dashboard or testing generated workpapers. Prefer a small generated ledger for a reproduction. Real ledger extracts, vendor bank details, employee identities, reviewer notes and audit evidence must stay out of public reports.

Include the generator seed, pipeline stage, a synthetic voucher example, and the affected rule or workpaper field. Distinguish rule flags, review selections, and injected labels; an alert is not a confirmed fraud finding.

Security handling is best effort. Do not send passwords, access tokens, customer records or unredacted private datasets through any repository form.
