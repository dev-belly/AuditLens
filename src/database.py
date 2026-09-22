"""SQLite data warehouse for AuditLens.

The pipeline writes its results into a single-file SQLite database so that the
findings can be interrogated with SQL - which is how audit analytics is actually
delivered inside an audit team. Excel exports are fine for one analyst; a queryable
warehouse is what lets the whole engagement team ask their own questions.

Schema
------
``transactions``
    One row per voucher, carrying the rule flags, the ML anomaly score and the
    final Audit Risk Score. The ground-truth columns (``anomaly_label``,
    ``anomaly_type``) are written too, because the benchmark is needed to
    reproduce the model evaluation - the dashboard simply never selects them.
``vendors``
    Vendor master data joined to the aggregated vendor risk score.
``employees``
    Employee master data.
``audit_alerts``
    Long-format rule alerts: one row per (voucher, rule) pair. This is the table
    an auditor filters when working through a rule.
``risk_summary``
    Headline metrics, one row, for quick reference.

Usage::

    python src/database.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allows `python src/database.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils import (
    AUDIT_SUMMARY_REPORT,
    DB_PATH,
    EMPLOYEES_CLEAN,
    RULE_ALERTS_CSV,
    SCORED_TRANSACTIONS,
    SQL_DIR,
    Timer,
    VENDOR_RISK_TABLE,
    ensure_directories,
    get_logger,
    load_dataframe,
    load_json,
)

LOGGER = get_logger(__name__)

#: Tables written to the warehouse and the primary key of each.
TABLE_SCHEMA: dict[str, str] = {
    "transactions": "transaction_id",
    "vendors": "vendor_id",
    "employees": "employee_id",
    "audit_alerts": "alert_id",
}

#: Columns that must not be selected by auditor-facing queries. They exist in the
#: warehouse for reproducibility of the model evaluation, not for decision-making.
GROUND_TRUTH_COLUMNS: tuple[str, ...] = ("anomaly_label", "anomaly_type")


def get_engine(db_path: Path = DB_PATH) -> Engine:
    """Create a SQLAlchemy engine for the AuditLens warehouse."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


#: Column types SQLite cannot bind. ``risk_reasons`` / ``rule_reasons`` are the
#: real cases: they hold per-voucher explanation lists. Note that a parquet
#: round-trip converts those Python lists into ``numpy.ndarray``, so checking for
#: ``list`` alone would silently miss them - hence the explicit type tuple.
_NON_SCALAR_TYPES: tuple[type, ...] = (list, tuple, set, dict, np.ndarray)


def _is_non_scalar(series: pd.Series) -> bool:
    """Return ``True`` when a column holds containers rather than scalar values."""
    if not (pd.api.types.is_object_dtype(series) or str(series.dtype) == "str"):
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    return isinstance(sample.iloc[0], _NON_SCALAR_TYPES)


def _prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Select and type the transaction columns written to SQLite.

    Explanation columns (``rule_reasons``, ``risk_reasons``) hold lists, which
    SQLite cannot store, so they are dropped here. Their flattened twins
    (``rule_reason_text``, ``risk_reason_text``) are what the warehouse exposes.
    The list form stays in the parquet file for the dashboard, which renders it
    as a bulleted list.

    Non-scalar columns are detected by inspection rather than by a hard-coded
    name list, so a new explanation column cannot silently break the load.
    """
    out = df.copy()

    dropped = [column for column in out.columns if _is_non_scalar(out[column])]
    if dropped:
        LOGGER.info("Dropping non-scalar columns before SQLite load: %s", ", ".join(dropped))
        out = out.drop(columns=dropped)

    for column in ("transaction_date", "posting_date", "approval_time", "invoice_time", "payment_time"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype(int)
        elif str(out[column].dtype) == "Int64":
            out[column] = out[column].astype("float64")

    # Final guard: nothing but scalars may reach the SQLite driver.
    remaining = [column for column in out.columns if _is_non_scalar(out[column])]
    if remaining:
        raise TypeError(f"Non-scalar columns would fail the SQLite load: {remaining}")

    # SQLite has no native date type; text ISO timestamps sort correctly.
    return out


def load_warehouse(
    transactions: pd.DataFrame | None = None,
    vendor_risk: pd.DataFrame | None = None,
    employees: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Build the SQLite warehouse from the pipeline outputs.

    Args:
        transactions: Scored transaction table. Loaded from disk when omitted.
        vendor_risk: Vendor risk table. Loaded from disk when omitted.
        employees: Employee master data. Loaded from disk when omitted.
        alerts: Long-format rule alerts. Loaded from disk when omitted.
        summary: Risk summary dictionary. Loaded from disk when omitted.

    Returns:
        Mapping of table name to row count.
    """
    ensure_directories()

    if transactions is None:
        transactions = load_dataframe(SCORED_TRANSACTIONS)
    if vendor_risk is None:
        vendor_risk = load_dataframe(VENDOR_RISK_TABLE)
    if employees is None:
        employees = load_dataframe(EMPLOYEES_CLEAN)
    if alerts is None:
        alerts = pd.read_csv(RULE_ALERTS_CSV) if RULE_ALERTS_CSV.exists() else pd.DataFrame()
    if summary is None and AUDIT_SUMMARY_REPORT.exists():
        summary = load_json(AUDIT_SUMMARY_REPORT)

    engine = get_engine()
    counts: dict[str, int] = {}

    with engine.begin() as connection:
        # Start from a clean slate so the build is idempotent.
        for table in (*TABLE_SCHEMA.keys(), "risk_summary"):
            connection.execute(text(f"DROP TABLE IF EXISTS {table}"))

    transactions_sql = _prepare_transactions(transactions)
    transactions_sql.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=5_000)
    counts["transactions"] = len(transactions_sql)

    # The whitelist is explicit rather than "every column" so the schema stays
    # readable in a SQL client. It has to be complete, though: the dashboard's
    # vendor page reads the component sub-scores (``vendor_risk_*``) to draw the
    # "why this vendor scores what it does" breakdown, and the dormancy view needs
    # the first/last payment dates. An earlier, narrower list silently dropped
    # them, which turned that breakdown into an empty chart rather than an error.
    vendor_columns = [
        column
        for column in (
            # Identity and master data
            "vendor_id", "vendor_name", "vendor_category", "bank_account", "registration_date",
            "country", "risk_level", "vendor_master_risk_score", "is_new_vendor",
            # Aggregated payment behaviour
            "vendor_transaction_count", "vendor_total_amount", "vendor_average_amount",
            "vendor_median_amount", "vendor_max_amount", "vendor_amount_std",
            "vendor_unique_accounts", "vendor_unique_departments",
            "vendor_first_transaction", "vendor_last_transaction",
            "vendor_weekend_share", "vendor_self_approval_share",
            # Scheme indicators
            "shared_bank_account", "amount_concentration",
            # Alert behaviour
            "alert_count", "alert_rate", "multi_alert_transaction_count",
            # Composite score and its components
            "vendor_risk_score", "vendor_risk_level", "vendor_risk_rank",
            "vendor_risk_alert_rate", "vendor_risk_master_risk",
            "vendor_risk_shared_bank_account", "vendor_risk_new_vendor",
            "vendor_risk_self_approval_share", "vendor_risk_weekend_share",
            "vendor_risk_amount_concentration",
        )
        if column in vendor_risk.columns
    ]
    vendors_sql = vendor_risk[vendor_columns].copy()
    for column in ("registration_date", "vendor_first_transaction", "vendor_last_transaction"):
        if column in vendors_sql.columns:
            vendors_sql[column] = pd.to_datetime(
                vendors_sql[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    # SQLite has no boolean type, so flags are stored as 0/1 integers.
    for column in ("shared_bank_account",):
        if column in vendors_sql.columns:
            vendors_sql[column] = vendors_sql[column].astype(int)
    vendors_sql.to_sql("vendors", engine, if_exists="replace", index=False, chunksize=2_000)
    counts["vendors"] = len(vendors_sql)

    employees.to_sql("employees", engine, if_exists="replace", index=False, chunksize=2_000)
    counts["employees"] = len(employees)

    if not alerts.empty:
        alerts_sql = alerts.copy()
        alerts_sql.insert(0, "alert_id", range(1, len(alerts_sql) + 1))
        alerts_sql.to_sql("audit_alerts", engine, if_exists="replace", index=False, chunksize=5_000)
    counts["audit_alerts"] = len(alerts)

    if summary:
        flat_summary = {
            key: (str(value) if isinstance(value, dict) else value)
            for key, value in summary.items()
        }
        pd.DataFrame([flat_summary]).to_sql("risk_summary", engine, if_exists="replace", index=False)
        counts["risk_summary"] = 1

    # Indexes make the analyst queries usable on a laptop.
    with engine.begin() as connection:
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_tx_vendor ON transactions(vendor_id)",
            "CREATE INDEX IF NOT EXISTS idx_tx_account ON transactions(account_code)",
            "CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(transaction_date)",
            "CREATE INDEX IF NOT EXISTS idx_tx_risk ON transactions(risk_level)",
            "CREATE INDEX IF NOT EXISTS idx_tx_score ON transactions(audit_risk_score)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_rule ON audit_alerts(rule_key)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_tx ON audit_alerts(transaction_id)",
        ):
            connection.execute(text(statement))

    LOGGER.info("Warehouse built at %s: %s", DB_PATH, counts)
    return counts


# --------------------------------------------------------------------------- #
# Query execution
# --------------------------------------------------------------------------- #
def split_sql_statements(sql_script: str) -> list[tuple[str, str]]:
    """Split a ``.sql`` file into ``(name, statement)`` pairs.

    Statements are delimited by lines of the form ``-- name: some_name``. This is
    a deliberately simple convention - it keeps ``sql/audit_queries.sql`` readable
    as plain SQL while still letting the runner label each result.
    """
    statements: list[tuple[str, str]] = []
    current_name: str | None = None
    buffer: list[str] = []

    for line in sql_script.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("-- name:"):
            if current_name and buffer:
                statements.append((current_name, "\n".join(buffer).strip()))
            current_name = stripped.split(":", 1)[1].strip()
            buffer = []
        elif stripped.startswith("--") or not stripped:
            continue
        else:
            buffer.append(line)

    if current_name and buffer:
        statements.append((current_name, "\n".join(buffer).strip()))
    return statements


def run_sql_file(path: Path = SQL_DIR / "audit_queries.sql") -> dict[str, pd.DataFrame]:
    """Execute every named query in a ``.sql`` file.

    Args:
        path: Path to the SQL file.

    Returns:
        Mapping of query name to result dataframe.
    """
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")

    script = path.read_text(encoding="utf-8")
    statements = split_sql_statements(script)
    results: dict[str, pd.DataFrame] = {}

    engine = get_engine()
    with engine.connect() as connection:
        for name, statement in statements:
            try:
                results[name] = pd.read_sql_query(text(statement), connection)
            except Exception as exc:  # surfaced rather than swallowed
                LOGGER.error("Query '%s' failed: %s", name, exc)
                results[name] = pd.DataFrame()

    LOGGER.info("Executed %s SQL queries from %s", len(results), path.name)
    return results


def query(sql: str) -> pd.DataFrame:
    """Run an ad-hoc SQL statement against the warehouse.

    Args:
        sql: A SELECT statement.

    Returns:
        The result set as a dataframe.
    """
    engine = get_engine()
    with engine.connect() as connection:
        return pd.read_sql_query(text(sql), connection)


def list_tables(db_path: Path = DB_PATH) -> list[str]:
    """Return the tables present in the warehouse."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def main() -> int:
    """CLI entry point: build the warehouse and run the analyst queries."""
    with Timer("warehouse build"):
        counts = load_warehouse()

    LOGGER.info("--- Warehouse contents ---")
    for table, count in counts.items():
        LOGGER.info("  %-16s %s rows", table, f"{count:,}")

    with Timer("SQL analytics"):
        results = run_sql_file()

    LOGGER.info("--- SQL analytics preview ---")
    for name, frame in results.items():
        LOGGER.info("  %-34s %s rows", name, f"{len(frame):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
