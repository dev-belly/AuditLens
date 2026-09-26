"""Checks that the dashboard and SQL runner reject incomplete pipeline output."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from dashboard import common
from src import database
from src.database import run_sql_file
from src.utils import DB_PATH, SCORED_TRANSACTIONS


@pytest.mark.parametrize("include_vendors", [False, True])
def test_dashboard_uses_parquet_when_warehouse_is_incomplete(
    tmp_path, monkeypatch, include_vendors: bool
) -> None:
    """A partial SQLite build must not blank out a valid scored ledger."""
    db_path = tmp_path / "partial.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE transactions (transaction_id TEXT)")
        connection.execute("INSERT INTO transactions VALUES ('PARTIAL_BUILD')")
        if include_vendors:
            connection.execute("CREATE TABLE vendors (vendor_id TEXT)")

    scored_path = tmp_path / "scored.parquet"
    pd.DataFrame(
        {
            "transaction_id": ["TX1"],
            "transaction_date": ["2025-03-11"],
            "risk_reason_text": ["Invoice number duplicated"],
        }
    ).to_parquet(scored_path)

    def query_partial(sql: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as connection:
            return pd.read_sql_query(sql, connection)

    monkeypatch.setattr(common, "DB_PATH", db_path)
    monkeypatch.setattr(common, "SCORED_TRANSACTIONS", scored_path)
    monkeypatch.setattr(common, "query", query_partial)

    assert not common.warehouse_ready()
    frame = common._load_transactions()
    assert frame["transaction_id"].tolist() == ["TX1"]
    assert frame["risk_reasons"].iloc[0] == ["Invoice number duplicated"]


def test_broken_analyst_query_aborts_the_run(tmp_path) -> None:
    """A SQL error must not be reported as an empty set of audit findings."""
    sql_path = tmp_path / "broken.sql"
    sql_path.write_text("-- name: broken_query\nSELECT * FROM missing_audit_table;\n")

    with pytest.raises(RuntimeError, match="Analyst query 'broken_query' failed"):
        run_sql_file(sql_path)


def test_failed_rebuild_preserves_the_previous_warehouse(tmp_path, monkeypatch) -> None:
    """A failed staging run must leave the last usable database untouched."""
    live_path = tmp_path / "auditlens.db"
    with sqlite3.connect(live_path) as connection:
        connection.execute("CREATE TABLE previous_build (version INTEGER)")
        connection.execute("INSERT INTO previous_build VALUES (1)")

    monkeypatch.setattr(database, "DB_PATH", live_path)
    with pytest.raises(RuntimeError, match="forced failure"):
        with database._staging_engine() as engine:
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE transactions (id INTEGER)")
            raise RuntimeError("forced failure")

    with sqlite3.connect(live_path) as connection:
        assert connection.execute("SELECT version FROM previous_build").fetchone() == (1,)
    assert not list(tmp_path.glob(".auditlens-build-*.db"))


def test_invalid_ledger_keys_and_orphaned_alerts_never_replace_warehouse(
    tmp_path, monkeypatch
) -> None:
    """Bad source joins must not publish a misleading audit population."""
    live_path = tmp_path / "auditlens.db"
    with sqlite3.connect(live_path) as connection:
        connection.execute("CREATE TABLE previous_build (version INTEGER)")
        connection.execute("INSERT INTO previous_build VALUES (1)")
    monkeypatch.setattr(database, "DB_PATH", live_path)

    transactions = pd.DataFrame({
        "transaction_id": ["TX1", "TX1"],
        "vendor_id": ["V1", "V1"],
        "account_code": ["1001", "1001"],
        "transaction_date": ["2025-01-01", "2025-01-02"],
        "risk_level": ["Low", "Low"],
        "audit_risk_score": [1.0, 2.0],
    })
    vendors = pd.DataFrame({"vendor_id": ["V1"]})
    employees = pd.DataFrame({"employee_id": ["E1"]})
    alerts = pd.DataFrame({
        "transaction_id": ["MISSING"], "rule_key": ["weekend"],
    })

    with pytest.raises(ValueError, match="transactions.transaction_id contains duplicate"):
        database.load_warehouse(transactions, vendors, employees, alerts, summary={})

    transactions.loc[1, "transaction_id"] = "TX2"
    with pytest.raises(ValueError, match="orphaned alert"):
        database.load_warehouse(transactions, vendors, employees, alerts, summary={})

    with sqlite3.connect(live_path) as connection:
        assert connection.execute("SELECT version FROM previous_build").fetchone() == (1,)
    assert not list(tmp_path.glob(".auditlens-build-*.db"))


@pytest.mark.skipif(
    not DB_PATH.exists() or not SCORED_TRANSACTIONS.exists(),
    reason="Run the pipeline first to check its persisted warehouse.",
)
def test_persisted_warehouse_matches_scored_ledger() -> None:
    """Catch an empty or partial database even when the pipeline logged success."""
    expected = len(pd.read_parquet(SCORED_TRANSACTIONS, columns=["transaction_id"]))
    assert expected > 0
    with sqlite3.connect(DB_PATH) as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"transactions", "vendors", "employees", "audit_alerts"}.issubset(tables)
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == expected
