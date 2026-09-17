#!/usr/bin/env python3
"""Inventory the DuckDB source: is any intraday data reachable?

Round 17 found `is_intraday_derived` and `quality_flags` columns in the daily
feature table, and quality_flags records `actual_minute_rows=240` for some rows.
This script lists every table and view in the database, the columns of each, and
the distinct quality-flag values, so the question "is minute data reachable"
can be answered from the artefact itself rather than from memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = "/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb"


def main() -> None:
    import duckdb

    con = duckdb.connect(DB, read_only=True, config={"memory_limit": "2GB"})
    print("== tables and views ==")
    rows = con.execute(
        "select table_schema, table_name, table_type from information_schema.tables order by 1, 2"
    ).fetchall()
    for schema, name, kind in rows:
        print(f"  {schema}.{name} [{kind}]")

    print()
    print("== columns matching intraday keywords ==")
    cols = con.execute(
        "select table_name, column_name, data_type from information_schema.columns "
        "where lower(column_name) similar to '%(min|tick|intraday|bar|second|time)%' "
        "order by 1, 2"
    ).fetchall()
    for name, column, dtype in cols:
        print(f"  {name}.{column} {dtype}")

    print()
    print("== row counts ==")
    for schema, name, kind in rows:
        if kind != "BASE TABLE":
            continue
        try:
            n = con.execute(f'select count(*) from "{schema}"."{name}"').fetchone()[0]
        except Exception as exc:  # pragma: no cover - diagnostics only
            n = f"error: {exc}"
        print(f"  {schema}.{name}: {n}")

    print()
    print("== distinct quality_flags (first 40) ==")
    try:
        flags = con.execute(
            "select quality_flags, count(*) as n from qlib_daily_features "
            "group by 1 order by 2 desc limit 40"
        ).fetchall()
        for flag, n in flags:
            print(f"  {n:>10}  {flag}")
    except Exception as exc:
        print("  quality_flags unavailable:", exc)

    con.close()


if __name__ == "__main__":
    main()
