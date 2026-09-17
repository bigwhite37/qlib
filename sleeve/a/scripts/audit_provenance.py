#!/usr/bin/env python3
"""Provenance probe: where did the daily features come from?

The quality flags say `intraday_from_sqlite_prefix`, which implies a SQLite
source carrying intraday rows.  This script reads the provenance tables so the
source can be located (or ruled out) from the database itself.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = "/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb"


def main() -> None:
    import duckdb

    con = duckdb.connect(DB, read_only=True, config={"memory_limit": "2GB"})
    for table in ("qlib_source_snapshots", "qlib_batches", "qlib_feature_import_audit", "qlib_contract_flags"):
        print("==", table, "==")
        cols = [row[0] for row in con.execute(f'describe "{table}"').fetchall()]
        print("  columns:", ", ".join(cols))
        try:
            rows = con.execute(f'select * from "{table}" limit 6').fetchall()
            for row in rows:
                text = " | ".join(str(value)[:120] for value in row)
                print("   ", text)
        except Exception as exc:
            print("   error:", exc)
        print()
    con.close()


if __name__ == "__main__":
    main()
