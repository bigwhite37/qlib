#!/usr/bin/env python3
"""Inventory the DuckDB: what data exists beyond the daily OHLCV panel."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path("/Users/shuzhenyi/code/python/qlib/sleeve/a")
sys.path.insert(0, str(ROOT / "src"))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib
ensure_local_qlib()

from lowvol_trend.config import load_config
from lowvol_trend.data import DuckDBPanelLoader

cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
init_local_qlib(cfg)
with DuckDBPanelLoader(cfg, use_cache=True) as loader:
    con = loader._connection()
    print("memory_limit:", con.execute("SELECT current_setting('memory_limit')").fetchone()[0])
    print()
    print("== tables and views ==")
    for row in con.execute(
        "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema='main' ORDER BY table_type, table_name"
    ).fetchall():
        print("  ", row[0], "|", row[1])
    print()
    print("== columns per object ==")
    for row in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall():
        name = row[0]
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [name],
        ).fetchall()
        try:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except Exception as exc:
            cnt = "n/a (" + str(exc)[:60] + ")"
        print(f"  {name}  rows={cnt}")
        for cname, ctype in cols:
            print(f"      {cname}: {ctype}")
    print()
    print("== database file info ==")
    for row in con.execute("PRAGMA database_size").fetchall():
        print("  ", row)
