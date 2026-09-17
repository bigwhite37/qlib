#!/usr/bin/env python3
"""Coverage and semantics of the unused DuckDB columns, queried cheaply."""

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
    print("== coverage, one month per year (active final rows) ==")
    for lo, hi in ((20160101, 20160131), (20200601, 20200630), (20240601, 20240630), (20260601, 20260630)):
        q = """
            SELECT COUNT(*),
                   COUNT(vwap), COUNT(adjclose), COUNT(change), COUNT(factor),
                   COUNT(CASE WHEN is_intraday_derived THEN 1 END)
            FROM qlib_daily_features
            WHERE row_status = 'active' AND trade_date BETWEEN ? AND ?
        """
        print("   ", (lo, hi), con.execute(q, [lo, hi]).fetchone())
    print()
    print("== sample rows (SH600519) ==")
    q2 = """
        SELECT trade_date, close, adjclose, factor, vwap, change, amount, volume
        FROM qlib_daily_features
        WHERE qlib_symbol = 'SH600519' AND row_status = 'active' AND trade_date BETWEEN 20240102 AND 20240110
        ORDER BY trade_date
    """
    for r in con.execute(q2).fetchall():
        print("   ", r)
    print()
    print("== identities, single symbol over one year ==")
    q3 = """
        WITH s AS (
            SELECT trade_date, close, adjclose, change, factor, vwap,
                   lag(close) OVER (ORDER BY trade_date) AS prev_close
            FROM qlib_daily_features
            WHERE qlib_symbol = 'SH600519' AND row_status = 'active' AND trade_date BETWEEN 20230101 AND 20231231
        )
        SELECT COUNT(*),
               SUM(CASE WHEN abs(adjclose - close) < 1e-6 THEN 1 ELSE 0 END),
               SUM(CASE WHEN abs(adjclose - close / factor) < 1e-4 THEN 1 ELSE 0 END),
               SUM(CASE WHEN prev_close IS NOT NULL AND abs(change - (close / prev_close - 1.0)) < 1e-6 THEN 1 ELSE 0 END),
               SUM(CASE WHEN prev_close IS NOT NULL AND abs(change - (close / prev_close - 1.0) * 100.0) < 1e-4 THEN 1 ELSE 0 END),
               SUM(CASE WHEN vwap IS NOT NULL AND vwap > 0 THEN 1 ELSE 0 END)
        FROM s
    """
    print("    n, adj==close, adj==close/factor, change==pct, change==pct*100, vwap>0:", con.execute(q3).fetchone())
    print()
    print("== quality flags ==")
    for r in con.execute(
        "SELECT quality_flags, COUNT(*) FROM qlib_daily_features WHERE row_status='active' GROUP BY 1 ORDER BY 2 DESC LIMIT 8"
    ).fetchall():
        print("   ", r)
