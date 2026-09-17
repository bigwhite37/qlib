#!/usr/bin/env python3
"""VWAP feature scan, fetched columnar to stay inside the memory budget."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shuzhenyi/code/python/qlib/sleeve/a")
sys.path.insert(0, str(ROOT / "src"))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib
ensure_local_qlib()

from lowvol_trend.config import load_config
from lowvol_trend.data import DuckDBPanelLoader

cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
init_local_qlib(cfg)
with DuckDBPanelLoader(cfg, use_cache=True) as loader:
    panel = loader.load(refresh_cache=False)
    con = loader._connection()
    lo, hi = 20200101, 20260914
    t_lo = int(np.searchsorted(panel.date_ints, lo))
    t_hi = int(np.searchsorted(panel.date_ints, hi, side="right"))
    symbols = panel.symbols
    con.execute("DROP TABLE IF EXISTS _vwap_univ")
    con.execute("CREATE TEMP TABLE _vwap_univ(sid INTEGER, qlib_symbol VARCHAR)")
    con.executemany("INSERT INTO _vwap_univ VALUES (?, ?)", list(enumerate(symbols)))
    first_index = int(panel.trade_indices[0])
    n_dates = t_hi - t_lo
    vwap = np.full((n_dates, len(symbols)), np.nan, dtype=np.float32)
    sql = """
        SELECT f.trade_index, u.sid, f.vwap
        FROM qlib_daily_features AS f
        JOIN _vwap_univ AS u ON f.qlib_symbol = u.qlib_symbol
        WHERE f.row_status = 'active' AND f.trade_date BETWEEN ? AND ?
    """
    chunk = 90
    for offset in range(0, n_dates, chunk):
        a = int(panel.date_ints[t_lo + offset])
        b = int(panel.date_ints[min(t_lo + offset + chunk, t_hi) - 1])
        res = con.execute(sql, [a, b]).fetchnumpy()
        if not res or len(res.get("trade_index", [])) == 0:
            continue
        tp = res["trade_index"].astype(np.int64) - first_index - t_lo
        jp = res["sid"].astype(np.int64)
        vals = res["vwap"]
        ok = (tp >= 0) & (tp < n_dates) & np.isfinite(vals)
        vwap[tp[ok], jp[ok]] = vals[ok].astype(np.float32)
        del res, tp, jp, vals, ok
    con.execute("DROP TABLE IF EXISTS _vwap_univ")
print("vwap finite:", int(np.isfinite(vwap).sum()), "of", vwap.size)

close = np.asarray(panel.field("close"), dtype=np.float64)[t_lo:t_hi]
high = np.asarray(panel.field("high"), dtype=np.float64)[t_lo:t_hi]
low = np.asarray(panel.field("low"), dtype=np.float64)[t_lo:t_hi]
valid = np.asarray(panel.valid)[t_lo:t_hi]
base = np.isfinite(vwap) & valid

fwd = {}
for h in (5, 10, 20):
    arr = np.full_like(close, np.nan)
    arr[:-h] = close[h:] / close[:-h] - 1.0
    fwd[h] = arr

vdf = pd.DataFrame(vwap)
vmean = vdf.rolling(20, min_periods=20).mean().to_numpy()
vstd = vdf.rolling(20, min_periods=20).std().to_numpy()
feats = {}
with np.errstate(all="ignore"):
    feats["vwap_dev"] = close / vwap - 1.0
    feats["vwap_dev5"] = vdf.rolling(5, min_periods=5).mean().to_numpy()
    feats["vwap_dev20"] = vmean
    feats["vwap_mom5"] = vwap / vdf.shift(5).to_numpy() - 1.0
    feats["vwap_mom20"] = vwap / vdf.shift(20).to_numpy() - 1.0
    feats["vwap_pos"] = (vwap - low) / (high - low)
    feats["vwap_cv20"] = vstd / np.where(vmean > 0, vmean, np.nan)
print()
for name, arr in feats.items():
    arr = np.asarray(arr, dtype=np.float64)
    out = []
    for h in (5, 10, 20):
        ics = []
        for t in range(0, n_dates - h):
            mm = base[t] & np.isfinite(arr[t]) & np.isfinite(fwd[h][t])
            if mm.sum() < 300:
                continue
            ics.append(pd.Series(arr[t][mm]).rank().corr(pd.Series(fwd[h][t][mm]).rank()))
        ics = np.asarray(ics)
        out.append((float(np.nanmean(ics)), float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics)))))
    print("   %-12s ic5=%+.4f (t=%+.1f)  ic10=%+.4f (t=%+.1f)  ic20=%+.4f (t=%+.1f)" % (
        name, out[0][0], out[0][1], out[1][0], out[1][1], out[2][0], out[2][1]))
