#!/usr/bin/env python3
"""Is VWAP informative, and does the account already include dividends?"""

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
    rows = con.execute(
        """
        SELECT trade_index, qlib_symbol, vwap, adjclose, close, factor
        FROM qlib_daily_features
        WHERE row_status = 'active' AND trade_date BETWEEN ? AND ?
        """,
        [lo, hi],
    ).fetchall()
print("rows fetched:", len(rows))

first_index = int(panel.trade_indices[0])
dates_lo = int(np.searchsorted(panel.date_ints, lo))
dates_hi = int(np.searchsorted(panel.date_ints, hi, side="right"))
n_dates = dates_hi - dates_lo
symbol_to_j = {s: j for j, s in enumerate(panel.symbols)}
vwap = np.full((n_dates, len(panel.symbols)), np.nan, dtype=np.float32)
adjr = np.full((n_dates, len(panel.symbols)), np.nan, dtype=np.float64)
for trade_index, symbol, v, a, c, f in rows:
    j = symbol_to_j.get(symbol)
    if j is None:
        continue
    t = int(trade_index) - first_index - dates_lo
    if 0 <= t < n_dates:
        if v is not None:
            vwap[t, j] = v
        if a is not None and c is not None and f:
            adjr[t, j] = float(a)

close = np.asarray(panel.field("close"), dtype=np.float64)[dates_lo:dates_hi]
raw = np.asarray(panel.raw_close(), dtype=np.float64)[dates_lo:dates_hi]
valid = np.asarray(panel.valid)[dates_lo:dates_hi]

print()
print("== vwap coverage ==")
print("   finite vwap:", int(np.isfinite(vwap).sum()), "of", vwap.size)
with np.errstate(all="ignore"):
    dev = close / vwap - 1.0
fin = np.isfinite(dev) & valid
print("   close/vwap-1 percentiles:", np.nanpercentile(dev[fin], [1, 5, 25, 50, 75, 95, 99]).round(5))

print()
print("== dividend check: adjusted vs raw price returns ==")
adj_ret = np.full_like(close, np.nan)
adj_ret[1:] = close[1:] / close[:-1] - 1.0
raw_ret = np.full_like(raw, np.nan)
raw_ret[1:] = raw[1:] / raw[:-1] - 1.0
m = valid & np.isfinite(adj_ret) & np.isfinite(raw_ret)
print("   mean daily adj-close return %.6f%%  mean daily raw return %.6f%%  difference %.4f%%/yr" % (
    np.nanmean(adj_ret[m]) * 100, np.nanmean(raw_ret[m]) * 100,
    (np.nanmean(adj_ret[m]) - np.nanmean(raw_ret[m])) * 252 * 100))
with np.errstate(all="ignore"):
    ratio = adjr / close
fin2 = np.isfinite(ratio)
years = pd.DatetimeIndex(panel.dates[dates_lo:dates_hi]).year
by_year = pd.DataFrame({"y": years, "r": ratio.ravel()})
print("   adjclose/close ratio by year (a drifting ratio means different dividend treatment):")
print(by_year.groupby("y")["r"].median().round(4).to_string())

print()
print("== IC of VWAP features vs forward returns ==")
base = np.isfinite(vwap) & valid
cl = close
fwd = {}
for h in (5, 10, 20):
    arr = np.full_like(cl, np.nan)
    arr[:-h] = cl[h:] / cl[:-h] - 1.0
    fwd[h] = arr
feats = {}
feats["vwap_dev"] = close / vwap - 1.0
feats["vwap_dev5"] = pd.DataFrame(feats["vwap_dev"]).rolling(5, min_periods=5).mean().to_numpy()
feats["vwap_dev20"] = pd.DataFrame(feats["vwap_dev"]).rolling(20, min_periods=20).mean().to_numpy()
with np.errstate(all="ignore"):
    feats["vwap_mom5"] = vwap / pd.DataFrame(vwap).shift(5).to_numpy() - 1.0
    feats["vwap_mom20"] = vwap / pd.DataFrame(vwap).shift(20).to_numpy() - 1.0
with np.errstate(all="ignore"):
    feats["vwap_pos"] = (vwap - np.asarray(panel.field("low"), dtype=np.float64)[dates_lo:dates_hi]) / (
        np.asarray(panel.field("high"), dtype=np.float64)[dates_lo:dates_hi]
        - np.asarray(panel.field("low"), dtype=np.float64)[dates_lo:dates_hi]
    )
for name, arr in feats.items():
    row = []
    for h in (5, 10, 20):
        ics = []
        for t in range(0, n_dates - h):
            mm = base[t] & np.isfinite(arr[t]) & np.isfinite(fwd[h][t])
            if mm.sum() < 300:
                continue
            ics.append(pd.Series(arr[t][mm]).rank().corr(pd.Series(fwd[h][t][mm]).rank()))
        ics = np.asarray(ics)
        row.append((float(np.nanmean(ics)), float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics)))))
    print("   %-12s ic5=%+.4f (t=%+.1f)  ic10=%+.4f (t=%+.1f)  ic20=%+.4f (t=%+.1f)" % (
        name, row[0][0], row[0][1], row[1][0], row[1][1], row[2][0], row[2][1]))
