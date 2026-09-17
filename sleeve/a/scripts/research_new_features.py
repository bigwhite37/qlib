#!/usr/bin/env python3
"""IC scan for candidate new features, computed one feature at a time.

Focus on A-share specific microstructure: limit-up/limit-down events, overnight
versus intraday decomposition, Amihud illiquidity, range-based volatility, close
position inside the daily range and volume-price interaction.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader, infer_board  # noqa: E402
from lowvol_trend.v2 import v2_base_mask  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402

HORIZONS = (5, 10, 20)


def main() -> None:
    out = Path("/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_v2_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    features.arrays = {k: v for k, v in features.arrays.items() if k in {"close", "raw_close", "ret1"}}
    gc.collect()
    dates = panel.dates
    symbols = panel.symbols
    close = pd.DataFrame(features.arr("close"), index=dates, columns=symbols, copy=False)
    raw = pd.DataFrame(features.arr("raw_close"), index=dates, columns=symbols, copy=False)
    ret1 = pd.DataFrame(features.arr("ret1"), index=dates, columns=symbols, copy=False)
    open_ = pd.DataFrame(np.asarray(panel.field("open")), index=dates, columns=symbols, copy=False)
    high = pd.DataFrame(np.asarray(panel.field("high")), index=dates, columns=symbols, copy=False)
    low = pd.DataFrame(np.asarray(panel.field("low")), index=dates, columns=symbols, copy=False)
    volume = pd.DataFrame(np.asarray(panel.field("volume")), index=dates, columns=symbols, copy=False)
    amount = pd.DataFrame(np.asarray(panel.field("amount")), index=dates, columns=symbols, copy=False) * 1000.0
    prev_raw = raw.shift(1)
    boards = np.array([infer_board(s) for s in symbols], dtype=object)
    limit = np.full((len(dates), len(symbols)), 0.10, dtype=np.float64)
    for j, board in enumerate(boards):
        if board == "star":
            limit[:, j] = 0.20
        elif board == "bse":
            limit[:, j] = 0.30
        elif board == "chinext":
            limit[:, j] = np.where(panel.date_ints >= 20200824, 0.20, 0.10)
    with np.errstate(all="ignore"):
        chg = (raw / prev_raw - 1.0).to_numpy(dtype=np.float64)
    up_limit = (chg >= limit - 0.005).astype(np.float32)
    down_limit = (chg <= -limit + 0.005).astype(np.float32)
    del chg, limit
    gc.collect()

    t_lo = int(np.searchsorted(dates, pd.Timestamp("2016-01-04")))
    t_hi = int(np.searchsorted(dates, pd.Timestamp("2026-09-14"), side="right"))
    cl = features.arr("close").astype(np.float64)
    fwd = {}
    for h in HORIZONS:
        arr = np.full_like(cl, np.nan)
        arr[:-h] = cl[h:] / cl[:-h] - 1.0
        fwd[h] = arr
    del cl
    gc.collect()

    def columns() -> Iterator[Tuple[str, np.ndarray]]:
        yield "limit_up_1", up_limit
        yield "limit_up_5", pd.DataFrame(up_limit, index=dates).rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32)
        yield "limit_up_20", pd.DataFrame(up_limit, index=dates).rolling(20, min_periods=20).sum().to_numpy(dtype=np.float32)
        yield "limit_down_20", pd.DataFrame(down_limit, index=dates).rolling(20, min_periods=20).sum().to_numpy(dtype=np.float32)
        with np.errstate(all="ignore"):
            gap = (open_ / prev_raw - 1.0).to_numpy(dtype=np.float32)
            intraday = (raw / open_ - 1.0).to_numpy(dtype=np.float32)
        yield "gap", gap
        yield "intraday", intraday
        yield "gap5", pd.DataFrame(gap, index=dates).rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32)
        yield "intraday5", pd.DataFrame(intraday, index=dates).rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32)
        del gap, intraday
        gc.collect()
        with np.errstate(all="ignore"):
            amihud = (ret1.abs() / amount.clip(lower=1.0)).to_numpy(dtype=np.float32)
        yield "amihud20", pd.DataFrame(amihud, index=dates).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        del amihud
        gc.collect()
        with np.errstate(all="ignore"):
            parkinson = (np.log(high / low) ** 2).to_numpy(dtype=np.float32)
        yield "parkinson20", pd.DataFrame(parkinson, index=dates).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        del parkinson
        gc.collect()
        with np.errstate(all="ignore"):
            range_pos = (close - low) / (high - low)
        yield "range_pos20", range_pos.rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        del range_pos
        gc.collect()
        with np.errstate(all="ignore"):
            vol_chg = volume.pct_change(fill_method=None)
        yield "corr_ret_vol20", ret1.rolling(20, min_periods=20).corr(vol_chg).to_numpy(dtype=np.float32)
        del vol_chg
        gc.collect()
        with np.errstate(all="ignore"):
            up_vol = volume.where(ret1 > 0, 0.0).rolling(20, min_periods=20).sum()
            dn_vol = volume.where(ret1 < 0, 0.0).rolling(20, min_periods=20).sum()
            ratio = (up_vol / dn_vol.clip(lower=1.0)).to_numpy(dtype=np.float32)
        yield "vol_ratio_ud", ratio
        del up_vol, dn_vol, ratio
        gc.collect()
        yield "turnover_accel", (
            amount.rolling(5, min_periods=5).mean() / amount.rolling(20, min_periods=20).mean()
        ).to_numpy(dtype=np.float32)
        yield "ret_max5", ret1.rolling(5, min_periods=5).max().to_numpy(dtype=np.float32)
        yield "ret_min5", ret1.rolling(5, min_periods=5).min().to_numpy(dtype=np.float32)
        yield "range_pos_today", ((close - low) / (high - low)).to_numpy(dtype=np.float32)

    rows: List[dict] = []
    for name, arr in columns():
        record = {"feature": name}
        for h in HORIZONS:
            ics = []
            for t in range(t_lo, min(t_hi, len(dates) - h)):
                m = base[t] & np.isfinite(arr[t]) & np.isfinite(fwd[h][t])
                if m.sum() < 200:
                    continue
                ics.append(pd.Series(arr[t][m]).rank().corr(pd.Series(fwd[h][t][m]).rank()))
            ics = np.asarray(ics)
            record["ic_h" + str(h)] = float(np.nanmean(ics))
            record["t_h" + str(h)] = float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics)))
        rows.append(record)
        print(
            "[newfeat] %-18s ic5=%+.4f ic10=%+.4f ic20=%+.4f (t20=%+.1f)"
            % (name, record["ic_h5"], record["ic_h10"], record["ic_h20"], record["t_h20"]),
            flush=True,
        )
        del arr
        gc.collect()
    table = pd.DataFrame(rows)
    table["abs_ic20"] = table["ic_h20"].abs()
    table.sort_values("abs_ic20", ascending=False).to_csv(out / "new_feature_ic.csv", index=False)
    print(table.sort_values("abs_ic20", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
