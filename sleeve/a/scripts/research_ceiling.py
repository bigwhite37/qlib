#!/usr/bin/env python3
"""Ceiling analysis: forward-return deciles of simple composite signals.

Answers "how much gross alpha is in the panel at all, after the base universe
filter?", which bounds everything the account can earn once costs are paid.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features, row_pct_rank  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402


def main() -> None:
    out = Path("/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    close = pd.DataFrame(features.arr("close"), index=panel.dates, columns=panel.symbols, copy=False)
    raw = pd.DataFrame(features.arr("raw_close"), index=panel.dates, columns=panel.symbols, copy=False)
    volume = pd.DataFrame(panel.field("volume"), index=panel.dates, columns=panel.symbols, copy=False)
    ret1 = close.pct_change(fill_method=None)
    vol20 = ret1.rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)
    high_np = np.asarray(panel.field("high"), dtype=np.float64)
    low_np = np.asarray(panel.field("low"), dtype=np.float64)
    close_np = np.asarray(features.arr("close"), dtype=np.float64)
    with np.errstate(all="ignore"):
        amp_series = (high_np - low_np) / close_np
    amp20 = (
        pd.DataFrame(amp_series, index=panel.dates).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    )
    amt = features.arr("amount20_yuan")
    dist60 = (close / close.rolling(60, min_periods=60).mean() - 1.0).to_numpy(dtype=np.float32)
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    ret5 = close.pct_change(5, fill_method=None).to_numpy(dtype=np.float32)
    r_vol = row_pct_rank(-vol20, base)
    r_amp = row_pct_rank(-amp20, base)
    r_amt = row_pct_rank(-amt, base)
    r_dist60 = row_pct_rank(-dist60, base)
    r_ret20 = row_pct_rank(-ret20, base)
    r_ret5 = row_pct_rank(ret5, base)

    composites: Dict[str, np.ndarray] = {
        "lowvol": r_vol,
        "illiq": r_amt,
        "lowamp": r_amp,
        "rev20_ma60": r_dist60,
        "rev20": r_ret20,
        "rev5": r_ret5,
        "combo3": (r_vol + r_amt + r_dist60) / 3.0,
        "combo5": (r_vol + r_amp + r_amt + r_dist60 + r_ret20) / 5.0,
    }
    horizons = [5, 10, 20]
    rows: List[dict] = []
    t_lo = int(np.searchsorted(panel.dates, pd.Timestamp("2016-01-04")))
    t_hi = int(np.searchsorted(panel.dates, pd.Timestamp("2026-09-14"), side="right"))
    cl = features.arr("close").astype(np.float64)
    rawv = features.arr("raw_close").astype(np.float64)
    for h in horizons:
        fwd_adj = np.full_like(cl, np.nan)
        fwd_raw = np.full_like(cl, np.nan)
        fwd_adj[:-h] = cl[h:] / cl[:-h] - 1.0
        fwd_raw[:-h] = rawv[h:] / rawv[:-h] - 1.0
        for name, score in composites.items():
            for label, lo, hi in [("top1", 0.99, 1.01), ("top5", 0.95, 1.01), ("top10", 0.90, 1.01), ("bottom10", 0.0, 0.10)]:
                sel = base & np.isfinite(score) & (score >= lo) & (score < hi)
                sel[:t_lo] = False
                sel[t_hi:] = False
                sel &= np.isfinite(fwd_adj)
                if sel.sum() < 1000:
                    continue
                rows.append(
                    {
                        "signal": name,
                        "bucket": label,
                        "horizon": h,
                        "n": int(sel.sum()),
                        "mean_adj": float(np.nanmean(fwd_adj[sel])),
                        "mean_raw": float(np.nanmean(fwd_raw[sel])),
                        "win_adj": float(np.nanmean((fwd_adj[sel] > 0).astype(float))),
                    }
                )
        print(f"[ceiling] h={h} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "ceiling_analysis.csv", index=False)
    print(table.pivot_table(index=["signal", "bucket"], columns="horizon", values=["mean_adj", "mean_raw", "win_adj"]).to_string())


if __name__ == "__main__":
    main()
