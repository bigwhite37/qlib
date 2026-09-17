#!/usr/bin/env python3
"""Cross-sectional information-coefficient scan of the panel feature set.

Measures the ceiling of what daily OHLCV features can predict: rank IC against
forward returns at 5/10/20 days, with an ICIR and t-statistic per feature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

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
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    close = pd.DataFrame(features.arr("close"), index=panel.dates, columns=panel.symbols, copy=False)
    volume = pd.DataFrame(panel.field("volume"), index=panel.dates, columns=panel.symbols, copy=False)
    high = pd.DataFrame(panel.field("high"), index=panel.dates, columns=panel.symbols, copy=False)
    low = pd.DataFrame(panel.field("low"), index=panel.dates, columns=panel.symbols, copy=False)
    ret1 = close.pct_change(fill_method=None)

    feats: Dict[str, np.ndarray] = {}
    for n in (1, 3, 5, 10, 20, 60, 120, 250):
        feats[f"ret{n}"] = close.pct_change(n, fill_method=None).to_numpy(dtype=np.float32)
    feats["vol20"] = ret1.rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)
    feats["vol60"] = features.arr("vol60")
    feats["downvol20"] = features.arr("downvol20")
    feats["atr_ratio"] = (features.arr("atr20") / features.arr("close")).astype(np.float32)
    feats["amount20"] = features.arr("amount20_yuan")
    feats["log_amount"] = np.log(np.clip(features.arr("amount20_yuan").astype(np.float64), 1.0, None)).astype(np.float32)
    feats["vol_surge"] = (volume / volume.rolling(20, min_periods=20).mean()).to_numpy(dtype=np.float32)
    hi20 = close.rolling(20, min_periods=20).max()
    lo20 = close.rolling(20, min_periods=20).min()
    feats["pos_range20"] = ((close - lo20) / (hi20 - lo20)).to_numpy(dtype=np.float32)
    feats["dist_ma20"] = (close / close.rolling(20, min_periods=20).mean() - 1.0).to_numpy(dtype=np.float32)
    feats["dist_ma60"] = (close / close.rolling(60, min_periods=60).mean() - 1.0).to_numpy(dtype=np.float32)
    feats["dist_hi250"] = (close / close.rolling(250, min_periods=120).max() - 1.0).to_numpy(dtype=np.float32)
    feats["up_ratio20"] = (ret1 > 0).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    feats["amp20"] = ((high - low) / close).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    feats["trend_quality"] = features.arr("trend_quality60")
    feats["skew20"] = ret1.rolling(20, min_periods=20).skew().to_numpy(dtype=np.float32)
    feats["mkt_b60"] = np.repeat(market.breadth_60.astype(np.float32)[:, None], base.shape[1], axis=1)

    t_lo = int(np.searchsorted(panel.dates, pd.Timestamp(args.start)))
    t_hi = int(np.searchsorted(panel.dates, pd.Timestamp(args.end), side="right"))
    horizons = [int(x) for x in args.horizons.split(",")]
    rows = []
    for h in horizons:
        fwd = np.full_like(features.arr("close"), np.nan, dtype=np.float32)
        cl = features.arr("close").astype(np.float64)
        with np.errstate(all="ignore"):
            fwd[:-h] = (cl[h:] / cl[:-h] - 1.0).astype(np.float32)
        fwd = fwd.astype(np.float32)
        for name, arr in feats.items():
            ics = []
            for t in range(t_lo, min(t_hi, len(panel.dates) - h)):
                m = base[t] & np.isfinite(arr[t]) & np.isfinite(fwd[t])
                if m.sum() < 200:
                    continue
                a = pd.Series(arr[t][m]).rank()
                b = pd.Series(fwd[t][m]).rank()
                ics.append(float(a.corr(b)))
            ics = np.asarray(ics, dtype=np.float64)
            if ics.size < 50:
                continue
            mean_ic = float(np.nanmean(ics))
            std_ic = float(np.nanstd(ics))
            rows.append(
                {
                    "feature": name,
                    "horizon": h,
                    "mean_ic": mean_ic,
                    "ic_std": std_ic,
                    "icir": mean_ic / std_ic * np.sqrt(243.0) if std_ic > 0 else np.nan,
                    "t_stat": mean_ic / std_ic * np.sqrt(len(ics)) if std_ic > 0 else np.nan,
                    "n_days": len(ics),
                }
            )
            print(f"[ic] h={h} {name:14s} ic={mean_ic:+.4f} icir={rows[-1]['icir']:+.2f} t={rows[-1]['t_stat']:+.1f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "ic_scan.csv", index=False)
    print(table.sort_values(["horizon", "mean_ic"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
