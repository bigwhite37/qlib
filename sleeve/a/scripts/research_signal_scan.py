#!/usr/bin/env python3
"""Scan simple panel features against the existing V2 policy labels.

Answers one question: does any cheap, causal feature separate high-win,
positive-expectancy policy outcomes from the rest of the base universe?

Only reads cached artefacts (panel .npy memmaps + v2_labels.parquet).
"""

from __future__ import annotations

import argparse
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
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2_labels.parquet")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/signal_scan")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-09-14")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    print("[scan] loading panel ...", flush=True)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    print("[scan] computing market features ...", flush=True)

    close = pd.DataFrame(features.arr("close"), index=panel.dates, columns=panel.symbols, copy=False)
    high = pd.DataFrame(panel.field("high"), index=panel.dates, columns=panel.symbols, copy=False)
    low = pd.DataFrame(panel.field("low"), index=panel.dates, columns=panel.symbols, copy=False)
    volume = pd.DataFrame(panel.field("volume"), index=panel.dates, columns=panel.symbols, copy=False)
    amount = pd.DataFrame(panel.field("amount"), index=panel.dates, columns=panel.symbols, copy=False)

    proxy = pd.Series(market.proxy, index=panel.dates)
    mkt = {
        "mkt_ret1": proxy.pct_change(fill_method=None),
        "mkt_ret5": proxy / proxy.shift(5) - 1.0,
        "mkt_ret20": proxy / proxy.shift(20) - 1.0,
        "mkt_ret60": proxy / proxy.shift(60) - 1.0,
        "mkt_b20": pd.Series(market.breadth_20, index=panel.dates),
        "mkt_b60": pd.Series(market.breadth_60, index=panel.dates),
        "mkt_q": pd.Series(market.drop_diffusion, index=panel.dates),
        "mkt_state_cap": pd.Series(market.effective_cap, index=panel.dates),
    }

    print("[scan] reading labels ...", flush=True)
    cols = ["signal_date", "signal_index", "instrument", "status", "exit_reason", "policy_return_net", "policy_win"]
    labels = pd.read_parquet(args.labels, columns=cols)
    labels = labels[labels["status"] == "closed"].reset_index(drop=True)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    mask = labels["signal_date"].between(args.start, args.end)
    labels = labels[mask].reset_index(drop=True)
    t_idx = labels["signal_index"].to_numpy(dtype=np.int64)
    symbol_index = {s: j for j, s in enumerate(panel.symbols)}
    j_idx = labels["instrument"].map(symbol_index).to_numpy(dtype=np.int64)
    print("[scan] label rows", len(labels), flush=True)
    assert np.array_equal(
        panel.dates[t_idx].to_numpy(), labels["signal_date"].to_numpy()
    ), "signal_index does not align with panel dates"

    def series_val(s: pd.Series) -> np.ndarray:
        return s.to_numpy(dtype=np.float32)[t_idx, j_idx]

    def arr_val(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float32)[t_idx, j_idx]

    def vec_val(a) -> np.ndarray:
        return np.asarray(a, dtype=np.float32)[t_idx]

    feats: Dict[str, np.ndarray] = {}
    feats["ret1"] = arr_val(features.arr("ret1"))
    for n in (3, 5, 10, 20, 60):
        feats[f"ret{n}"] = series_val(close.pct_change(n, fill_method=None))
    feats["rel20"] = feats["ret20"] - vec_val(mkt["mkt_ret20"])
    feats["rel60"] = feats["ret60"] - vec_val(mkt["mkt_ret60"])
    feats["rel5"] = feats["ret5"] - vec_val(mkt["mkt_ret5"])
    feats["dist_ma5"] = series_val(close / close.rolling(5, min_periods=5).mean() - 1.0)
    feats["dist_ma20"] = series_val(close / close.rolling(20, min_periods=20).mean() - 1.0)
    feats["dist_ma60"] = series_val(close / close.rolling(60, min_periods=60).mean() - 1.0)
    feats["atr_ratio"] = arr_val(features.arr("atr20")) / arr_val(features.arr("close"))
    feats["vol20"] = series_val(close.pct_change(fill_method=None).rolling(20, min_periods=20).std())
    feats["vol60"] = arr_val(features.arr("vol60"))
    feats["downvol20"] = arr_val(features.arr("downvol20"))
    feats["amt_rank"] = arr_val(features.arr("amount_rank"))
    v20 = volume.rolling(20, min_periods=20).mean()
    feats["vol_surge"] = series_val(volume / v20)
    hi20 = close.rolling(20, min_periods=20).max()
    lo20 = close.rolling(20, min_periods=20).min()
    feats["pos_in_range20"] = series_val((close - lo20) / (hi20 - lo20))
    hi60 = close.rolling(60, min_periods=60).max()
    feats["dist_hi60"] = series_val(close / hi60 - 1.0)
    hi250 = close.rolling(250, min_periods=120).max()
    feats["dist_hi250"] = series_val(close / hi250 - 1.0)
    ret1s = close.pct_change(fill_method=None)
    feats["up_days20"] = series_val((ret1s > 0).rolling(20, min_periods=20).mean())
    feats["amp20"] = series_val(((high - low) / close).rolling(20, min_periods=20).mean())
    feats["trend_quality"] = arr_val(features.arr("trend_quality60"))
    feats["rs60"] = arr_val(features.arr("rs60"))
    feats["skew20"] = series_val(ret1s.rolling(20, min_periods=20).skew())
    amt_yuan = amount * 1000.0
    feats["log_amount"] = series_val(np.log(amt_yuan.clip(lower=1.0)))
    for k, s in mkt.items():
        feats[k] = np.repeat(s.to_numpy(dtype=np.float32)[t_idx], 1)

    y = labels["policy_win"].to_numpy(dtype=np.float32)
    r = labels["policy_return_net"].to_numpy(dtype=np.float32)
    dates = labels["signal_date"]

    # Cross-sectional deciles within each signal date.
    rows: List[dict] = []
    frame = pd.DataFrame(feats)
    frame["signal_date"] = dates.to_numpy()
    for name in feats:
        col = frame[name]
        # market features are constant within a date: use time-series quantiles.
        if name.startswith("mkt_"):
            q = pd.qcut(col.rank(method="first"), 10, labels=False, duplicates="drop")
        else:
            q = frame.groupby("signal_date")[name].transform(
                lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
            )
        for dec in range(10):
            sel = (q == dec).to_numpy()
            if sel.sum() < 1000:
                continue
            rows.append(
                {
                    "feature": name,
                    "decile": dec,
                    "n": int(sel.sum()),
                    "win": float(np.nanmean(y[sel])),
                    "ret": float(np.nanmean(r[sel])),
                }
            )
        print(f"[scan] {name} done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "decile_scan.csv", index=False)

    summary = (
        table.groupby("feature")
        .apply(
            lambda g: pd.Series(
                {
                    "win_lo": g.loc[g["decile"].idxmin(), "win"],
                    "win_hi": g.loc[g["decile"].idxmax(), "win"],
                    "ret_lo": g.loc[g["decile"].idxmin(), "ret"],
                    "ret_hi": g.loc[g["decile"].idxmax(), "ret"],
                    "spread_win": g["win"].max() - g["win"].min(),
                    "spread_ret": g["ret"].max() - g["ret"].min(),
                }
            ),
            include_groups=False,
        )
        .sort_values("spread_ret", ascending=False)
    )
    summary.to_csv(out_dir / "feature_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
