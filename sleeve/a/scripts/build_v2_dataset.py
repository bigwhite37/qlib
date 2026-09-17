#!/usr/bin/env python3
"""Build the V2 feature/label matrix once and cache it as a memmap.

Rows follow ``cache/v2_labels.parquet`` exactly: each label row (signal date,
instrument) has one feature row.  Alpha158 families (VWAP removed) are computed
vectorised; a small set of market and relative-return features is appended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.alpha158_features import Alpha158Features, alpha158_feature_names  # noqa: E402
from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402


EXTRA_FEATURES = [
    "ret1",
    "ret5",
    "ret20",
    "ret60",
    "rel_ret5",
    "rel_ret20",
    "rel_ret60",
    "market_proxy_ret1",
    "market_proxy_ret5",
    "market_proxy_ret20",
    "market_proxy_ret60",
    "market_breadth20",
    "market_breadth60",
    "market_drop_diffusion",
    "market_effective_cap",
    "market_raw_cap",
    "market_recovery_phase",
    "amount_rank",
    "raw_price",
    "valid_count",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(ROOT / "cache" / "v2_labels.parquet"))
    parser.add_argument("--output", default=str(ROOT / "cache" / "v2_features.npy"))
    parser.add_argument("--chunk-years", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[v2-data] building panel features ...", flush=True)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    labels = pd.read_parquet(args.labels).sort_values(["signal_index", "instrument"]).reset_index(drop=True)
    print("[v2-data] labels", labels.shape, flush=True)

    t_idx = labels["signal_index"].to_numpy(dtype=np.int64)
    symbol_index = {symbol: j for j, symbol in enumerate(panel.symbols)}
    j_idx = labels["instrument"].map(symbol_index).to_numpy(dtype=np.int64)
    if np.isnan(j_idx).any():
        raise RuntimeError("Some labels reference instruments outside the panel")

    # Extra features at row level.
    close = pd.DataFrame(features.arr("close"), index=features.dates, columns=features.symbols)
    ret1 = features.arr("ret1")
    ret5 = close.pct_change(5, fill_method=None).to_numpy(dtype=np.float32)
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    ret60 = close.pct_change(60, fill_method=None).to_numpy(dtype=np.float32)
    proxy = pd.Series(market.proxy, index=features.dates)
    m_ret1 = proxy.pct_change(fill_method=None).to_numpy(dtype=np.float32)
    m_ret5 = (proxy / proxy.shift(5) - 1.0).to_numpy(dtype=np.float32)
    m_ret20 = (proxy / proxy.shift(20) - 1.0).to_numpy(dtype=np.float32)
    m_ret60 = (proxy / proxy.shift(60) - 1.0).to_numpy(dtype=np.float32)

    alpha_names = alpha158_feature_names()
    feature_names: List[str] = list(alpha_names) + EXTRA_FEATURES
    n_rows, n_features = len(labels), len(feature_names)
    matrix = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32, shape=(n_rows, n_features))
    alpha_engine = Alpha158Features(features)

    years = sorted(labels["signal_date"].dt.year.unique())
    for start_year in range(int(years[0]), int(years[-1]) + 1, args.chunk_years):
        end_year = start_year + args.chunk_years - 1
        mask = labels["signal_date"].dt.year.between(start_year, end_year).to_numpy()
        if not mask.any():
            continue
        rows = np.flatnonzero(mask)
        tt, jj = t_idx[rows], j_idx[rows]
        print(f"[v2-data] chunk {start_year}-{end_year}: {len(rows)} rows", flush=True)
        alpha = alpha_engine.compute(tt, jj)
        col = 0
        for name in alpha_names:
            matrix[rows, col] = alpha[name]
            col += 1
        extras = {
            "ret1": ret1[tt, jj],
            "ret5": ret5[tt, jj],
            "ret20": ret20[tt, jj],
            "ret60": ret60[tt, jj],
            "rel_ret5": ret5[tt, jj] - m_ret5[tt],
            "rel_ret20": ret20[tt, jj] - m_ret20[tt],
            "rel_ret60": ret60[tt, jj] - m_ret60[tt],
            "market_proxy_ret1": m_ret1[tt],
            "market_proxy_ret5": m_ret5[tt],
            "market_proxy_ret20": m_ret20[tt],
            "market_proxy_ret60": m_ret60[tt],
            "market_breadth20": market.breadth_20[tt],
            "market_breadth60": market.breadth_60[tt],
            "market_drop_diffusion": market.drop_diffusion[tt],
            "market_effective_cap": market.effective_cap[tt],
            "market_raw_cap": market.raw_cap[tt],
            "market_recovery_phase": market.recovery_phase[tt],
            "amount_rank": features.arr("amount_rank")[tt, jj],
            "raw_price": features.arr("raw_close")[tt, jj],
            "valid_count": features.arr("valid_count")[tt, jj],
        }
        for name in EXTRA_FEATURES:
            matrix[rows, col] = extras[name]
            col += 1
        del alpha, extras
        matrix.flush()
        print(f"[v2-data] chunk done: {col} features", flush=True)

    with (ROOT / "cache" / "v2_features.json").open("w", encoding="utf-8") as fh:
        json.dump({"feature_names": feature_names, "n_rows": n_rows, "labels": args.labels}, fh, ensure_ascii=False)
    print("[v2-data] saved", args.output, matrix.shape, flush=True)


if __name__ == "__main__":
    main()
