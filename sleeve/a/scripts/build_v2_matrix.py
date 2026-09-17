#!/usr/bin/env python3
"""Build the V2 row matrix: policy labels + features for every base-universe row.

Two passes, each designed for a 3 GB budget:

1. the shared policy simulator produces the labels (chunked numpy accumulation);
2. features are computed on date slices and written straight into the output
   memmap, so no panel-wide feature frame is ever held in memory.

Rows follow the label table order, so label row i and matrix row i describe the
same (signal date, stock) pair.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402
from lowvol_trend.v2_policy import ExitPolicy, simulate_entries  # noqa: E402

WARMUP_DAYS = 300


def feature_columns(panel, market, lo: int, hi: int) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (name, full-slice array) for the date range [lo, hi)."""

    dates = panel.dates[lo:hi]
    symbols = panel.symbols
    frames = {
        name: pd.DataFrame(np.asarray(panel.field(name))[lo:hi], index=dates, columns=symbols, copy=False)
        for name in ("open", "close", "high", "low", "volume", "amount", "factor")
    }
    close = frames["close"]
    high = frames["high"]
    low = frames["low"]
    volume = frames["volume"]
    ret1 = close.pct_change(fill_method=None)
    proxy = pd.Series(market.proxy[lo:hi], index=dates)
    m_ret1 = proxy.pct_change(fill_method=None)

    def yield_close(name: str, value) -> Tuple[str, np.ndarray]:
        return name, np.asarray(value, dtype=np.float32)

    for n in (1, 2, 3, 5, 10, 20, 60, 120, 250):
        if n == 1:
            yield "ret1", np.asarray(ret1, dtype=np.float32)
            continue
        yield "ret" + str(n), close.pct_change(n, fill_method=None).to_numpy(dtype=np.float32)
    vol20 = ret1.rolling(20, min_periods=20).std()
    vol60 = ret1.rolling(60, min_periods=60).std()
    yield yield_close("vol20", vol20)
    yield yield_close("vol60", vol60)
    yield yield_close("vol_ratio", vol20 / vol60)
    yield "downvol20", ret1.where(ret1 < 0, 0.0).pow(2).rolling(20, min_periods=20).mean().pow(0.5).to_numpy(dtype=np.float32)
    yield yield_close("skew20", ret1.rolling(20, min_periods=20).skew())
    yield yield_close("kurt20", ret1.rolling(20, min_periods=20).kurt())
    yield "amp20", ((high - low) / close).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    atr_high = high.to_numpy(dtype=np.float64)
    atr_low = low.to_numpy(dtype=np.float64)
    atr_prev = close.shift(1).to_numpy(dtype=np.float64)
    atr_close = close.to_numpy(dtype=np.float64)
    true_range = np.maximum(
        atr_high - atr_low,
        np.maximum(np.abs(atr_high - atr_prev), np.abs(atr_low - atr_prev)),
    )
    yield "atr_ratio", (
        pd.DataFrame(true_range, index=dates, columns=symbols, copy=False)
        .rolling(20, min_periods=20)
        .mean()
        .to_numpy(dtype=np.float32)
        / np.where(atr_close > 0, atr_close, np.nan)
    ).astype(np.float32)
    del atr_high, atr_low, atr_prev, atr_close, true_range
    yield "up_ratio20", (ret1 > 0).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    yield "up_ratio60", (ret1 > 0).rolling(60, min_periods=60).mean().to_numpy(dtype=np.float32)
    yield "maxret20", ret1.rolling(20, min_periods=20).max().to_numpy(dtype=np.float32)
    yield "minret20", ret1.rolling(20, min_periods=20).min().to_numpy(dtype=np.float32)
    amount_yuan = frames["amount"] * 1000.0
    amount20 = amount_yuan.rolling(20, min_periods=10).mean()
    yield "log_amount", np.log(np.clip(amount20.to_numpy(dtype=np.float64), 1.0, None)).astype(np.float32)
    yield "vol_surge", (volume / volume.rolling(20, min_periods=20).mean()).to_numpy(dtype=np.float32)
    yield "vol_surge60", (
        volume.rolling(5, min_periods=5).mean() / volume.rolling(60, min_periods=60).mean()
    ).to_numpy(dtype=np.float32)
    yield "turnover_proxy", (amount20 / np.clip(close, 1e-9, None)).to_numpy(dtype=np.float32)
    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    ma120 = close.rolling(120, min_periods=120).mean()
    yield "dist_ma5", (close / ma5 - 1.0).to_numpy(dtype=np.float32)
    yield "dist_ma20", (close / ma20 - 1.0).to_numpy(dtype=np.float32)
    yield "dist_ma60", (close / ma60 - 1.0).to_numpy(dtype=np.float32)
    yield "dist_ma120", (close / ma120 - 1.0).to_numpy(dtype=np.float32)
    yield "ma20_ma60", (ma20 / ma60 - 1.0).to_numpy(dtype=np.float32)
    yield "ma60_ma120", (ma60 / ma120 - 1.0).to_numpy(dtype=np.float32)
    hi20 = close.rolling(20, min_periods=20).max()
    lo20 = close.rolling(20, min_periods=20).min()
    hi60 = close.rolling(60, min_periods=60).max()
    lo60 = close.rolling(60, min_periods=60).min()
    yield "pos_range20", ((close - lo20) / (hi20 - lo20)).to_numpy(dtype=np.float32)
    yield "pos_range60", ((close - lo60) / (hi60 - lo60)).to_numpy(dtype=np.float32)
    yield "dist_hi60", (close / hi60 - 1.0).to_numpy(dtype=np.float32)
    yield "dist_hi250", (close / close.rolling(250, min_periods=120).max() - 1.0).to_numpy(dtype=np.float32)
    yield "dist_lo250", (close / close.rolling(250, min_periods=120).min() - 1.0).to_numpy(dtype=np.float32)
    net_move = close - close.shift(60)
    swing = close.diff().abs().rolling(60, min_periods=60).sum()
    with np.errstate(all="ignore"):
        trend_quality = (net_move / swing).to_numpy(dtype=np.float64)
    trend_quality[~np.isfinite(trend_quality)] = np.nan
    yield "trend_quality", trend_quality.astype(np.float32)
    del net_move, swing
    ret60 = close.pct_change(60, fill_method=None).to_numpy(dtype=np.float32)
    proxy_ret60 = (proxy / proxy.shift(60) - 1.0).to_numpy(dtype=np.float32)
    with np.errstate(all="ignore"):
        rs60 = ret60 - proxy_ret60[:, None]
    rs60[~np.isfinite(ret60)] = np.nan
    yield "rs60", rs60.astype(np.float32)
    mkt_ret = pd.Series(m_ret1, index=dates)
    cov = ret1.rolling(60, min_periods=40).cov(mkt_ret)
    var = mkt_ret.rolling(60, min_periods=40).var()
    yield "beta60", cov.div(var, axis=0).to_numpy(dtype=np.float32)
    yield "corr60", ret1.rolling(60, min_periods=40).corr(mkt_ret).to_numpy(dtype=np.float32)
    rel5 = ret1.rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32) - 5.0 * m_ret1.to_numpy(dtype=np.float32)[:, None]
    yield "rel_ret5", rel5.astype(np.float32)
    yield "rel_ret20", (close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32) - (proxy / proxy.shift(20) - 1.0).to_numpy(dtype=np.float32)[:, None]).astype(np.float32)
    yield "rel_ret60", (ret60 - proxy_ret60[:, None]).astype(np.float32)

    # ---- A-share microstructure block (selected by measured 20-day rank IC) ----
    raw = close / frames["factor"]
    raw_prev = raw.shift(1)
    boards = panel.instrument_meta["board"].reindex(panel.symbols).fillna("unknown").to_numpy(dtype=object)
    limit = np.full((hi - lo, len(symbols)), 0.10, dtype=np.float64)
    date_ints = panel.date_ints[lo:hi]
    for j, board in enumerate(boards):
        if board == "star":
            limit[:, j] = 0.20
        elif board == "bse":
            limit[:, j] = 0.30
        elif board == "chinext":
            limit[:, j] = np.where(date_ints >= 20200824, 0.20, 0.10)
    with np.errstate(all="ignore"):
        change = (raw / raw_prev - 1.0).to_numpy(dtype=np.float64)
    up_limit = (change >= limit - 0.005).astype(np.float32)
    down_limit = (change <= -limit + 0.005).astype(np.float32)
    del change, limit, raw_prev
    yield "limit_up_1", up_limit
    yield "limit_up_5", pd.DataFrame(up_limit, index=dates).rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32)
    yield "limit_up_20", pd.DataFrame(up_limit, index=dates).rolling(20, min_periods=20).sum().to_numpy(dtype=np.float32)
    yield "limit_down_20", pd.DataFrame(down_limit, index=dates).rolling(20, min_periods=20).sum().to_numpy(dtype=np.float32)
    with np.errstate(all="ignore"):
        raw_prev2 = (close / frames["factor"]).shift(1)
        gap = (frames["open"] / raw_prev2 - 1.0).to_numpy(dtype=np.float32)
        intraday = ((close / frames["factor"]) / frames["open"] - 1.0).to_numpy(dtype=np.float32)
    yield "gap", gap
    yield "intraday", intraday
    yield "gap5", pd.DataFrame(gap, index=dates).rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32)
    yield "intraday5", pd.DataFrame(intraday, index=dates).rolling(5, min_periods=5).sum().to_numpy(dtype=np.float32)
    del gap, intraday, raw_prev2
    amount_yuan_all = frames["amount"] * 1000.0
    with np.errstate(all="ignore"):
        amihud = (ret1.abs() / amount_yuan_all.clip(lower=1.0)).to_numpy(dtype=np.float32)
    yield "amihud20", pd.DataFrame(amihud, index=dates).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    del amihud
    with np.errstate(all="ignore"):
        parkinson = (np.log(high / low) ** 2).to_numpy(dtype=np.float32)
    yield "parkinson20", pd.DataFrame(parkinson, index=dates).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    del parkinson
    with np.errstate(all="ignore"):
        range_pos = ((close - low) / (high - low)).to_numpy(dtype=np.float32)
    yield "range_pos20", pd.DataFrame(range_pos, index=dates).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    del range_pos
    with np.errstate(all="ignore"):
        vol_change = volume.pct_change(fill_method=None)
    yield "corr_ret_vol20", ret1.rolling(20, min_periods=20).corr(vol_change).to_numpy(dtype=np.float32)
    del vol_change
    with np.errstate(all="ignore"):
        up_vol = volume.where(ret1 > 0, 0.0).rolling(20, min_periods=20).sum()
        down_vol = volume.where(ret1 < 0, 0.0).rolling(20, min_periods=20).sum()
        ratio_ud = (up_vol / down_vol.clip(lower=1.0)).to_numpy(dtype=np.float32)
    yield "vol_ratio_ud", ratio_ud
    del up_vol, down_vol, ratio_ud
    with np.errstate(all="ignore"):
        accel = (
            amount_yuan_all.rolling(5, min_periods=5).mean() / amount_yuan_all.rolling(20, min_periods=20).mean()
        ).to_numpy(dtype=np.float32)
    yield "turnover_accel", accel
    del accel, amount_yuan_all
    yield "ret_max5", ret1.rolling(5, min_periods=5).max().to_numpy(dtype=np.float32)
    yield "ret_min5", ret1.rolling(5, min_periods=5).min().to_numpy(dtype=np.float32)


MARKET_FEATURES = (
    "mkt_ret1",
    "mkt_ret5",
    "mkt_ret20",
    "mkt_ret60",
    "mkt_b20",
    "mkt_b60",
    "mkt_q",
    "mkt_cap",
    "mkt_rawcap",
    "mkt_recovery",
)


def market_columns(market, lo: int, hi: int) -> Dict[str, np.ndarray]:
    proxy = pd.Series(market.proxy)
    return {
        "mkt_ret1": proxy.pct_change(fill_method=None).to_numpy(dtype=np.float32)[lo:hi],
        "mkt_ret5": (proxy / proxy.shift(5) - 1.0).to_numpy(dtype=np.float32)[lo:hi],
        "mkt_ret20": (proxy / proxy.shift(20) - 1.0).to_numpy(dtype=np.float32)[lo:hi],
        "mkt_ret60": (proxy / proxy.shift(60) - 1.0).to_numpy(dtype=np.float32)[lo:hi],
        "mkt_b20": market.breadth_20.astype(np.float32)[lo:hi],
        "mkt_b60": market.breadth_60.astype(np.float32)[lo:hi],
        "mkt_q": market.drop_diffusion.astype(np.float32)[lo:hi],
        "mkt_cap": market.effective_cap.astype(np.float32)[lo:hi],
        "mkt_rawcap": market.raw_cap.astype(np.float32)[lo:hi],
        "mkt_recovery": market.recovery_phase.astype(np.float32)[lo:hi],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--profit-target", type=float, default=0.05)
    parser.add_argument("--stop", default="none", choices=["none", "atr", "fixed"])
    parser.add_argument("--stop-pct", type=float, default=0.15)
    parser.add_argument("--hold", type=int, default=20)
    parser.add_argument("--trailing", type=int, default=0)
    parser.add_argument("--extreme-exit", type=int, default=1)
    parser.add_argument("--feature-chunk-days", type=int, default=250)
    parser.add_argument("--label-chunk-days", type=int, default=400)
    args = parser.parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[matrix] building lean features ...", flush=True)
    features = build_v2_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    t_lo = int(np.searchsorted(panel.dates, pd.Timestamp(args.start)))
    t_hi = int(np.searchsorted(panel.dates, pd.Timestamp(args.end), side="right"))
    entry = np.zeros_like(base)
    entry[t_lo:t_hi] = base[t_lo:t_hi]

    if args.stop == "none":
        policy = ExitPolicy(use_stop=False, use_trailing=bool(args.trailing), profit_target_pct=args.profit_target,
                            max_hold_days=args.hold, exit_on_market_extreme=bool(args.extreme_exit))
    elif args.stop == "atr":
        policy = ExitPolicy(use_trailing=bool(args.trailing), profit_target_pct=args.profit_target,
                            max_hold_days=args.hold, exit_on_market_extreme=bool(args.extreme_exit))
    else:
        policy = ExitPolicy(stop_atr_mult=0.0, stop_min=args.stop_pct, stop_max=args.stop_pct, use_stop=True,
                            use_trailing=False, profit_target_pct=args.profit_target, max_hold_days=args.hold,
                            exit_on_market_extreme=bool(args.extreme_exit))
    print("[matrix] policy:", policy.describe(), flush=True)
    extreme = market.effective_state == "extreme"
    # The simulator only reads these five arrays; releasing the rest before the
    # 8M-row label pass keeps the peak inside the 3 GB budget.
    names: List[str] = [name for name, _ in feature_columns(panel, market, max(0, t_lo - 1), t_lo)]
    names.extend(MARKET_FEATURES)
    keep = {"close", "raw_close", "atr20", "ma20", "ma60"}
    features.arrays = {k: v for k, v in features.arrays.items() if k in keep}
    gc.collect()
    t0 = time.time()
    n_dates = len(panel.dates)
    horizon_overlap = int(policy.max_hold_days) + int(policy.max_fill_wait_days) + 5
    capacity = int(entry[t_lo:t_hi].sum()) + 10_000
    store = {
        "signal_index": np.empty(capacity, dtype=np.int32),
        "entry_index": np.empty(capacity, dtype=np.int32),
        "symbol_index": np.empty(capacity, dtype=np.int32),
        "entry_filled": np.empty(capacity, dtype=bool),
        "exit_fill_index": np.empty(capacity, dtype=np.int32),
        "exit_signal_index": np.empty(capacity, dtype=np.int32),
        "exit_reason_code": np.empty(capacity, dtype=np.int8),
        "hold_days": np.empty(capacity, dtype=np.int16),
        "net_return": np.empty(capacity, dtype=np.float32),
        "gross_return": np.empty(capacity, dtype=np.float32),
        "stop_pct": np.empty(capacity, dtype=np.float32),
    }
    position = 0
    for a in range(t_lo, t_hi, args.label_chunk_days):
        b = min(a + args.label_chunk_days, t_hi)
        sim_end = min(b + horizon_overlap, n_dates)
        raw = simulate_entries(
            features, cfg, entry, policy, market_extreme=extreme,
            start_index=a, end_index=sim_end, return_frame=False,
        )
        n_chunk = len(raw["signal_index"])
        keep = raw["signal_index"] < b
        n_keep = int(keep.sum())
        for key, target in store.items():
            target[position:position + n_keep] = raw[key][keep]
        position += n_keep
        del raw
        gc.collect()
        print(
            f"[matrix] labels {panel.dates[a].date()}..{panel.dates[b-1].date()} rows={n_keep}",
            flush=True,
        )
    t_idx = store["signal_index"][:position].astype(np.int64)
    j_idx = store["symbol_index"][:position].astype(np.int64)
    n_rows = position
    label_path = prefix.with_name(prefix.name + "_labels.npz")
    np.savez(
        label_path,
        signal_index=store["signal_index"][:position],
        symbol_index=store["symbol_index"][:position],
        entry_index=store["entry_index"][:position],
        entry_filled=store["entry_filled"][:position],
        exit_fill_index=store["exit_fill_index"][:position],
        exit_signal_index=store["exit_signal_index"][:position],
        exit_reason_code=store["exit_reason_code"][:position],
        hold_days=store["hold_days"][:position],
        net_return=store["net_return"][:position],
        gross_return=store["gross_return"][:position],
        stop_pct=store["stop_pct"][:position],
        dates=panel.dates.to_numpy(dtype="datetime64[ns]"),
    )
    del store
    gc.collect()
    print(f"[matrix] {n_rows} label rows -> {label_path} in {time.time()-t0:.1f}s", flush=True)

    n_cols = len(names)
    del features
    gc.collect()
    if not np.all(np.diff(t_idx) >= 0):
        raise RuntimeError("label rows are expected to be ordered by signal date")
    parts = []
    chunk_days = args.feature_chunk_days
    left = 0
    for start_date in range(t_lo, t_hi, chunk_days):
        stop = min(start_date + chunk_days, t_hi)
        lo = max(0, start_date - WARMUP_DAYS)
        right = int(np.searchsorted(t_idx, stop, side="left"))
        if right <= left:
            continue
        local_t = t_idx[left:right] - lo
        local_j = j_idx[left:right]
        block = np.empty((right - left, n_cols), dtype=np.float32)
        col = 0
        for name, values in feature_columns(panel, market, lo, stop):
            block[:, col] = values[local_t, local_j]
            col += 1
            del values
        market_cols = market_columns(market, lo, stop)
        for name in MARKET_FEATURES:
            block[:, col] = market_cols[name][local_t]
            col += 1
        del market_cols
        part_path = prefix.with_name(prefix.name + "_features_part" + str(len(parts)) + ".npy")
        np.save(part_path, block)
        parts.append({"file": part_path.name, "start_row": int(left), "end_row": int(right)})
        print(
            f"[matrix] features {panel.dates[start_date].date()}..{panel.dates[stop-1].date()} rows={right-left} -> {part_path.name}",
            flush=True,
        )
        left = right
        del block
        gc.collect()

    with prefix.with_name(prefix.name + "_features.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "feature_names": names,
                "n_rows": int(n_rows),
                "policy": policy.describe(),
                "n_features": int(n_cols),
                "parts": parts,
                "start": args.start,
                "end": args.end,
            },
            fh,
            ensure_ascii=False,
        )
    print("[matrix] saved", len(parts), "feature parts,", n_rows, "rows", flush=True)


if __name__ == "__main__":
    main()
