#!/usr/bin/env python3
"""Entry x exit-policy grid search on the policy simulator.

Research step only: it answers "does this entry family produce positive,
high-hit-rate trades under this exit policy?" before any model or account
backtest is built.  Ranking uses a fast account proxy; every candidate that
survives is re-run through the real Qlib account engine afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
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
from lowvol_trend.v2_policy import ExitPolicy, policy_stats, simulate_entries  # noqa: E402


def rank_of(values, mask):
    return row_pct_rank(np.asarray(values, dtype=np.float32), np.asarray(mask))


def build_entry_masks(features, cfg, base, market) -> Dict[str, np.ndarray]:
    close = pd.DataFrame(features.arr("close"), index=features.dates, columns=features.symbols, copy=False)
    high = pd.DataFrame(features.panel.field("high"), index=features.dates, columns=features.symbols, copy=False)
    low = pd.DataFrame(features.panel.field("low"), index=features.dates, columns=features.symbols, copy=False)
    volume = pd.DataFrame(features.panel.field("volume"), index=features.dates, columns=features.symbols, copy=False)
    ret1 = features.arr("ret1")
    ret5 = close.pct_change(5, fill_method=None).to_numpy(dtype=np.float32)
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    ret60 = close.pct_change(60, fill_method=None).to_numpy(dtype=np.float32)
    ma20 = features.arr("ma20")
    ma60 = features.arr("ma60")
    atr20 = features.arr("atr20")
    hi20 = close.rolling(20, min_periods=20).max().to_numpy(dtype=np.float32)
    vol20 = volume.rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    vol60 = features.arr("vol60")
    dist_high = hi20 - features.arr("close")

    r5 = rank_of(ret5, base)
    r20 = rank_of(ret20, base)
    r60 = rank_of(ret60, base)
    rvol60 = rank_of(vol60, base)

    masks: Dict[str, np.ndarray] = {}
    masks["rev5_bot20"] = base & np.isfinite(r5) & (r5 <= 0.20)
    masks["rev5_bot20_above60"] = masks["rev5_bot20"] & (features.arr("close") > ma60)
    masks["rev5_bot20_above20"] = masks["rev5_bot20"] & (features.arr("close") > ma20)
    masks["rev20_bot20_above60"] = base & np.isfinite(r20) & (r20 <= 0.20) & (features.arr("close") > ma60)
    masks["mom20_top20_above20"] = base & np.isfinite(r20) & (r20 >= 0.80) & (features.arr("close") > ma20)
    masks["mom60_top20_above20"] = base & np.isfinite(r60) & (r60 >= 0.80) & (features.arr("close") > ma20)
    masks["lowvol30_up"] = base & np.isfinite(rvol60) & (rvol60 <= 0.30) & (features.arr("close") > ma20) & (ma20 > ma60)
    masks["breakout20"] = (
        base & np.isfinite(hi20) & (features.arr("close") >= hi20) & (ret1 > 0)
    )
    masks["pullback_atr"] = (
        base
        & np.isfinite(dist_high)
        & np.isfinite(atr20)
        & (dist_high >= 1.0 * atr20)
        & (dist_high <= 4.0 * atr20)
        & (features.arr("close") > ma60)
    )
    masks["volsurge"] = (
        base & np.isfinite(vol20) & (vol20 > 0) & (volume.to_numpy(dtype=np.float32) > 2.0 * vol20) & (ret1 > 0.03)
    )
    masks["up_day_above20"] = base & (ret1 > 0.0) & (ret1 <= 0.05) & (features.arr("close") > ma20)
    rng = np.random.default_rng(20240915)
    masks["random_base"] = base & (rng.random(base.shape) < 0.02)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[grid] features ...", flush=True)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    masks = build_entry_masks(features, cfg, base, market)
    print("[grid] base per day (mean):", float(base.sum(axis=1).mean()), flush=True)
    for name, m in masks.items():
        print(f"[grid] mask {name}: total={int(m.sum())} per_day={m.sum(axis=1).mean():.1f}", flush=True)

    policies: Dict[str, ExitPolicy] = {
        "p_stop_trail20": ExitPolicy(),
        "p_tp5": ExitPolicy(profit_target_pct=0.05),
        "p_tp5_h10": ExitPolicy(profit_target_pct=0.05, max_hold_days=10),
        "p_tp3_h10": ExitPolicy(profit_target_pct=0.03, max_hold_days=10),
        "p_tp8_h20": ExitPolicy(profit_target_pct=0.08, max_hold_days=20),
        "p_nostop_h5": ExitPolicy(use_stop=False, use_trailing=False, max_hold_days=5),
        "p_nostop_h10": ExitPolicy(use_stop=False, use_trailing=False, max_hold_days=10),
        "p_nostop_h20": ExitPolicy(use_stop=False, use_trailing=False, max_hold_days=20),
        "p_stop5_h10": ExitPolicy(stop_atr_mult=1.0, stop_min=0.05, stop_max=0.05, use_trailing=False, max_hold_days=10),
    }
    if args.quick:
        policies = {"p_stop_trail20": policies["p_stop_trail20"], "p_tp5_h10": policies["p_tp5_h10"]}

    start_index = int(np.searchsorted(panel.dates, pd.Timestamp(args.start)))
    end_index = int(np.searchsorted(panel.dates, pd.Timestamp(args.end), side="right"))
    extreme = market.effective_state == "extreme"

    rows: List[dict] = []
    for pname, policy in policies.items():
        for mname, mask in masks.items():
            t0 = time.time()
            frame = simulate_entries(
                features, cfg, mask, policy, market_extreme=extreme, start_index=start_index, end_index=end_index
            )
            stats = policy_stats(frame)
            stats.update({"policy": pname, "entry": mname, "seconds": round(time.time() - t0, 1)})
            rows.append(stats)
            print(
                f"[grid] {pname:16s} {mname:22s} n={stats['n']:7d} win={stats['win']:.3f} "
                f"ret={stats['ret']:+.4f} hold={stats.get('hold', float('nan')):.1f} ({stats['seconds']}s)",
                flush=True,
            )
            table = pd.DataFrame(rows)
            table.to_csv(out_dir / "policy_grid.csv", index=False)
    table = pd.DataFrame(rows)
    table = table.sort_values("ret", ascending=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
