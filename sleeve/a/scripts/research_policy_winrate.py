#!/usr/bin/env python3
"""Policy family scan focused on the win-rate / expectancy trade-off.

Runs the shared policy simulator over a random subsample of the base universe
so that dozens of exit policies can be compared quickly.  The winner is then
used to regenerate the full label set.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
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

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from lowvol_trend.v2_policy import ExitPolicy, policy_stats, simulate_entries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.2)
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    parser.add_argument("--stage", default="all")
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
    rng = np.random.default_rng(20260101)
    entry = base & (rng.random(base.shape) < args.sample)
    print("[winrate] sample per day:", float(entry.sum(axis=1).mean()), flush=True)
    t_lo = int(np.searchsorted(panel.dates, pd.Timestamp(args.start)))
    t_hi = int(np.searchsorted(panel.dates, pd.Timestamp(args.end), side="right"))
    extreme = market.effective_state == "extreme"

    grid = []
    for tp, stop, hold in itertools.product(
        [0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
        ["none", "atr", "fixed10", "fixed15", "fixed20"],
        [10, 20, 40],
    ):
        if stop == "none":
            pol = ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=tp, max_hold_days=hold)
        elif stop == "atr":
            pol = ExitPolicy(profit_target_pct=tp, max_hold_days=hold)
        else:
            pct = float(stop.replace("fixed", "")) / 100.0
            pol = ExitPolicy(
                stop_atr_mult=0.0,
                stop_min=pct,
                stop_max=pct,
                use_trailing=False,
                profit_target_pct=tp,
                max_hold_days=hold,
            )
        grid.append((tp, stop, hold, pol))

    rows: List[dict] = []
    for tp, stop, hold, pol in grid:
        t0 = time.time()
        frame = simulate_entries(features, cfg, entry, pol, market_extreme=extreme, start_index=t_lo, end_index=t_hi)
        stats = policy_stats(frame)
        filled = frame[frame["entry_filled"] & frame["net_return"].notna()]
        stats.update({"tp": tp, "stop": stop, "hold": hold})
        rows.append(stats)
        print(
            f"[winrate] tp={tp:.2f} stop={stop:8s} hold={hold:2d} n={stats['n']:7d} win={stats.get('win', float('nan')):.4f} "
            f"ret={stats.get('ret', float('nan')):+.4f} avgW={stats.get('avg_win', 0):+.4f} avgL={stats.get('avg_loss', 0):+.4f} "
            f"h={stats.get('hold', 0):.1f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out / "policy_winrate_grid.csv", index=False)
    table = pd.DataFrame(rows)
    table["score"] = table["win"] + 5.0 * table["ret"]
    print(table.sort_values("score", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
