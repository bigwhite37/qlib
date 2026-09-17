#!/usr/bin/env python3
"""Test market-state interaction with the high-win reversal entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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

from lowvol_trend.config import Config, load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import FeatureStore, build_features, row_pct_rank  # noqa: E402
from scripts.research_variants import run_variant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2022-12-31")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[timing-grid] building features ...", flush=True)
    f = build_features(panel, cfg)
    close = pd.DataFrame(f.arr("close"), index=f.dates, columns=f.symbols)
    base = f.arr("base_pass")
    trend = (f.arr("close") > f.arr("ma60")) & (f.arr("ma20") > f.arr("ma60")) & (f.arr("ma60") > f.arr("ma60_prev5"))
    repair = (f.arr("ret1") > 0) & (f.arr("ret1") <= 0.04) & (f.arr("close") > f.arr("ma5"))
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    signal = base & trend & (ret20 <= -0.05) & repair
    score = row_pct_rank(-ret20, base)
    arrays = dict(f.arrays)
    arrays["entry_signal"] = signal
    arrays["entry_score"] = np.where(signal, score, np.nan).astype(np.float32)
    f_variant = FeatureStore(panel=f.panel, cfg=f.cfg, market=f.market, arrays=arrays, diagnostics=dict(f.diagnostics))

    variants = [
        ("timing_on_extreme_on", {}),
        ("timing_on_extreme_off", {"exit_market_extreme": False}),
        ("timing_off", {"use_market_timing": False}),
        ("timing_off_no_vol_no_dd", {"use_market_timing": False, "use_vol_target": False, "use_drawdown_control": False}),
        ("timing_on_extreme_off_stop2", {"exit_market_extreme": False, "stop_atr_mult": 2.0, "stop_min": 0.02, "stop_max": 0.04}),
        ("timing_on_extreme_off_hold10", {"exit_market_extreme": False, "max_hold_days": 10}),
        ("timing_on_extreme_off_pt1.5", {"exit_market_extreme": False, "profit_target_pct": 0.015}),
        ("timing_on_extreme_off_pt2.5", {"exit_market_extreme": False, "profit_target_pct": 0.025}),
        ("timing_on_extreme_off_rank_on", {"exit_market_extreme": False, "exit_rank": True}),
        ("timing_on_extreme_off_no_trend_pt2_wide", {"exit_market_extreme": False, "stop_atr_mult": 4.0, "stop_min": 0.03, "stop_max": 0.08}),
    ]
    rows: List[Dict] = []
    for tag, overrides in variants:
        def mutate(c: Config, overrides=overrides):
            c.strategy.profit_target_pct = 0.02
            c.strategy.exit_trend_ma20 = False
            c.strategy.exit_trend_ma60 = False
            c.strategy.exit_rank = False
            c.strategy.stop_atr_mult = 2.5
            c.strategy.stop_min = 0.02
            c.strategy.stop_max = 0.05
            c.strategy.max_hold_days = 15
            for key, value in overrides.items():
                setattr(c.strategy, key, value)

        result = run_variant(f_variant, cfg, args.start, args.end, mutate)
        result["variant"] = tag
        rows.append(result)
        print(
            "[timing-grid] {variant}: win={win:.3f} exp={expectancy:.4f} cagr={cagr:.4f} mdd={mdd:.3f} rounds={rounds}".format(
                **result
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    cols = ["variant", "win", "expectancy", "avg_win", "avg_loss", "profit_factor", "cagr", "mdd", "rounds"]
    print(frame[cols].sort_values("win", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
