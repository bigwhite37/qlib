#!/usr/bin/env python3
"""Combined entry/exit grid for a positive-expectancy, win-rate>=50% candidate."""

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
    print("[combo] building features ...", flush=True)
    f = build_features(panel, cfg)
    close = pd.DataFrame(f.arr("close"), index=f.dates, columns=f.symbols)
    base = f.arr("base_pass")
    trend = (f.arr("close") > f.arr("ma60")) & (f.arr("ma20") > f.arr("ma60")) & (f.arr("ma60") > f.arr("ma60_prev5"))
    repair = (f.arr("ret1") > 0) & (f.arr("ret1") <= 0.04) & (f.arr("close") > f.arr("ma5"))
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    rs_ok = np.isfinite(f.arr("rs_rank")) & (f.arr("rs_rank") >= 0.70)
    signals = {
        "rev20": base & trend & (ret20 <= -0.05) & repair,
        "rev20_rs": base & trend & rs_ok & (ret20 <= -0.05) & repair,
        "rev20_deep": base & trend & (ret20 <= -0.08) & repair,
        "rev20_band": base & trend & rs_ok & (ret20 <= -0.03) & (ret20 >= -0.12) & repair,
    }
    exit_policies = [
        ("pt2_stop4_hold15", dict(profit_target_pct=0.02, stop_atr_mult=4.0, stop_min=0.03, stop_max=0.08, max_hold_days=15, use_market_timing=False)),
        ("pt2_stop4_hold20", dict(profit_target_pct=0.02, stop_atr_mult=4.0, stop_min=0.03, stop_max=0.08, max_hold_days=20, use_market_timing=False)),
        ("pt2.5_stop5_hold20", dict(profit_target_pct=0.025, stop_atr_mult=5.0, stop_min=0.03, stop_max=0.09, max_hold_days=20, use_market_timing=False)),
    ]
    rows: List[Dict] = []
    for sig_name, signal in signals.items():
        score = row_pct_rank(-ret20, base)
        arrays = dict(f.arrays)
        arrays["entry_signal"] = signal
        arrays["entry_score"] = np.where(signal, score, np.nan).astype(np.float32)
        f_variant = FeatureStore(panel=f.panel, cfg=f.cfg, market=f.market, arrays=arrays, diagnostics=dict(f.diagnostics))
        for exit_name, overrides in exit_policies:
            def mutate(c: Config, overrides=overrides):
                c.strategy.exit_trend_ma20 = False
                c.strategy.exit_trend_ma60 = False
                c.strategy.exit_rank = False
                c.strategy.exit_market_extreme = True
                c.strategy.profit_target_pct = 0.02
                c.strategy.stop_atr_mult = 4.0
                c.strategy.stop_min = 0.03
                c.strategy.stop_max = 0.08
                c.strategy.max_hold_days = 15
                c.strategy.use_market_timing = False
                for key, value in overrides.items():
                    setattr(c.strategy, key, value)

            result = run_variant(f_variant, cfg, args.start, args.end, mutate)
            result["variant"] = f"{sig_name}__{exit_name}"
            rows.append(result)
            print(
                "[combo] {variant}: win={win:.3f} exp={expectancy:.4f} cagr={cagr:.4f} mdd={mdd:.3f} rounds={rounds}".format(
                    **result
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    cols = ["variant", "win", "expectancy", "avg_win", "avg_loss", "profit_factor", "cagr", "mdd", "rounds"]
    print(frame[cols].sort_values("expectancy", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
