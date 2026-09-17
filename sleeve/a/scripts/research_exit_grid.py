#!/usr/bin/env python3
"""Exit-policy grid for the high-win-rate reversal entry."""

from __future__ import annotations

import argparse
import copy
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
    print("[exit-grid] building features ...", flush=True)
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

    grid = [
        (0.020, 1.5, 0.02, 0.04, 15, False),
        (0.020, 2.0, 0.02, 0.04, 15, False),
        (0.020, 2.5, 0.02, 0.05, 15, False),
        (0.020, 3.0, 0.02, 0.05, 15, False),
        (0.025, 2.0, 0.02, 0.04, 15, False),
        (0.015, 2.0, 0.02, 0.04, 15, False),
        (0.020, 2.0, 0.02, 0.04, 10, False),
        (0.020, 2.0, 0.02, 0.04, 20, False),
        (0.015, 1.5, 0.02, 0.04, 10, False),
        (0.025, 2.0, 0.02, 0.04, 15, True),
        (0.020, 2.0, 0.025, 0.05, 15, False),
        (0.030, 2.0, 0.02, 0.04, 15, False),
    ]
    rows: List[Dict] = []
    for pt, stop, smin, smax, hold, rank in grid:
        def mutate(c: Config, pt=pt, stop=stop, smin=smin, smax=smax, hold=hold, rank=rank):
            c.strategy.profit_target_pct = pt
            c.strategy.exit_trend_ma20 = False
            c.strategy.exit_trend_ma60 = False
            c.strategy.exit_rank = rank
            c.strategy.stop_atr_mult = stop
            c.strategy.stop_min = smin
            c.strategy.stop_max = smax
            c.strategy.max_hold_days = hold

        result = run_variant(f_variant, cfg, args.start, args.end, mutate)
        result["tag"] = f"pt{pt}_stop{stop}_{smin}-{smax}_hold{hold}_rank{int(rank)}"
        rows.append(result)
        print(
            "[exit-grid] {tag}: win={win:.3f} exp={expectancy:.4f} cagr={cagr:.4f} rounds={rounds}".format(
                **result
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    cols = ["tag", "win", "expectancy", "avg_win", "avg_loss", "profit_factor", "cagr", "mdd", "rounds"]
    print(frame[cols].sort_values(["win", "expectancy"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
