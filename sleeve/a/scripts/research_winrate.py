#!/usr/bin/env python3
"""Focused win-rate research on the development segment.

The objective is a complete round-trip win rate >= 50% after fees, without
destroying expectancy.  We explore a small, pre-registered set of exit
semantics around profit targets, trend-failure exits and holding periods.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import pandas as pd  # noqa: E402

from lowvol_trend.config import Config, load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402

# Make ``scripts`` importable when run as a file.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.research_variants import run_variant  # noqa: E402


def _set(cfg: Config, **kwargs) -> None:
    for key, value in kwargs.items():
        setattr(cfg.strategy, key, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2019-12-31")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[winrate] building features ...", flush=True)
    features = build_features(panel, cfg)

    def make(pt: float, no_trend: bool = True, no_rank: bool = False, hold: int = 20, stop: float = 2.5):
        def mutate(c: Config) -> None:
            c.strategy.profit_target_pct = pt
            c.strategy.exit_trend_ma20 = not no_trend
            c.strategy.exit_trend_ma60 = not no_trend
            c.strategy.exit_rank = not no_rank
            c.strategy.max_hold_days = hold
            c.strategy.stop_atr_mult = stop

        return mutate

    variants = [
        ("v0_frozen", lambda c: None),
        ("no_trend", make(0.0, no_trend=True)),
        ("no_trend_pt1", make(0.01, no_trend=True)),
        ("no_trend_pt1.5", make(0.015, no_trend=True)),
        ("no_trend_pt2", make(0.02, no_trend=True)),
        ("no_trend_pt2.5", make(0.025, no_trend=True)),
        ("no_trend_pt3", make(0.03, no_trend=True)),
        ("no_trend_pt2_no_rank", make(0.02, no_trend=True, no_rank=True)),
        ("no_trend_pt2_hold5", make(0.02, no_trend=True, hold=5)),
        ("no_trend_pt2_hold10", make(0.02, no_trend=True, hold=10)),
        ("no_trend_pt2_stop1.5", make(0.02, no_trend=True, stop=1.5)),
        ("no_trend_pt2_stop3.5", make(0.02, no_trend=True, stop=3.5)),
    ]
    rows: List[Dict] = []
    for tag, mutate in variants:
        result = run_variant(features, cfg, args.start, args.end, mutate)
        result["variant"] = tag
        rows.append(result)
        print(
            "[winrate] {variant}: win={win:.3f} exp={expectancy:.4f} cagr={cagr:.4f} rounds={rounds}".format(
                **result
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    cols = ["variant", "win", "expectancy", "avg_win", "avg_loss", "profit_factor", "cagr", "mdd", "rounds"]
    print(frame[cols].sort_values("win", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
