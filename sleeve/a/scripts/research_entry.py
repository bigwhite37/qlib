#!/usr/bin/env python3
"""Entry-semantics research for raising the complete-round win rate.

All variants share one execution stack and one frozen exit policy, and differ
only in the entry conditions / cross-sectional score.  The default exit policy
is deliberately closer to a high-hit-rate design: no trend-failure exits, a
small profit target, a wider ATR stop and no rank exit.
"""

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


def _with_signal(base: FeatureStore, signal: np.ndarray, score: np.ndarray) -> FeatureStore:
    arrays = dict(base.arrays)
    arrays["entry_signal"] = signal.astype(bool)
    arrays["entry_score"] = np.where(signal, score, np.nan).astype(np.float32)
    return FeatureStore(panel=base.panel, cfg=base.cfg, market=base.market, arrays=arrays, diagnostics=dict(base.diagnostics))


def _exit_policy(c: Config) -> None:
    c.strategy.profit_target_pct = 0.02
    c.strategy.exit_trend_ma20 = False
    c.strategy.exit_trend_ma60 = False
    c.strategy.exit_rank = False
    c.strategy.stop_atr_mult = 3.5
    c.strategy.stop_min = 0.03
    c.strategy.stop_max = 0.06
    c.strategy.max_hold_days = 15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--exit-policy", choices=["win", "v0"], default="win")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[entry] building features ...", flush=True)
    f = build_features(panel, cfg)

    close = pd.DataFrame(f.arr("close"), index=f.dates, columns=f.symbols)
    base = f.arr("base_pass")
    trend = (
        (f.arr("close") > f.arr("ma60"))
        & (f.arr("ma20") > f.arr("ma60"))
        & (f.arr("ma60") > f.arr("ma60_prev5"))
    )
    repair = (f.arr("ret1") > 0) & (f.arr("ret1") <= cfg.signal.max_daily_gain) & (f.arr("close") > f.arr("ma5"))
    dd = f.arr("dist_high20")
    atr = f.arr("atr20")
    drawdown_ok = (dd >= cfg.signal.min_drawdown_atr * atr) & (dd <= cfg.signal.max_drawdown_atr * atr)
    rs_ok = np.isfinite(f.arr("rs_rank")) & (f.arr("rs_rank") >= 0.70)
    above_ma20 = f.arr("close") > f.arr("ma20")
    strong_market = ~np.isin(f.market.effective_state, ["weak", "extreme"])[:, None]

    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rsi = (100.0 - 100.0 / (1.0 + gain / (loss + 1e-12))).to_numpy(dtype=np.float32)
    ret5 = close.pct_change(5, fill_method=None).to_numpy(dtype=np.float32)
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    low_vol_rank = np.where(np.isfinite(f.arr("vol60")), f.arr("vol_rank"), np.nan)
    inv_vol_score = 1.0 - np.nan_to_num(f.arr("vol_rank"), nan=1.0)

    def sig(cond: np.ndarray, score: np.ndarray) -> np.ndarray:
        return base & cond

    variants: List[Dict] = [
        {
            "tag": "v0_signal",
            "signal": f.arr("entry_signal"),
            "score": f.arr("entry_score"),
        },
        {
            "tag": "trend_rs_above_ma20",
            "signal": sig(trend & rs_ok & above_ma20, None),
            "score": f.arr("score"),
        },
        {
            "tag": "trend_rs_above_ma20_drawdown",
            "signal": sig(trend & rs_ok & above_ma20 & drawdown_ok, None),
            "score": f.arr("score"),
        },
        {
            "tag": "trend_rs_above_ma20_repair",
            "signal": sig(trend & rs_ok & above_ma20 & repair, None),
            "score": f.arr("score"),
        },
        {
            "tag": "trend_rs_above_ma20_dd_repair",
            "signal": sig(trend & rs_ok & above_ma20 & drawdown_ok & repair, None),
            "score": f.arr("score"),
        },
        {
            "tag": "trend_low_vol_dd_repair",
            "signal": sig(trend & (low_vol_rank <= 0.30) & drawdown_ok & repair, None),
            "score": row_pct_rank(inv_vol_score, base),
        },
        {
            "tag": "trend_rsi_oversold_repair",
            "signal": sig(trend & (rsi <= 45) & repair, None),
            "score": row_pct_rank(-rsi, base),
        },
        {
            "tag": "trend_reversal5_repair",
            "signal": sig(trend & (ret5 <= -0.03) & repair, None),
            "score": row_pct_rank(-ret5, base),
        },
        {
            "tag": "trend_reversal20_repair",
            "signal": sig(trend & (ret20 <= -0.05) & repair, None),
            "score": row_pct_rank(-ret20, base),
        },
        {
            "tag": "trend_rs_strong_market",
            "signal": sig(trend & rs_ok & strong_market, None),
            "score": f.arr("score"),
        },
        {
            "tag": "trend_rs_above_ma20_strong_market",
            "signal": sig(trend & rs_ok & above_ma20 & strong_market, None),
            "score": f.arr("score"),
        },
        {
            "tag": "trend_low_vol_strong_market",
            "signal": sig(trend & (low_vol_rank <= 0.30) & strong_market & drawdown_ok & repair, None),
            "score": row_pct_rank(inv_vol_score, base),
        },
    ]
    rows = []
    for item in variants:
        signal = item["signal"]
        score = item.get("score")
        if score is None:
            score = f.arr("score")
        score = np.asarray(score, dtype=np.float32)
        f_variant = _with_signal(f, signal, score)
        exit_mutate = _exit_policy if args.exit_policy == "win" else (lambda c: None)
        result = run_variant(f_variant, cfg, args.start, args.end, exit_mutate)
        result["variant"] = item["tag"]
        rows.append(result)
        print(
            "[entry] {variant}: win={win:.3f} exp={expectancy:.4f} cagr={cagr:.4f} rounds={rounds}".format(**result),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    cols = ["variant", "win", "expectancy", "avg_win", "avg_loss", "rounds", "cagr", "mdd"]
    print(frame[cols].sort_values("win", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
