#!/usr/bin/env python3
"""Compare candidate ranking signals under the frozen V2 exit policy.

The round-1 report showed the simple composite (low volatility, low liquidity,
reversal) delivers about +1.0pp of 20-day alpha while the rolling LightGBM model
delivered only +0.45pp.  This harness settles the question with the same policy
simulator the labels and the account use.
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
from lowvol_trend.features import row_pct_rank  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402
from lowvol_trend.v2_policy import ExitPolicy, policy_stats, simulate_entries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5_predictions.parquet")
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research/rank_compare.csv")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_v2_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    extreme = market.effective_state == "extreme"

    close = pd.DataFrame(features.arr("close"), index=panel.dates, columns=panel.symbols, copy=False)
    volume = pd.DataFrame(panel.field("volume"), index=panel.dates, columns=panel.symbols, copy=False)
    high = pd.DataFrame(panel.field("high"), index=panel.dates, columns=panel.symbols, copy=False)
    low = pd.DataFrame(panel.field("low"), index=panel.dates, columns=panel.symbols, copy=False)
    ret1 = close.pct_change(fill_method=None)
    vol20 = ret1.rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)
    amp20 = ((high - low) / close).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    amount = features.arr("amount20_yuan")
    log_amount = np.log(np.clip(amount.astype(np.float64), 1.0, None)).astype(np.float32)
    dist_ma60 = (close / close.rolling(60, min_periods=60).mean() - 1.0).to_numpy(dtype=np.float32)
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    vol_surge = (volume / volume.rolling(20, min_periods=20).mean()).to_numpy(dtype=np.float32)
    r_vol = row_pct_rank(-vol20, base)
    r_amp = row_pct_rank(-amp20, base)
    r_amt = row_pct_rank(-log_amount, base)
    r_dist = row_pct_rank(-dist_ma60, base)
    r_ret20 = row_pct_rank(-ret20, base)
    r_liq = row_pct_rank(amount, base)

    scores: Dict[str, np.ndarray] = {
        "illiq": r_amt,
        "lowvol": r_vol,
        "combo3": (r_vol + r_amt + r_dist) / 3.0,
        "combo5": (r_vol + r_amp + r_amt + r_dist + r_ret20) / 5.0,
        "combo_lowvol_illiq": (r_vol + r_amt) / 2.0,
        "amt_only": r_liq,
    }
    pred = pd.read_parquet(args.predictions, columns=["datetime", "symbol_index", "pred_rel"])
    model_score = np.full(base.shape, np.nan, dtype=np.float32)
    date_index = {d: i for i, d in enumerate(panel.dates)}
    t_pos = pd.to_datetime(pred["datetime"]).map(date_index).to_numpy()
    j_pos = pred["symbol_index"].to_numpy(dtype=np.int64)
    model_score[t_pos, j_pos] = pred["pred_rel"].to_numpy(dtype=np.float32)
    scores["model"] = model_score
    # rank the model score inside each day so it can be blended with the ranks
    model_rank = row_pct_rank(model_score, base)
    scores["combo5_plus_model"] = 0.5 * scores["combo5"] + 0.5 * model_rank
    scores["combo3_plus_model"] = 0.5 * scores["combo3"] + 0.5 * model_rank

    policies = {
        "live_tp5_stop8_h20": ExitPolicy(profit_target_pct=0.05, stop_atr_mult=0.0, stop_min=0.08, stop_max=0.08,
                                         use_trailing=False, max_hold_days=20),
        "notp_h20": ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=None, max_hold_days=20),
        "tp5_notp_h40": ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=0.05, max_hold_days=40),
        "notp_h40": ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=None, max_hold_days=40),
    }
    t_lo = int(np.searchsorted(panel.dates, pd.Timestamp(args.start)))
    t_hi = int(np.searchsorted(panel.dates, pd.Timestamp(args.end), side="right"))
    rows: List[dict] = []
    for q in (0.01, 0.05, 0.10, 0.20):
        for name, score in scores.items():
            thr = np.full(base.shape[0], np.nan, dtype=np.float32)
            finite = base & np.isfinite(score)
            for t in range(t_lo, t_hi):
                row = score[t]
                m = finite[t]
                if m.sum() < 50:
                    continue
                thr[t] = np.nanquantile(row[m], 1.0 - q)
            mask = finite & (score >= thr[:, None])
            mask[:t_lo] = False
            mask[t_hi:] = False
            for pname, policy in policies.items():
                frame = simulate_entries(features, cfg, mask, policy, market_extreme=extreme,
                                         start_index=t_lo, end_index=t_hi)
                stats = policy_stats(frame)
                stats.update({"rank": name, "topq": q, "policy": pname})
                rows.append(stats)
                print(
                    f"[rank] {name:20s} top{q:>4.0%} {pname:18s} n={stats['n']:7d} win={stats.get('win', float('nan')):.4f} "
                    f"ret={stats.get('ret', float('nan')):+.4f} hold={stats.get('hold', float('nan')):5.1f} "
                    f"per_year={stats.get('per_year', 0):7.0f}",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(args.output, index=False)
    table = pd.DataFrame(rows)
    table["annual_contrib"] = table["ret"] * table["per_year"]
    print(table.sort_values("annual_contrib", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
