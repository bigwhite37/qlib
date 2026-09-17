#!/usr/bin/env python3
"""In-process V2 configuration sweep.

Loads the panel, features, market state and predictions once, then runs the
real Qlib account engine for a set of pre-registered configurations.  Keeping
the setup out of the loop makes it cheap to test policy/position variants while
staying inside the 3 GB memory budget.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.bootstrap import init_local_qlib  # noqa: E402
from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import FeatureStore, build_features  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402
from lowvol_trend.ledger import ExecutionLedger  # noqa: E402
from lowvol_trend.metrics import (  # noqa: E402
    compute_frequency_stats,
    compute_nav_metrics,
    compute_round_trip_stats,
    nav_series,
    quarterly_returns,
)
from lowvol_trend.qlib_backtest import LowVolExchange  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from lowvol_trend.v2_strategy import (  # noqa: E402
    V2MarketAdapter,
    V2Strategy,
    add_composite_score,
    apply_gate,
    apply_v2_signals,
    policy_from_config,
    prediction_matrices,
    slim_for_v2,
)

from qlib.backtest import backtest  # noqa: E402
from qlib.backtest.executor import SimulatorExecutor  # noqa: E402


def mem(tag: str) -> None:
    try:
        import psutil

        print(f"[memory] {tag}: {psutil.Process().memory_info().rss / 1024**3:.2f} GiB", flush=True)
    except Exception:
        pass


def run_config(base_cfg, features, market, benchmark, exchange, policy, overrides: Dict[str, Any], label: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for key, value in overrides.items():
        if key == "initial_cash":
            cfg.execution.initial_cash = float(value)
            continue
        setattr(cfg.strategy, key, value)
    # The frozen per-position stop must come from the exit policy that also
    # produced the labels, not from the V0 defaults.
    cfg.strategy.stop_atr_mult = policy.stop_atr_mult
    cfg.strategy.stop_min = policy.stop_min
    cfg.strategy.stop_max = policy.stop_max
    ledger = ExecutionLedger(features, cfg)
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        trade_type=SimulatorExecutor.TT_PARAL,
        indicator_config={"show_indicator": False},
    )
    strategy = V2Strategy(features, cfg, ledger, market, policy=policy, trade_exchange=exchange)
    t0 = time.time()
    portfolio_dict, _ = backtest(
        start_time=cfg.data.backtest_start,
        end_time=cfg.data.evaluation_end,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=cfg.execution.initial_cash,
        exchange_kwargs={"exchange": exchange},
    )
    report, positions = portfolio_dict["1day"]
    nav = nav_series(report, cfg.execution.initial_cash)
    nav_metrics = compute_nav_metrics(nav)
    round_trips = ledger.round_trips_frame()
    rt = compute_round_trip_stats(round_trips)
    freq = compute_frequency_stats(positions, cfg)
    quarters = quarterly_returns(nav)
    complete = quarters[quarters["complete"]]
    out = {
        "label": label,
        "cagr": nav_metrics.cagr,
        "vol": nav_metrics.annual_vol,
        "mdd": nav_metrics.max_drawdown,
        "win": rt.win_rate,
        "trades": rt.n,
        "avg_ret": rt.avg_return,
        "hold": rt.avg_hold_days,
        "f63": freq.rolling_63_min,
        "f252": freq.rolling_252_min,
        "neg_q": int((complete["return"] <= 0).sum()) if len(complete) else 0,
        "n_q": int(len(complete)),
        "seconds": round(time.time() - t0, 1),
    }
    del ledger, strategy, executor, report, positions, round_trips, quarters
    gc.collect()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5_predictions.parquet")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research/sweep.csv")
    parser.add_argument("--stage", default="core")
    parser.add_argument("--eval-start", default="2019-01-02")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    cfg.strategy.use_corr_filter = False
    cfg.strategy.use_drawdown_control = False
    cfg.data.backtest_start = args.eval_start
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_v2_features(panel, cfg)
    print("[features] lean V2 builder used", flush=True)
    base = v2_base_mask(features, cfg)
    add_composite_score(features, base)
    market_state = compute_v2_market_state(features, base, cfg)
    market = V2MarketAdapter(market_state)
    market.median_return = market_state.proxy_return.astype(np.float64)
    mem("after slim")
    predictions = pd.read_parquet(
        args.predictions, columns=["datetime", "symbol_index", "pred_rel", "pred_win", "pred_return"]
    )
    rel_mat, win_mat = prediction_matrices(features, predictions)
    del predictions
    gc.collect()
    composite = features.arrays["composite_rank"]
    from lowvol_trend.features import row_pct_rank as _row_rank

    def _blended(weight: float) -> FeatureStore:
        signal = base & np.isfinite(rel_mat)
        model_rank = _row_rank(np.where(signal, rel_mat, np.nan), signal)
        blended = np.where(signal, weight * model_rank + (1.0 - weight) * composite, np.nan)
        blended = np.where(signal & ~np.isfinite(model_rank), composite, blended)
        return FeatureStore(
            panel=features.panel,
            cfg=features.cfg,
            market=features.market,
            arrays={**dict(features.arrays), "entry_signal": signal, "entry_score": blended.astype(np.float32)},
            diagnostics=dict(features.diagnostics),
        )

    print("[sweep] rank matrices ready; candidates", int((base & np.isfinite(rel_mat)).sum()), flush=True)
    mem("after signals")
    benchmark = pd.Series(market_state.proxy_return.astype(np.float64), index=features.dates, name="proxy")
    open_cost = cfg.execution.commission_rate + cfg.execution.transfer_fee_rate + cfg.execution.friction_bps_per_side / 10_000.0
    close_cost = 0.001 + open_cost
    exchange = LowVolExchange(
        features,
        cfg,
        freq="day",
        start_time=cfg.data.backtest_start,
        end_time=cfg.data.evaluation_end,
        codes=features.symbols,
        deal_price="$close",
        limit_threshold=None,
        volume_threshold=None,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=cfg.execution.min_commission,
        trade_unit=cfg.execution.lot_size,
        impact_cost=0.0,
    )
    mem("after exchange")

    policies = {
        "tp5": dict(use_stop=False, profit_target_pct=0.05, use_trailing=False, max_hold_days=20),
        "tp3": dict(use_stop=False, profit_target_pct=0.03, use_trailing=False, max_hold_days=20),
        "tp2": dict(use_stop=False, profit_target_pct=0.02, use_trailing=False, max_hold_days=20),
        "tp8": dict(use_stop=False, profit_target_pct=0.08, use_trailing=False, max_hold_days=20),
        "tp4_h30": dict(use_stop=False, profit_target_pct=0.04, use_trailing=False, max_hold_days=30),
        "tp5_stop8": dict(use_stop=True, stop_atr_mult=0.0, stop_min=0.08, stop_max=0.08, use_trailing=False, profit_target_pct=0.05, max_hold_days=20),
        "tp3_trail": dict(use_stop=False, profit_target_pct=0.03, use_trailing=True, trailing_activate_mult=1.0, trailing_distance_mult=0.5, max_hold_days=20),
        "tp3_stop6": dict(use_stop=True, stop_atr_mult=0.0, stop_min=0.06, stop_max=0.06, use_trailing=False, profit_target_pct=0.03, max_hold_days=20),
        "tp2_stop8": dict(use_stop=True, stop_atr_mult=0.0, stop_min=0.08, stop_max=0.08, use_trailing=False, profit_target_pct=0.02, max_hold_days=20),
        "tp2_stop4": dict(use_stop=True, stop_atr_mult=0.0, stop_min=0.04, stop_max=0.04, use_trailing=False, profit_target_pct=0.02, max_hold_days=20),
        "tp4_stop8": dict(use_stop=True, stop_atr_mult=0.0, stop_min=0.08, stop_max=0.08, use_trailing=False, profit_target_pct=0.04, max_hold_days=20),
        "tp5_stop12": dict(use_stop=True, stop_atr_mult=0.0, stop_min=0.12, stop_max=0.12, use_trailing=False, profit_target_pct=0.05, max_hold_days=20),
        "tp5_h40": dict(use_stop=False, use_trailing=False, profit_target_pct=0.05, max_hold_days=40),
        "tp8_h40": dict(use_stop=False, use_trailing=False, profit_target_pct=0.08, max_hold_days=40),
        "tp35_h40": dict(use_stop=False, use_trailing=False, profit_target_pct=0.035, max_hold_days=40),
        "tp5_h30": dict(use_stop=False, use_trailing=False, profit_target_pct=0.05, max_hold_days=30),
    }
    base_overrides = {
        "max_positions": 12,
        "max_new_per_day": 3,
        "max_gross": 0.95,
        "use_vol_target": True,
        "vol_target": 0.10,
        "max_weight": 0.08,
    }
    configs: List[tuple] = []
    if args.stage == "core":
        for pname in ("tp5", "tp3", "tp2", "tp8", "tp4_h30", "tp5_stop8", "tp3_trail"):
            configs.append((pname, pname, dict(base_overrides)))
    elif args.stage == "risk":
        for gross, vol, pos in ((0.95, 0.10, 12), (0.80, 0.10, 12), (0.95, 0.08, 12), (0.95, 0.06, 12), (0.95, 0.10, 20), (0.95, 0.10, 6)):
            configs.append(
                (
                    "tp5",
                    f"tp5_g{gross}_v{vol}_p{pos}",
                    {
                        "max_positions": pos,
                        "max_new_per_day": max(2, pos // 4),
                        "max_gross": gross,
                        "use_vol_target": True,
                        "vol_target": vol,
                        "max_weight": min(0.15, 1.2 / max(pos, 1)),
                    },
                )
            )
    elif args.stage == "round2":
        wide = dict(base_overrides)
        configs.extend(
            [
                ("tp5_stop8", "r2_tp5_stop8", dict(wide)),
                ("tp5_stop12", "r2_tp5_stop12", dict(wide)),
                ("tp3_stop6", "r2_tp3_stop6", dict(wide)),
                ("tp4_stop8", "r2_tp4_stop8", dict(wide)),
                ("tp2_stop8", "r2_tp2_stop8", dict(wide)),
                ("tp2_stop4", "r2_tp2_stop4", dict(wide)),
                ("tp5_stop8", "r2_p20", {**wide, "max_positions": 20, "max_new_per_day": 4, "max_weight": 0.05}),
                ("tp2_stop8", "r2_p20_tp2", {**wide, "max_positions": 20, "max_new_per_day": 4, "max_weight": 0.05}),
                ("tp5_stop8", "r2_gross80", {**wide, "max_gross": 0.80}),
                ("tp5_stop8", "r2_vol8", {**wide, "vol_target": 0.08}),
                ("tp5_stop8", "r2_vol12", {**wide, "vol_target": 0.12}),
                ("tp5_stop8", "r2_notiming", {**wide, "use_market_timing": False}),
                ("tp5_stop8", "r2_ddctl", {**wide, "use_drawdown_control": True, "drawdown_tiers": ((0.03, 0.70), (0.05, 0.45), (0.07, 0.20))}),
                ("tp5_stop8", "r2_ddctl_notiming", {**wide, "use_drawdown_control": True, "drawdown_tiers": ((0.03, 0.70), (0.05, 0.45), (0.07, 0.20)), "use_market_timing": False}),
                ("tp5_stop8", "r2_p20_ddctl", {**wide, "max_positions": 20, "max_new_per_day": 4, "max_weight": 0.05, "use_drawdown_control": True, "drawdown_tiers": ((0.03, 0.70), (0.05, 0.45), (0.07, 0.20))}),
            ]
        )
    elif args.stage == "gates":
        wide = dict(base_overrides)
        configs.extend(
            [
                ("tp5_stop8", "g_p00_m000", {**wide, "p_min": 0.0, "mu_min": -1.0}),
                ("tp5_stop8", "g_p45_m000", {**wide, "p_min": 0.45, "mu_min": -1.0}),
                ("tp5_stop8", "g_p50_m000", {**wide, "p_min": 0.50, "mu_min": -1.0}),
                ("tp5_stop8", "g_p55_m000", {**wide, "p_min": 0.55, "mu_min": -1.0}),
                ("tp5_stop8", "g_p00_m000_tp2", {**wide, "p_min": 0.0, "mu_min": -1.0}),
                ("tp2_stop8", "g_p45_m000_tp2", {**wide, "p_min": 0.45, "mu_min": -1.0}),
                ("tp2_stop8", "g_p50_m000_tp2", {**wide, "p_min": 0.50, "mu_min": -1.0}),
                ("tp2_stop8", "g_p55_m000_tp2", {**wide, "p_min": 0.55, "mu_min": -1.0}),
                ("tp3_stop6", "g_p50_m000_tp3", {**wide, "p_min": 0.50, "mu_min": -1.0}),
                ("tp5_stop8", "g_p50_m005", {**wide, "p_min": 0.50, "mu_min": 0.005}),
                ("tp5_stop8", "g_p50_m010", {**wide, "p_min": 0.50, "mu_min": 0.010}),
                ("tp5_stop8", "g_p55_ddctl", {**wide, "p_min": 0.55, "mu_min": -1.0, "use_drawdown_control": True, "drawdown_tiers": ((0.03, 0.70), (0.05, 0.45), (0.07, 0.20))}),
            ]
        )
    elif args.stage == "round3":
        wide = dict(base_overrides)
        configs.extend(
            [
                ("tp5_h40", "n3_blend50", {**wide, "rank_blend": 0.5}),
                ("tp5_h40", "n3_blend100", {**wide, "rank_blend": 1.0}),
                ("tp5_h40", "n3_blend0", {**wide, "rank_blend": 0.0}),
                ("tp5_h40", "n3_blend25", {**wide, "rank_blend": 0.25}),
                ("tp5_h40", "n3_blend75", {**wide, "rank_blend": 0.75}),
                ("tp5_h40", "n3_blend50_vol13", {**wide, "rank_blend": 0.5, "vol_target": 0.13}),
                ("tp5_h40", "n3_blend50_vol16", {**wide, "rank_blend": 0.5, "vol_target": 0.16}),
                ("tp5_h40", "n3_blend50_cash1m", {**wide, "rank_blend": 0.5, "initial_cash": 1000000.0}),
                ("tp5_h40", "n3_blend50_cash1m_vol13", {**wide, "rank_blend": 0.5, "initial_cash": 1000000.0, "vol_target": 0.13}),
                ("tp5_h40", "n3_blend50_pos20", {**wide, "rank_blend": 0.5, "max_positions": 20, "max_new_per_day": 4, "max_weight": 0.05}),
                ("tp5_h40", "n3_blend50_gross60", {**wide, "rank_blend": 0.5, "max_gross": 0.60}),
                ("tp5_h30", "n3_h30_blend50", {**wide, "rank_blend": 0.5}),
                ("tp8_h40", "n3_tp8_blend50", {**wide, "rank_blend": 0.5}),
                ("tp35_h40", "n3_tp35_blend50", {**wide, "rank_blend": 0.5}),
            ]
        )
    elif args.stage == "timing":
        for timing in (1, 0):
            overrides = dict(base_overrides)
            overrides["use_market_timing"] = bool(timing)
            configs.append(("tp5", f"tp5_timing{timing}", overrides))
    rows = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for pname, label, overrides in configs:
        policy = policy_from_config(cfg, **policies[pname])
        mu_min = float(overrides.pop("mu_min", -1.0))
        p_min = float(overrides.pop("p_min", 0.0))
        blend = float(overrides.pop("rank_blend", 1.0))
        gated = _blended(blend) if blend < 1.0 else apply_gate(features, base, rel_mat, win_mat, mu_min, p_min)
        result = run_config(cfg, gated, market, benchmark, exchange, policy, overrides, label)
        result["mu_min"] = mu_min
        result["p_min"] = p_min
        result["blend"] = blend
        del gated
        gc.collect()
        rows.append(result)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(
            f"[sweep] {label}: cagr={result['cagr']:+.4f} vol={result['vol']:.4f} mdd={result['mdd']:.4f} "
            f"win={result['win']:.4f} trades={result['trades']} avg={result['avg_ret']:+.4f} hold={result['hold']:.1f} "
            f"f63={result['f63']:.3f} negQ={result['neg_q']}/{result['n_q']} ({result['seconds']}s)",
            flush=True,
        )
        mem(f"after {label}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
