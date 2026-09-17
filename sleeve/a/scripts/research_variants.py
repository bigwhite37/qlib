#!/usr/bin/env python3
"""In-process research harness for strategy-semantics ablations.

Loads the panel and features once, then runs the same Qlib executor/Exchange
under a set of pre-registered rule variants.  This keeps data preparation out
of the loop and makes it cheap to test whether an exit rule helps on the
development / validation segments before freezing anything.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

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
from lowvol_trend.ledger import ExecutionLedger  # noqa: E402
from lowvol_trend.metrics import (  # noqa: E402
    compute_frequency_stats,
    compute_nav_metrics,
    compute_round_trip_stats,
    nav_series,
    quarterly_returns,
)
from lowvol_trend.qlib_backtest import LowVolExchange, LowVolTrendStrategy  # noqa: E402

from qlib.backtest import backtest  # noqa: E402
from qlib.backtest.executor import SimulatorExecutor  # noqa: E402


def run_variant(features, base_cfg: Config, start: str, end: str, mutate) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    mutate(cfg)
    ledger = ExecutionLedger(features, cfg)
    benchmark = pd.Series(features.market.median_return, index=features.dates, name="market_proxy")
    open_cost = (
        cfg.execution.commission_rate
        + cfg.execution.transfer_fee_rate
        + cfg.execution.friction_bps_per_side / 10_000.0
    )
    close_cost = (
        0.001
        + cfg.execution.commission_rate
        + cfg.execution.transfer_fee_rate
        + cfg.execution.friction_bps_per_side / 10_000.0
    )
    exchange = LowVolExchange(
        features,
        cfg,
        freq="day",
        start_time=start,
        end_time=end,
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
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        trade_type=SimulatorExecutor.TT_PARAL,
        indicator_config={"show_indicator": False},
    )
    strategy = LowVolTrendStrategy(features, cfg, ledger, features.market, trade_exchange=exchange)
    portfolio_dict, _ = backtest(
        start_time=start,
        end_time=end,
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
    rt_stats = compute_round_trip_stats(round_trips)
    frequency = compute_frequency_stats(positions, cfg)
    quarters = quarterly_returns(nav)
    complete = quarters[quarters["complete"]]
    if len(round_trips):
        winners = round_trips[round_trips["profit"] > 0]["return_pct"]
        losers = round_trips[round_trips["profit"] <= 0]["return_pct"]
        avg_win = float(winners.mean()) if len(winners) else 0.0
        avg_loss = float(losers.mean()) if len(losers) else 0.0
        expectancy = float(round_trips["return_pct"].mean())
    else:
        avg_win = avg_loss = expectancy = 0.0
    return {
        "cagr": nav_metrics.cagr,
        "vol": nav_metrics.annual_vol,
        "mdd": nav_metrics.max_drawdown,
        "win": rt_stats.win_rate,
        "rounds": rt_stats.n,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": rt_stats.profit_factor,
        "f63": frequency.rolling_63_min,
        "f252": frequency.rolling_252_min,
        "neg_quarters": int((complete["return"] <= 0).sum()) if len(complete) else 0,
        "final_nav": nav_metrics.final_nav,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[research] building features ...", flush=True)
    features = build_features(panel, cfg)

    def noop(cfg: Config) -> None:
        pass

    variants: List[tuple] = [
        ("v0_frozen", noop),
        ("no_trend_exits", lambda c: (setattr(c.strategy, "exit_trend_ma60", False), setattr(c.strategy, "exit_trend_ma20", False))),
        ("no_rank_exit", lambda c: setattr(c.strategy, "exit_rank", False)),
        ("no_trend_no_rank", lambda c: (setattr(c.strategy, "exit_trend_ma60", False), setattr(c.strategy, "exit_trend_ma20", False), setattr(c.strategy, "exit_rank", False))),
        ("profit_target_3pct", lambda c: setattr(c.strategy, "profit_target_pct", 0.03)),
        ("profit_target_5pct", lambda c: setattr(c.strategy, "profit_target_pct", 0.05)),
        ("tight_stop", lambda c: (setattr(c.strategy, "stop_atr_mult", 1.5), setattr(c.strategy, "stop_min", 0.02), setattr(c.strategy, "stop_max", 0.04))),
        ("hold10", lambda c: setattr(c.strategy, "max_hold_days", 10)),
        ("no_trend_hold10", lambda c: (setattr(c.strategy, "exit_trend_ma60", False), setattr(c.strategy, "exit_trend_ma20", False), setattr(c.strategy, "max_hold_days", 10))),
    ]
    rows = []
    for tag, mutate in variants:
        result = run_variant(features, cfg, args.start, args.end, mutate)
        result["variant"] = tag
        rows.append(result)
        print(f"[research] {tag}: {result}", flush=True)
    frame = pd.DataFrame(rows)[["variant", "cagr", "vol", "mdd", "win", "rounds", "f63", "f252", "neg_quarters", "final_nav"]]
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
