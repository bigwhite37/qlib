#!/usr/bin/env python3
"""Run the V0 rule strategy through Qlib's backtest engine.

Usage:
    python scripts/run_qrun_v0.py --start 2016-01-04 --end 2016-06-30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.bootstrap import init_local_qlib  # noqa: E402
from lowvol_trend.config import Config, load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import LaggedFeatureStore, build_features  # noqa: E402
from lowvol_trend.ledger import ExecutionLedger  # noqa: E402
from lowvol_trend.metrics import (  # noqa: E402
    acceptance_report,
    build_constraints_frame,
    compute_frequency_stats,
    compute_nav_metrics,
    compute_round_trip_stats,
    nav_series,
    quarterly_returns,
    segment_metrics,
)
from lowvol_trend.qlib_backtest import LowVolExchange, LowVolTrendStrategy  # noqa: E402
from lowvol_trend.v1 import apply_v1_signals, build_candidate_frame, rolling_predict  # noqa: E402

from qlib.backtest import backtest  # noqa: E402
from qlib.backtest.executor import SimulatorExecutor  # noqa: E402
from qlib.contrib.evaluate import indicator_analysis, risk_analysis  # noqa: E402
from qlib.workflow import R  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=str(ROOT / "output" / "smoke"))
    parser.add_argument("--experiment", default="sleeve_a_v0")
    parser.add_argument("--recorder", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--conservative-st", action="store_true")
    parser.add_argument("--show-indicator", action="store_true")
    parser.add_argument("--signal", choices=["v0", "v1"], default="v0")
    parser.add_argument("--baseline", action="store_true", help="fixed-position rule baseline")
    parser.add_argument("--v1-cache", default=str(ROOT / "cache" / "v1_predictions.parquet"))
    parser.add_argument("--v1-refresh", action="store_true")
    parser.add_argument("--v1-gate", type=float, default=0.60)
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--friction-bps", type=float, default=None)
    parser.add_argument("--signal-lag", type=int, default=0, help="stress: shift signal/exit features by N trading days")
    return parser.parse_args()


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value



def _signal_frame(features):
    import numpy as np

    signal = features.arr("entry_signal")
    score = features.arr("entry_score")
    t_idx, j_idx = np.nonzero(signal)
    if len(t_idx) == 0:
        return pd.DataFrame(columns=["datetime", "instrument", "score"])
    return pd.DataFrame(
        {
            "datetime": features.dates[t_idx],
            "instrument": np.asarray(features.symbols, dtype=object)[j_idx],
            "score": score[t_idx, j_idx],
        }
    ).sort_values(["datetime", "instrument"])


def main():
    args = parse_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    if args.conservative_st:
        cfg.execution.unknown_st_conservative = True
    if args.baseline:
        cfg.strategy.use_market_timing = False
        cfg.strategy.use_vol_target = False
        cfg.strategy.use_drawdown_control = False
    if args.initial_cash is not None:
        cfg.execution.initial_cash = float(args.initial_cash)
    if args.friction_bps is not None:
        cfg.execution.friction_bps_per_side = float(args.friction_bps)
    end = args.end or cfg.data.evaluation_end
    start = args.start
    print("[qlib]", init_local_qlib(cfg))
    t0 = time.time()
    with DuckDBPanelLoader(cfg, use_cache=not args.no_cache) as loader:
        panel = loader.load(refresh_cache=False)
    print(f"[data] {panel.n_dates} dates x {panel.n_symbols} symbols in {time.time()-t0:.1f}s; cache={panel.source_info.get('cache')}")
    t0 = time.time()
    features = build_features(panel, cfg)
    print(f"[features] built in {time.time()-t0:.1f}s; last day entry candidates={features.diagnostics['n_entry_last']}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    signal_tag = "baseline" if args.baseline else args.signal
    recorder_name = args.recorder or f"{signal_tag}_{start}_{end}_{int(time.time())}"
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

    v1_artifacts = None
    with R.start(experiment_name=args.experiment, recorder_name=recorder_name):
        recorder = R.get_recorder()
        if args.signal == "v1":
            cache_path = Path(args.v1_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path = cache_path.with_suffix(".metrics.csv")
            boundaries_path = cache_path.with_suffix(".boundaries.csv")
            calibration_path = cache_path.with_suffix(".calibration.csv")
            if cache_path.exists() and not args.v1_refresh:
                from lowvol_trend.v1 import V1Artifacts

                v1_artifacts = V1Artifacts(
                    predictions=pd.read_parquet(cache_path),
                    quarter_metrics=pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else [],
                    feature_names=[],
                    boundary_checks=pd.read_csv(boundaries_path).to_dict(orient="records") if boundaries_path.exists() else [],
                    calibration=pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame(),
                )
                print(f"[v1] loaded cached predictions: {len(v1_artifacts.predictions)} rows")
            else:
                t0 = time.time()
                candidate_frame = build_candidate_frame(features, cfg)
                v1_artifacts = rolling_predict(candidate_frame, cfg)
                v1_artifacts.predictions.to_parquet(cache_path, index=False)
                pd.DataFrame(v1_artifacts.quarter_metrics).to_csv(metrics_path, index=False)
                pd.DataFrame(v1_artifacts.boundary_checks).to_csv(boundaries_path, index=False)
                (v1_artifacts.calibration if v1_artifacts.calibration is not None else pd.DataFrame()).to_csv(
                    calibration_path, index=False
                )
                print(
                    f"[v1] rolling training/prediction done in {time.time()-t0:.1f}s; "
                    f"{len(v1_artifacts.predictions)} predictions"
                )
            features = apply_v1_signals(features, v1_artifacts, gate=args.v1_gate)
            print(
                f"[v1] gated candidate signals={int(features.arr('entry_signal').sum())} "
                f"across {int(features.arr('entry_signal').any(axis=1).sum())} decision days"
            )

        if args.signal_lag:
            features = LaggedFeatureStore(features, args.signal_lag)
            print(f"[stress] signal/exit features lagged by {args.signal_lag} trading day(s)")

        ledger = ExecutionLedger(features, cfg)
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
            indicator_config={"show_indicator": bool(args.show_indicator)},
        )
        strategy = LowVolTrendStrategy(features, cfg, ledger, features.market, trade_exchange=exchange)

        t0 = time.time()
        portfolio_dict, indicator_dict = backtest(
            start_time=start,
            end_time=end,
            strategy=strategy,
            executor=executor,
            benchmark=benchmark,
            account=cfg.execution.initial_cash,
            exchange_kwargs={"exchange": exchange},
        )
        print(f"[backtest] done in {time.time()-t0:.1f}s")
        report, positions = portfolio_dict["1day"]
        indicator_df, indicator_obj = indicator_dict["1day"]

        nav = nav_series(report, cfg.execution.initial_cash)
        nav_metrics = compute_nav_metrics(nav)
        round_trips = ledger.round_trips_frame()
        open_positions = ledger.open_positions_frame(
            features.arr("raw_close")[-1],
            current_index=len(features.dates) - 1,
        )
        round_trip_stats = compute_round_trip_stats(round_trips)
        frequency = compute_frequency_stats(positions, cfg)
        constraints = build_constraints_frame(positions, cfg)
        quarters = quarterly_returns(nav)
        segments = segment_metrics(nav, round_trips, positions, cfg, cfg.backtest.segments)
        acceptance = acceptance_report(nav_metrics, frequency, round_trip_stats, quarters, cfg)

        net_ret = nav.pct_change().dropna()
        risk = risk_analysis(net_ret, freq="day", mode="product")
        indicator_stats = indicator_analysis(indicator_df, method="value_weighted")

        # Standard Qlib observability artifacts (same artifact names as
        # PortAnaRecord: report_normal / positions_normal / portfolio_analysis).
        signals = _signal_frame(features)
        artifacts = {
            "report_normal.pkl": report,
            "pred.pkl": signals,
            "positions_normal.pkl": positions,
            "portfolio_analysis.pkl": risk,
            "indicator_analysis.pkl": indicator_stats,
            "orders.pkl": ledger.orders_frame(),
            "fills.pkl": ledger.fills_frame(),
            "round_trips.pkl": round_trips,
            "open_positions.pkl": open_positions,
            "daily_decisions.pkl": pd.DataFrame(strategy.decision_log),
            "constraints_daily.pkl": constraints,
            "market_daily.pkl": pd.DataFrame(
                {
                    "proxy": features.market.proxy,
                    "proxy_ma60": features.market.proxy_ma,
                    "breadth_60": features.market.breadth_60,
                    "breadth_20": features.market.breadth_20,
                    "drop_diffusion": features.market.drop_diffusion,
                    "raw_state": features.market.raw_state,
                    "effective_state": features.market.effective_state,
                    "effective_cap": features.market.effective_cap,
                    "entry_allowed": features.market.entry_allowed,
                },
                index=features.dates,
            ),
            "acceptance.pkl": {
                "nav_metrics": nav_metrics.__dict__,
                "round_trip_stats": round_trip_stats.__dict__,
                "frequency": frequency.__dict__,
                "checks": acceptance.checks,
                "passed": acceptance.passed,
            },
        }
        if v1_artifacts is not None:
            artifacts["v1_predictions.pkl"] = v1_artifacts.predictions
            artifacts["v1_quarter_metrics.pkl"] = pd.DataFrame(v1_artifacts.quarter_metrics)
            artifacts["v1_boundary_checks.pkl"] = pd.DataFrame(v1_artifacts.boundary_checks)
            if v1_artifacts.calibration is not None:
                artifacts["v1_calibration.pkl"] = v1_artifacts.calibration
        R.save_objects(**artifacts)
        R.log_metrics(
            cagr=nav_metrics.cagr,
            annual_vol=nav_metrics.annual_vol,
            max_drawdown=nav_metrics.max_drawdown,
            total_return=nav_metrics.total_return,
            round_trip_win_rate=round_trip_stats.win_rate,
            round_trips=round_trip_stats.n,
            rolling_252_min=frequency.rolling_252_min,
            rolling_63_min=frequency.rolling_63_min,
            acceptance_passed=int(acceptance.passed),
        )
        R.log_params(
            start=start,
            end=end,
            signal=signal_tag,
            initial_cash=cfg.execution.initial_cash,
            max_positions=cfg.strategy.max_positions,
            vol_target=cfg.strategy.vol_target,
            buy_premium=cfg.execution.buy_premium,
            participation_rate=cfg.execution.participation_rate,
            conservative_st=int(cfg.execution.unknown_st_conservative),
            use_market_timing=int(cfg.strategy.use_market_timing),
            use_vol_target=int(cfg.strategy.use_vol_target),
            use_drawdown_control=int(cfg.strategy.use_drawdown_control),
        )
        recorder_id = recorder.id

    # Local reports / audit tables.
    report.to_csv(out / "report_normal.csv")
    signals.to_csv(out / "signals.csv", index=False)
    nav.to_csv(out / "nav.csv")
    ledger.orders_frame().to_csv(out / "orders.csv", index=False)
    ledger.fills_frame().to_csv(out / "fills.csv", index=False)
    round_trips.to_csv(out / "round_trips.csv", index=False)
    open_positions.to_csv(out / "open_positions.csv", index=False)
    pd.DataFrame(strategy.decision_log).to_csv(out / "daily_decisions.csv", index=False)
    constraints.to_csv(out / "constraints_daily.csv")
    quarters.to_csv(out / "quarterly_returns.csv", index=False)
    pd.DataFrame(segments).T.to_csv(out / "segments.csv")
    pd.DataFrame(acceptance.checks).to_csv(out / "acceptance_checks.csv", index=False)
    pd.DataFrame(
        {
            "proxy": features.market.proxy,
            "proxy_ma60": features.market.proxy_ma,
            "proxy_ma20": features.market.proxy_short_ma,
            "breadth_60": features.market.breadth_60,
            "breadth_20": features.market.breadth_20,
            "drop_diffusion": features.market.drop_diffusion,
            "raw_state": features.market.raw_state,
            "effective_state": features.market.effective_state,
            "effective_cap": features.market.effective_cap,
            "entry_allowed": features.market.entry_allowed,
        },
        index=features.dates,
    ).to_csv(out / "market_daily.csv")
    if v1_artifacts is not None:
        v1_artifacts.predictions.to_parquet(out / "v1_predictions.parquet", index=False)
        pd.DataFrame(v1_artifacts.quarter_metrics).to_csv(out / "v1_quarter_metrics.csv", index=False)
        pd.DataFrame(v1_artifacts.boundary_checks).to_csv(out / "v1_boundary_checks.csv", index=False)
        if v1_artifacts.calibration is not None:
            v1_artifacts.calibration.to_csv(out / "v1_calibration.csv", index=False)
    summary = {
        "run": {
            "experiment": args.experiment,
            "recorder": recorder_name,
            "recorder_id": recorder_id,
            "signal": signal_tag,
            "start": start,
            "end": end,
            "initial_cash": cfg.execution.initial_cash,
            "conservative_st": cfg.execution.unknown_st_conservative,
            "baseline": bool(args.baseline),
            "signal_lag": int(args.signal_lag),
        },
        "data": {
            "db_path": cfg.data.db_path,
            "memory_limit": panel.source_info.get("audit", {}).get("memory_limit"),
            "n_dates": panel.n_dates,
            "n_symbols": panel.n_symbols,
        },
        "nav": nav_metrics.__dict__,
        "round_trip": round_trip_stats.__dict__,
        "frequency": frequency.__dict__,
        "acceptance": {"passed": acceptance.passed, "checks": acceptance.checks},
        "counts": {
            "orders": len(ledger.orders),
            "fills": len(ledger.fills),
            "round_trips": len(ledger.round_trips),
        },
        "segments": segments,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(summary), fh, ensure_ascii=False, indent=2, default=str)
    print(risk.to_string())
    print("fills", len(ledger.fills), "round_trips", len(ledger.round_trips), "orders", len(ledger.orders))
    print("acceptance passed:", acceptance.passed)
    for check in acceptance.checks:
        print(f"  {check['name']}: {check['passed']} value={check['value']} ({check['threshold']})")
    print("output ->", out)
    print("recorder ->", recorder_id)


if __name__ == "__main__":
    main()
