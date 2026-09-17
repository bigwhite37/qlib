#!/usr/bin/env python3
"""Run the V2 strategy through Qlib's account engine.

Inputs: cached V2 predictions (rolling quarterly models) plus the frozen exit
policy.  Outputs: the same observable artifacts as the V0 runner (Qlib recorder,
NAV, orders, fills, round trips, constraints, market state) plus the V2
opportunity-coverage table required by the design.
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
from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402
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
from lowvol_trend.qlib_backtest import LowVolExchange  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from lowvol_trend.v2_strategy import (  # noqa: E402
    V2MarketAdapter,
    V2Strategy,
    add_composite_score,
    apply_v2_signals,
    policy_from_config,
)

from qlib.backtest import backtest  # noqa: E402
from qlib.backtest.executor import SimulatorExecutor  # noqa: E402
from qlib.contrib.evaluate import indicator_analysis, risk_analysis  # noqa: E402
from qlib.workflow import R  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--predictions", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5_predictions.parquet")
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--output", default=str(ROOT / "output" / "v2_run"))
    parser.add_argument("--experiment", default="sleeve_a_v2")
    parser.add_argument("--recorder", default=None)
    parser.add_argument("--mu-min", type=float, default=-1.0)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--rank-by", default="pred_rel")
    parser.add_argument("--rank-blend", type=float, default=0.0, help="weight on the model rank (rest = rule composite)")
    parser.add_argument("--composite", default="combo3", help="rule composite spec used for the blend")
    parser.add_argument("--predictions2", default="", help="second prediction file for the rank ensemble")
    parser.add_argument("--blend2", type=float, default=0.0, help="weight of the second model rank")
    parser.add_argument("--blend-mode", default="mean", choices=["mean", "min"],
                        help="combine the model and rule ranks by average or by hard consensus")
    parser.add_argument("--exclude-boards", default="",
                        help="comma list of boards to drop from the base universe (main,chinext,star,bse)")
    parser.add_argument("--entry-quality-pct", type=float, default=0.0,
                        help="only enter on days whose best candidate beats this trailing percentile")
    parser.add_argument("--entry-quality-window", type=int, default=252)
    parser.add_argument("--entry-min-score", type=float, default=None,
                        help="skip entries whose blended score is below this level (patience)")
    parser.add_argument("--composite-only", type=int, default=0,
                        help="rank by the rule composite alone (makes --rank-blend 0 mean the composite)")
    parser.add_argument("--weak-cap", type=float, default=None, help="override the weak-market gross cap")
    parser.add_argument(
        "--state-caps",
        default="",
        help="override the market ladder's caps per state, e.g. weak:0.5,neutral:0.6,strong:0.65",
    )
    parser.add_argument("--min-amount-rank", type=float, default=None, help="V2 base-universe liquidity floor")
    parser.add_argument("--vol-window", type=int, default=None, help="portfolio volatility estimation window")
    parser.add_argument("--buy-premium", type=float, default=None, help="frozen maximum buy price as a fraction above the decision close")
    parser.add_argument("--cap-scale", type=float, default=None,
                        help="multiply every market-ladder gross cap by this factor")
    parser.add_argument("--risk-parity", type=int, default=0,
                        help="re-weight held positions toward inverse-vol at the current gross")
    parser.add_argument("--risk-parity-strength", type=float, default=None)
    parser.add_argument("--exit-rank", type=int, default=0,
                        help="exit a holding once its blended score falls out of the top quantile")
    parser.add_argument("--rank-exit-pct", type=float, default=None,
                        help="percentile below which a holding is considered out of favour")
    parser.add_argument("--rank-exit-confirm", type=int, default=None,
                        help="consecutive out-of-favour days before the rank exit fires")
    parser.add_argument("--rank-exit-min-hold", type=int, default=None)
    parser.add_argument("--scale-out", type=float, default=None,
                        help="fraction of the position to sell at the first profit target")
    parser.add_argument("--profit-target-2", type=float, default=None,
                        help="second profit target for the remainder after a scale-out")
    parser.add_argument("--profit-trigger-high", type=int, default=0,
                        help="trigger the profit target on an intraday touch of the high")
    parser.add_argument("--dd-peak-window", type=int, default=None,
                        help="measure the drawdown against the rolling N-day NAV high (0 = all-time peak)")
    parser.add_argument("--profit-atr-mult", type=float, default=None,
                        help="ATR-scaled profit target, in multiples of ATR at entry")
    parser.add_argument("--profit-atr-floor", type=float, default=None)
    parser.add_argument("--profit-atr-cap", type=float, default=None)
    parser.add_argument("--head-rerank", type=int, default=0, help="re-rank the top K candidates by --head-key")
    parser.add_argument("--head-key", default="vol60", help="feature used for the head re-rank")
    parser.add_argument("--head-sign", type=float, default=-1.0, help="-1 prefers low values of --head-key, +1 high")
    parser.add_argument("--corr-sizing", type=int, default=0, help="scale candidate size by its correlation with the book")
    parser.add_argument("--corr-sizing-strength", type=float, default=1.0)
    parser.add_argument("--corr-sizing-base", type=float, default=0.30)
    parser.add_argument("--corr-sizing-floor", type=float, default=0.25)
    parser.add_argument("--market-vol-brake", type=float, default=0.0,
                        help="cap gross exposure at this market-volatility budget (0 disables)")
    parser.add_argument("--composite-weights", default="", help="comma-separated weights for the composite components")
    parser.add_argument("--stale-days", type=int, default=0, help="exit a position still below entry after N days")
    parser.add_argument("--profit-lock", type=float, default=0.0, help="give back at most this fraction from the running max")
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--risk-trim-mode", default="scale", choices=["scale", "close_weakest"])
    parser.add_argument("--max-weight", type=float, default=None, help="single-name weight cap (fraction of NAV)")
    parser.add_argument("--risk-per-trade", type=float, default=None, help="per-trade stop risk budget (fraction of NAV)")
    parser.add_argument("--profit-target", type=float, default=0.05)
    parser.add_argument("--use-stop", type=int, default=0)
    parser.add_argument("--hold", type=int, default=20)
    parser.add_argument("--trailing", type=int, default=0)
    parser.add_argument("--extreme-exit", type=int, default=1, help="policy exit on the market extreme state")
    parser.add_argument("--extreme-liquidates", type=int, default=1, help="market ladder forces a full liquidation at 0% cap")
    parser.add_argument("--drawdown-tiers", default="", help="e.g. 0.06:0.6,0.09:0.35,0.12:0.15")
    parser.add_argument("--friction-bps", type=float, default=None, help="per-side friction cost in bp")
    parser.add_argument("--signal-lag", type=int, default=0, help="stress: shift signal/exit features by N days")
    parser.add_argument("--conservative-st", type=int, default=0, help="apply the 5% limit to every main-board name")
    parser.add_argument("--keep-all-features", action="store_true", help="skip the memory slimming step")
    parser.add_argument("--corr-filter", type=int, default=0, help="reject candidates correlated with current holdings")
    parser.add_argument("--corr-threshold", type=float, default=0.80)
    parser.add_argument("--corr-replace", type=int, default=0, help="replace a correlated holding with a higher-scoring candidate")
    parser.add_argument("--stop-atr-mult", type=float, default=None)
    parser.add_argument("--stop-min", type=float, default=None)
    parser.add_argument("--stop-max", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=12)
    parser.add_argument("--max-new", type=int, default=3)
    parser.add_argument("--gross", type=float, default=0.95)
    parser.add_argument("--use-market-timing", type=int, default=1)
    parser.add_argument("--use-vol-target", type=int, default=1)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--no-predictions", type=int, default=0, help="baseline: use the rule entry signal")
    return parser.parse_args()


def _mem(tag: str) -> None:
    """Print resident memory so the 3 GB budget can be verified stage by stage."""

    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1024**3
    except Exception:
        return
    print(f"[memory] {tag}: {rss:.2f} GiB", flush=True)


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def coverage_frame(features, base, pred_daily, cfg, ledger) -> pd.DataFrame:
    """Opportunity coverage table (design section 7)."""

    dates = features.dates
    base_counts = base.sum(axis=1)
    entry_signal = features.arr("entry_signal")
    frame = pd.DataFrame(
        {
            "date": dates,
            "base_candidates": base_counts.astype(int),
            "signal_candidates": entry_signal.sum(axis=1).astype(int),
        }
    )
    if pred_daily is not None and not pred_daily.empty:
        frame = frame.merge(pred_daily, on="date", how="left")
    orders = ledger.orders_frame()
    if not orders.empty:
        order_dates = pd.to_datetime(orders["decision_date"])
        planned = orders.assign(_d=order_dates).groupby("_d").size()
        planned.index = pd.DatetimeIndex(planned.index)
        frame = frame.merge(planned.rename("planned_orders"), left_on="date", right_index=True, how="left")
    fills = ledger.fills_frame()
    if not fills.empty:
        fill_dates = pd.to_datetime(fills["date"])
        filled = fills.assign(_d=fill_dates).groupby("_d").size()
        filled.index = pd.DatetimeIndex(filled.index)
        frame = frame.merge(filled.rename("fills"), left_on="date", right_index=True, how="left")
    for col in ("n_pred", "n_gate_win", "planned_orders", "fills"):
        if col not in frame:
            frame[col] = 0
        frame[col] = frame[col].fillna(0).astype(int)
    frame["fill_rate"] = np.where(frame["planned_orders"] > 0, frame["fills"] / frame["planned_orders"], np.nan)
    return frame


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    cfg.strategy.use_corr_filter = bool(args.corr_filter)
    cfg.strategy.corr_sizing = bool(args.corr_sizing)
    cfg.strategy.corr_sizing_strength = float(args.corr_sizing_strength)
    cfg.strategy.corr_sizing_base = float(args.corr_sizing_base)
    cfg.strategy.corr_sizing_floor = float(args.corr_sizing_floor)
    cfg.strategy.head_rerank_k = int(args.head_rerank)
    cfg.strategy.head_rerank_key = str(args.head_key)
    cfg.strategy.head_rerank_sign = float(args.head_sign)
    cfg.strategy.profit_trigger_on_high = bool(args.profit_trigger_high)
    if args.scale_out is not None:
        cfg.strategy.scale_out_fraction = float(args.scale_out)
    if args.profit_target_2 is not None:
        cfg.strategy.profit_target_pct_2 = float(args.profit_target_2)
    # The rank exit has never been active in the V2 live path (it existed only in
    # the older portfolio.evaluate_exits), and the config default is True, so the
    # flag must be assigned explicitly to keep the historical behaviour as default.
    cfg.strategy.exit_rank = bool(args.exit_rank)
    if args.entry_min_score is not None:
        cfg.strategy.entry_min_score = float(args.entry_min_score)
    cfg.strategy.rebalance_inverse_vol = bool(args.risk_parity)
    if args.risk_parity_strength is not None:
        cfg.strategy.rebalance_strength = float(args.risk_parity_strength)
    if args.rank_exit_pct is not None:
        cfg.strategy.rank_exit_below_quantile = float(args.rank_exit_pct)
    if args.rank_exit_confirm is not None:
        cfg.strategy.rank_exit_confirm_days = int(args.rank_exit_confirm)
    if args.rank_exit_min_hold is not None:
        cfg.strategy.rank_exit_min_hold = int(args.rank_exit_min_hold)
    if args.profit_atr_mult is not None:
        cfg.strategy.profit_target_atr_mult = float(args.profit_atr_mult)
    if args.profit_atr_floor is not None:
        cfg.strategy.profit_target_atr_floor = float(args.profit_atr_floor)
    if args.profit_atr_cap is not None:
        cfg.strategy.profit_target_atr_cap = float(args.profit_atr_cap)
    if args.dd_peak_window is not None:
        cfg.strategy.drawdown_peak_window = int(args.dd_peak_window)
    cfg.strategy.corr_threshold = float(args.corr_threshold)
    cfg.strategy.correlation_replace = bool(args.corr_replace)
    cfg.strategy.use_market_timing = bool(args.use_market_timing)
    cfg.strategy.use_vol_target = bool(args.use_vol_target)
    cfg.strategy.use_drawdown_control = bool(args.drawdown_tiers)
    if args.drawdown_tiers:
        tiers = []
        for item in args.drawdown_tiers.split(","):
            level, cap = item.split(":")
            tiers.append((float(level), float(cap)))
        cfg.strategy.drawdown_tiers = tuple(tiers)
    cfg.strategy.risk_trim_mode = args.risk_trim_mode
    cfg.strategy.extreme_liquidates = bool(args.extreme_liquidates)
    cfg.strategy.vol_target = float(args.vol_target)
    cfg.strategy.max_positions = int(args.max_positions)
    cfg.strategy.max_new_per_day = int(args.max_new)
    cfg.strategy.max_gross = float(args.gross)
    if args.initial_cash is not None:
        cfg.execution.initial_cash = float(args.initial_cash)
    if args.friction_bps is not None:
        cfg.execution.friction_bps_per_side = float(args.friction_bps)
    if args.min_amount_rank is not None:
        cfg.universe.v2_min_amount_rank = float(args.min_amount_rank)
    if args.vol_window is not None:
        cfg.strategy.vol_estimate_window = int(args.vol_window)
    if args.buy_premium is not None:
        cfg.execution.buy_premium = float(args.buy_premium)
    cfg.execution.unknown_st_conservative = bool(args.conservative_st)
    if args.max_weight is not None:
        cfg.strategy.max_weight = float(args.max_weight)
    if args.risk_per_trade is not None:
        cfg.strategy.risk_per_trade = float(args.risk_per_trade)
    # The design's per-trade risk cap (0.4% of NAV per stop) stays in force: with
    # an 8% stop it sizes positions at 5% of NAV, below the 8% single-name cap.
    print("[qlib]", init_local_qlib(cfg))
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    micro_specs = ("combo3p", "combo4", "combo4l", "combo5l")
    features = build_v2_features(panel, cfg, micro=args.composite in micro_specs)
    print("[features] lean V2 builder used", flush=True)
    _mem("features built")
    base = v2_base_mask(features, cfg)
    if args.exclude_boards:
        # Board composition is a universe design choice that has never been varied:
        # the strategy currently ranks across the main board, ChiNext, STAR and the
        # Beijing exchange together, and they have different limit rules and
        # investor bases.
        from lowvol_trend.data import infer_board as _infer_board

        excluded = {b.strip() for b in str(args.exclude_boards).split(",") if b.strip()}
        board_codes = np.array([_infer_board(s) for s in features.symbols], dtype=object)
        keep_columns = np.array([b not in excluded for b in board_codes], dtype=bool)
        base = base & keep_columns[None, :]
        print(
            "[universe] excluded boards %s -> %d of %d symbols remain"
            % (sorted(excluded), int(keep_columns.sum()), len(keep_columns)),
            flush=True,
        )
        del board_codes, keep_columns
    comp_weights = [float(x) for x in args.composite_weights.split(",")] if args.composite_weights else None
    add_composite_score(features, base, spec=args.composite, weights=comp_weights)
    market_state = compute_v2_market_state(features, base, cfg)
    if not args.keep_all_features:
        # Everything the market state, the base mask and the composite needed has
        # been computed by now; release the rest before the prediction matrix.
        from lowvol_trend.v2_strategy import slim_for_v2

        slim_for_v2(features)
        _mem("features slimmed")
    if args.state_caps:
        # Research switch: replace the ladder's per-state gross caps.  The state
        # machine itself is untouched, so this isolates the value of the cap
        # *levels* (and of the whole ladder) from the value of the state signal.
        state_overrides = {}
        for item in args.state_caps.split(","):
            name, sep, value = item.partition(":")
            if not sep:
                raise SystemExit("--state-caps expects e.g. weak:0.5,neutral:0.6,strong:0.65")
            state_overrides[name.strip()] = np.float32(float(value))
        for state_name, state_cap in state_overrides.items():
            state_mask = market_state.effective_state == state_name
            market_state.effective_cap = np.where(
                state_mask, state_cap, market_state.effective_cap
            ).astype(np.float32)
        print(
            "[ladder] state caps %s -> mean cap %.3f"
            % (
                {k: float(v) for k, v in state_overrides.items()},
                float(np.mean(market_state.effective_cap)),
            ),
            flush=True,
        )
    if args.weak_cap is not None:
        # Research switch: the design's ladder gives weak markets a 35% cap.  A
        # reversal/illiquidity strategy may prefer a different level there, so the
        # weak and post-extreme-recovery caps can be overridden without touching
        # the state machine itself.
        mask = (market_state.effective_state == "weak") | (market_state.recovery_phase > 0)
        market_state.effective_cap = np.where(
            mask, np.float32(args.weak_cap), market_state.effective_cap
        ).astype(np.float32)
        print("[ladder] weak/recovery cap overridden to " + str(args.weak_cap), flush=True)
    if args.cap_scale is not None:
        market_state.effective_cap = (market_state.effective_cap * np.float32(args.cap_scale)).astype(np.float32)
        print(
            "[ladder] every gross cap scaled by %.2f (mean cap %.3f)"
            % (args.cap_scale, float(np.mean(market_state.effective_cap))),
            flush=True,
        )
    if args.market_vol_brake > 0:
        # Market-level volatility brake: scale the gross cap by the ratio of the
        # budget to the equal-weighted market proxy's 20-day realised volatility.
        proxy = pd.Series(market_state.proxy.astype(np.float64), index=panel.dates)
        mkt_vol = proxy.pct_change().rolling(20, min_periods=20).std() * np.sqrt(252.0)
        ratio = (float(args.market_vol_brake) / mkt_vol).clip(upper=1.0).fillna(1.0).to_numpy()
        market_state.effective_cap = (market_state.effective_cap.astype(np.float64) * ratio).astype(np.float32)
        print(
            "[brake] market vol budget %.3f -> mean cap multiplier %.3f (min %.3f)"
            % (args.market_vol_brake, float(np.mean(ratio)), float(np.min(ratio))),
            flush=True,
        )
    market = V2MarketAdapter(market_state)
    proxy_ret = market_state.proxy_return.astype(np.float64)
    market.median_return = proxy_ret
    benchmark = pd.Series(proxy_ret, index=features.dates, name="v2_market_proxy")

    predictions = None
    pred_daily = None
    diag = {"n_signals": int(features.arr("entry_signal").sum())}
    if not args.no_predictions:
        predictions = pd.read_parquet(
            args.predictions,
            columns=["datetime", "symbol_index", "pred_rel", "pred_win", "pred_return"],
        )
        extra_rank = None
        if args.predictions2 and args.blend2 > 0:
            from lowvol_trend.features import row_pct_rank as _row_rank
            from lowvol_trend.v2_strategy import prediction_matrices as _pred_mats

            second = pd.read_parquet(
                args.predictions2, columns=["datetime", "symbol_index", "pred_rel", "pred_win", "pred_return"]
            )
            rel2, _ = _pred_mats(features, second)
            del second
            extra_rank = _row_rank(rel2, base)
            del rel2
            import gc as _gc

            _gc.collect()
            print(f"[ensemble] second ranker loaded, weight {args.blend2}", flush=True)
        features, diag = apply_v2_signals(
            features, predictions, args.mu_min, args.p_min, base,
            rank_by=args.rank_by, blend=args.rank_blend,
            extra_rank=extra_rank, extra_weight=float(args.blend2),
            force_composite=bool(args.composite_only),
            blend_mode=str(args.blend_mode),
        )
        if float(getattr(args, "entry_quality_pct", 0.0) or 0.0) > 0.0:
            # Time-series patience gate: only open a slot on days whose BEST
            # candidate is above the given percentile of the trailing distribution
            # of daily best candidates.  An absolute threshold cannot do this - the
            # maximum of a percentile-rank average over ~3000 names is always high,
            # which is why --entry-min-score turned out to be a silent no-op.
            window = int(getattr(args, "entry_quality_window", 252) or 252)
            score_matrix = features.arr("entry_score")
            with np.errstate(all="ignore"):
                day_max = np.nanmax(np.where(np.isfinite(score_matrix), score_matrix, np.nan), axis=1)
            day_max = np.nan_to_num(day_max, nan=0.0)
            series = pd.Series(day_max)
            rolling_threshold = series.rolling(window, min_periods=60).quantile(
                float(args.entry_quality_pct)
            )
            threshold_values = rolling_threshold.to_numpy()
            open_day = (series.to_numpy() >= threshold_values) | ~np.isfinite(threshold_values)
            features.arrays["entry_day_ok"] = open_day.astype(bool)
            print(
                "[patience] day gate open on %.1f%% of days (pct=%.2f, window=%d)"
                % (100.0 * float(open_day.mean()), float(args.entry_quality_pct), window),
                flush=True,
            )
            del score_matrix, day_max, series, rolling_threshold, threshold_values

        # Keep only the small per-day aggregate; the row-level table is released
        # before the backtest so the run stays inside the memory budget.
        pred_daily = (
            predictions.groupby("datetime")
            .agg(n_pred=("pred_rel", "size"), n_gate_win=("pred_win", lambda s: int((s >= args.p_min).sum())))
            .reset_index()
            .rename(columns={"datetime": "date"})
        )
    del predictions
    import gc

    gc.collect()
    _mem("signals applied")
    print("[signals]", diag, flush=True)

    policy_kwargs = dict(
        use_stop=bool(args.use_stop),
        # a non-positive target disables profit-taking entirely
        profit_target_pct=(float(args.profit_target) if args.profit_target > 0 else None),
        use_trailing=bool(args.trailing),
        max_hold_days=int(args.hold),
        exit_on_market_extreme=bool(args.extreme_exit),
        exit_below_entry_days=int(args.stale_days),
        profit_lock_pct=float(args.profit_lock),
    )
    if args.stop_atr_mult is not None:
        policy_kwargs["stop_atr_mult"] = float(args.stop_atr_mult)
    if args.stop_min is not None:
        policy_kwargs["stop_min"] = float(args.stop_min)
    if args.stop_max is not None:
        policy_kwargs["stop_max"] = float(args.stop_max)
    if args.signal_lag:
        from lowvol_trend.features import LaggedFeatureStore

        features = LaggedFeatureStore(features, args.signal_lag)
        print(f"[stress] signal/exit features lagged by {args.signal_lag} trading day(s)", flush=True)
    policy = policy_from_config(cfg, **policy_kwargs)
    # Freeze the ledger's per-position stop to the exit policy so the live risk
    # checks and the labels share one definition.
    cfg.strategy.stop_atr_mult = policy.stop_atr_mult
    cfg.strategy.stop_min = policy.stop_min
    cfg.strategy.stop_max = policy.stop_max
    print("[policy]", policy.describe(), flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    recorder_name = args.recorder or f"v2_{int(time.time())}"
    open_cost = (
        cfg.execution.commission_rate + cfg.execution.transfer_fee_rate + cfg.execution.friction_bps_per_side / 10_000.0
    )
    close_cost = (
        0.001
        + cfg.execution.commission_rate
        + cfg.execution.transfer_fee_rate
        + cfg.execution.friction_bps_per_side / 10_000.0
    )
    with R.start(experiment_name=args.experiment, recorder_name=recorder_name):
        recorder = R.get_recorder()
        ledger = ExecutionLedger(features, cfg)
        exchange = LowVolExchange(
            features,
            cfg,
            freq="day",
            start_time=args.start,
            end_time=args.end,
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
        strategy = V2Strategy(
            features, cfg, ledger, market, policy=policy, trade_exchange=exchange
        )
        t0 = time.time()
        portfolio_dict, indicator_dict = backtest(
            start_time=args.start,
            end_time=args.end,
            strategy=strategy,
            executor=executor,
            benchmark=benchmark,
            account=cfg.execution.initial_cash,
            exchange_kwargs={"exchange": exchange},
        )
        print(f"[backtest] done in {time.time()-t0:.1f}s", flush=True)
        _mem("backtest finished")
        report, positions = portfolio_dict["1day"]
        indicator_df, indicator_obj = indicator_dict["1day"]
        nav = nav_series(report, cfg.execution.initial_cash)
        nav_metrics = compute_nav_metrics(nav)
        round_trips = ledger.round_trips_frame()
        open_positions = ledger.open_positions_frame(
            features.arr("raw_close")[-1], current_index=len(features.dates) - 1
        )
        round_trip_stats = compute_round_trip_stats(round_trips)
        frequency = compute_frequency_stats(positions, cfg)
        constraints = build_constraints_frame(positions, cfg)
        quarters = quarterly_returns(nav)
        segments = segment_metrics(nav, round_trips, positions, cfg, cfg.backtest.segments)
        acceptance = acceptance_report(nav_metrics, frequency, round_trip_stats, quarters, cfg)
        coverage = coverage_frame(features, base, pred_daily, cfg, ledger)
        _mem("stats computed")
        net_ret = nav.pct_change().dropna()
        risk = risk_analysis(net_ret, freq="day", mode="product")
        indicator_stats = indicator_analysis(indicator_df, method="value_weighted")
        market_daily = pd.DataFrame(
            {
                "proxy": market_state.proxy,
                "breadth_20": market_state.breadth_20,
                "breadth_60": market_state.breadth_60,
                "drop_diffusion": market_state.drop_diffusion,
                "raw_state": market_state.raw_state,
                "effective_state": market_state.effective_state,
                "effective_cap": market_state.effective_cap,
                "recovery_phase": market_state.recovery_phase,
            },
            index=features.dates,
        )
        artifacts = {
            "report_normal.pkl": report,
            "positions_normal.pkl": positions,
            "portfolio_analysis.pkl": risk,
            "indicator_analysis.pkl": indicator_stats,
            "orders.pkl": ledger.orders_frame(),
            "fills.pkl": ledger.fills_frame(),
            "round_trips.pkl": round_trips,
            "open_positions.pkl": open_positions,
            "daily_decisions.pkl": pd.DataFrame(strategy.decision_log),
            "constraints_daily.pkl": constraints,
            "coverage_daily.pkl": coverage,
            "market_daily.pkl": market_daily,
            "exit_log.pkl": pd.DataFrame(strategy.exit_log),
            "acceptance.pkl": {
                "nav_metrics": nav_metrics.__dict__,
                "round_trip_stats": round_trip_stats.__dict__,
                "frequency": frequency.__dict__,
                "checks": acceptance.checks,
                "passed": acceptance.passed,
            },
        }
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
            start=args.start,
            end=args.end,
            profit_target=args.profit_target,
            use_stop=int(args.use_stop),
            hold=int(args.hold),
            mu_min=args.mu_min,
            p_min=args.p_min,
            max_positions=int(args.max_positions),
            use_market_timing=int(args.use_market_timing),
            use_vol_target=int(args.use_vol_target),
            vol_target=float(args.vol_target),
        )
        recorder_id = recorder.id

    nav.to_csv(out / "nav.csv")
    report.to_csv(out / "report_normal.csv")
    ledger.orders_frame().to_csv(out / "orders.csv", index=False)
    ledger.fills_frame().to_csv(out / "fills.csv", index=False)
    round_trips.to_csv(out / "round_trips.csv", index=False)
    open_positions.to_csv(out / "open_positions.csv", index=False)
    pd.DataFrame(strategy.decision_log).to_csv(out / "daily_decisions.csv", index=False)
    constraints.to_csv(out / "constraints_daily.csv")
    coverage.to_csv(out / "coverage_daily.csv", index=False)
    market_daily.to_csv(out / "market_daily.csv")
    quarters.to_csv(out / "quarterly_returns.csv", index=False)
    pd.DataFrame(segments).T.to_csv(out / "segments.csv")
    pd.DataFrame(acceptance.checks).to_csv(out / "acceptance_checks.csv", index=False)
    summary = {
        "run": vars(args),
        "nav": nav_metrics.__dict__,
        "round_trip": round_trip_stats.__dict__,
        "frequency": frequency.__dict__,
        "acceptance": {"passed": acceptance.passed, "checks": acceptance.checks},
        "counts": {"orders": len(ledger.orders), "fills": len(ledger.fills), "round_trips": len(ledger.round_trips)},
        "segments": segments,
        "signals": diag,
        "policy": policy.describe(),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(summary), fh, ensure_ascii=False, indent=2, default=str)
    print("acceptance passed:", acceptance.passed)
    for check in acceptance.checks:
        print(f"  {check['name']}: {check['passed']} value={check['value']} ({check['threshold']})")
    print("output ->", out)
    print("recorder ->", recorder_id)


if __name__ == "__main__":
    main()
