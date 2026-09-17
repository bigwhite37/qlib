"""Core unit tests for sleeve/a.

These tests do not touch the 1.2GB DuckDB; they exercise the pure rules, fee
schedule, acceptance metrics and the no-future-leak property with synthetic
panels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from lowvol_trend.config import StrategyConfig, load_config
from lowvol_trend.data import PanelData
from lowvol_trend.execution import board_limit_pct, compute_fees, effective_limit_pct, max_affordable_quantity
from lowvol_trend.features import build_features, row_pct_rank
from lowvol_trend.metrics import (
    compute_frequency_stats,
    compute_nav_metrics,
    compute_round_trip_stats,
    nav_series,
    quarterly_returns,
)
from lowvol_trend.portfolio import drawdown_cap


def _make_panel(n_dates: int = 120, n_symbols: int = 4, seed: int = 7) -> PanelData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    symbols = [f"SH60000{i}" for i in range(n_symbols)]
    base = 10.0 + np.arange(n_symbols) * 5.0
    close = np.zeros((n_dates, n_symbols), dtype=np.float32)
    for j in range(n_symbols):
        drift = 0.0005 * (j - 1.5)
        ret = rng.normal(drift, 0.012, size=n_dates)
        close[:, j] = (base[j] * np.cumprod(1.0 + ret)).astype(np.float32)
    high = (close * 1.02).astype(np.float32)
    low = (close * 0.98).astype(np.float32)
    open_ = (close * 0.999).astype(np.float32)
    factor = np.ones_like(close)
    volume = (np.abs(rng.normal(1_000_000, 100_000, size=close.shape)) + 100_000).astype(np.float32)
    amount = (close * volume / 1000.0).astype(np.float32)
    valid = np.ones_like(close, dtype=bool)
    # vwap is present in the real DuckDB and is now a feature source; the
    # synthetic panel mirrors that so the feature builder is exercised the same way.
    vwap = ((high + low + close) / 3.0).astype(np.float32)
    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
        "factor": factor,
        "vwap": vwap,
    }
    meta = pd.DataFrame(
        {
            "symbol": symbols,
            "start_date": [20150105] * n_symbols,
            "end_date": [20260914] * n_symbols,
            "board": ["sh_main"] * n_symbols,
        }
    ).set_index("symbol", drop=False)
    return PanelData(
        dates=dates,
        date_ints=np.asarray([int(d.strftime("%Y%m%d")) for d in dates], dtype=np.int64),
        trade_indices=np.arange(n_dates, dtype=np.int64),
        symbols=symbols,
        fields=fields,
        valid=valid,
        instrument_meta=meta,
        db_path="synthetic",
    )


def test_config_loads_and_rejects_unknown_keys(tmp_path):
    cfg = load_config()
    assert cfg.data.memory_limit == "2GB"
    assert cfg.strategy.max_positions == 12
    path = tmp_path / "bad.yaml"
    path.write_text("strategy:\n  no_such_parameter: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_drawdown_cap_tiers():
    cfg = load_config()
    tiers = cfg.strategy.drawdown_tiers
    assert drawdown_cap(100.0, 100.0, tiers) == 1.0
    assert drawdown_cap(96.0, 100.0, tiers) == 0.50
    assert drawdown_cap(94.0, 100.0, tiers) == 0.25
    assert drawdown_cap(80.0, 100.0, tiers) == 0.25


def test_fee_schedule_historical_stamp_tax_and_min_commission():
    cfg = load_config()
    buy = compute_fees("BUY", 1_000_000.0, 20200101, cfg)
    assert buy["stamp_tax"] == 0.0
    assert buy["commission"] == pytest.approx(250.0)
    assert buy["total_fee"] > 250.0  # transfer + friction
    min_buy = compute_fees("BUY", 100.0, 20200101, cfg)
    assert min_buy["commission"] == pytest.approx(5.0)
    sell_old = compute_fees("SELL", 1_000_000.0, 20230827, cfg)
    sell_new = compute_fees("SELL", 1_000_000.0, 20230828, cfg)
    assert sell_old["stamp_tax"] == pytest.approx(1000.0)
    assert sell_new["stamp_tax"] == pytest.approx(500.0)
    assert sell_old["total_fee"] > sell_new["total_fee"]


def test_board_limit_dates_and_conservative_st():
    cfg = load_config()
    assert board_limit_pct("sh_main", 20200101) == pytest.approx(0.10)
    assert board_limit_pct("chinext", 20200821) == pytest.approx(0.10)
    assert board_limit_pct("chinext", 20200824) == pytest.approx(0.20)
    assert board_limit_pct("star", 20200101) == pytest.approx(0.20)
    assert board_limit_pct("bse", 20220101) == pytest.approx(0.30)
    cfg.execution.unknown_st_conservative = True
    assert effective_limit_pct("sh_main", 20200101, cfg) == pytest.approx(0.05)
    assert effective_limit_pct("star", 20200101, cfg) == pytest.approx(0.20)


def test_max_affordable_quantity_rounds_lots_and_respects_cash():
    cfg = load_config()
    qty = max_affordable_quantity(10_000.0, 10.0, cfg)
    assert qty % cfg.execution.lot_size == 0
    assert qty >= 900
    assert qty <= 1000
    assert max_affordable_quantity(50.0, 10.0, cfg) == 0


def test_row_pct_rank_direction():
    values = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    mask = np.ones_like(values, dtype=bool)
    rank = row_pct_rank(values, mask)
    assert rank[0, 3] == pytest.approx(1.0)
    assert rank[0, 0] == pytest.approx(0.25)
    mask[0, 0] = False
    values[0, 0] = 100.0
    rank = row_pct_rank(values, mask)
    assert np.isnan(rank[0, 0])


def test_nav_metrics_and_quarterly_returns():
    index = pd.bdate_range("2020-01-01", periods=260)
    nav = pd.Series(np.linspace(100.0, 121.0, len(index)), index=index)
    report = pd.DataFrame({"account": nav})
    full_nav = nav_series(report, 100.0)
    metrics = compute_nav_metrics(full_nav)
    assert metrics.final_nav == pytest.approx(121.0)
    assert metrics.cagr > 0.15
    assert metrics.max_drawdown == pytest.approx(0.0)
    quarters = quarterly_returns(full_nav)
    assert set(quarters["quarter"].str[-1]) <= {"1", "2", "3", "4"}


def test_round_trip_win_rate():
    frame = pd.DataFrame(
        {
            "profit": [10.0, -5.0, 20.0, -1.0],
            "return_pct": [0.01, -0.005, 0.02, -0.001],
            "holding_days": [5, 7, 9, 2],
            "exit_reason": ["trailing_stop", "stop_loss", "time_exit", "rank_exit"],
        }
    )
    stats = compute_round_trip_stats(frame)
    assert stats.n == 4
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.profit_factor == pytest.approx(30.0 / 6.0)


def test_market_extreme_and_neutral_state():
    from lowvol_trend.market import compute_market_state

    cfg = load_config()
    n = 120
    ret = np.zeros((n, 1), dtype=np.float32)
    close = np.ones((n, 1), dtype=np.float32) * 10.0
    ma60 = np.ones((n, 1), dtype=np.float32) * 9.0
    ma20 = np.ones((n, 1), dtype=np.float32) * 9.0
    eligible = np.ones((n, 1), dtype=bool)
    state = compute_market_state(ret, close, ma60, ma20, eligible, cfg.market)
    assert state.raw_state[-1] in {"neutral", "weak", "strong"}
    ret[-1, 0] = -0.10
    state = compute_market_state(ret, close, ma60, ma20, eligible, cfg.market)
    assert state.raw_state[-1] == "extreme"
    assert state.effective_cap[-1] == pytest.approx(0.0)


def test_v1_label_horizon_is_t_plus_11():
    from lowvol_trend.v1 import build_candidate_frame

    cfg = load_config()
    cfg.universe.min_history_days = 20
    cfg.universe.recent_window = 20
    cfg.universe.min_recent_bars = 5
    panel = _make_panel(n_dates=120, n_symbols=2)
    features = build_features(panel, cfg)
    # Force every valid row into the candidate set so the label construction is
    # observable; the V1 label must be close[T+11]/close[T+1]-1.
    signal = np.zeros_like(features.arr("entry_signal"), dtype=bool)
    signal[: 120 - 12, :] = True
    features.arrays["entry_signal"] = signal
    features.arrays["entry_score"] = np.where(signal, 1.0, np.nan).astype(np.float32)
    frame = build_candidate_frame(features, cfg, include_alpha158=False)
    close = features.arr("close")
    sample = frame.iloc[0]
    datetime = sample.name[0]
    instrument = sample.name[1]
    t = features.date_to_index[pd.Timestamp(datetime).normalize()]
    j = features.symbol_to_index[instrument]
    expected = close[t + 11, j] / close[t + 1, j] - 1.0
    expected_binary = 1.0 if expected > 0.005 else 0.0
    assert sample[("label", "LABEL0")] == pytest.approx(expected_binary)
    assert pd.Timestamp(sample[("meta", "label_end_date")]) == features.dates[t + 11]


def test_alpha158_feature_family_runs():
    from lowvol_trend.alpha158_features import ALPHA158_WINDOWS, Alpha158Features, alpha158_feature_names

    cfg = load_config()
    cfg.universe.min_history_days = 20
    panel = _make_panel(n_dates=120, n_symbols=2)
    features = build_features(panel, cfg)
    t_idx = np.asarray([110], dtype=np.int64)
    j_idx = np.asarray([0], dtype=np.int64)
    computed = Alpha158Features(features, windows=(5,)).compute(t_idx, j_idx)
    assert len(computed) == 12 + 29
    for name in ("KMID", "KUP", "KMID2", "ROC5", "RSV5", "RANK5", "SUMP5"):
        assert name in computed
        assert np.isfinite(computed[name]).all()
    assert len(alpha158_feature_names(ALPHA158_WINDOWS)) == 12 + 29 * len(ALPHA158_WINDOWS)


def test_lagged_feature_store_keeps_execution_prices_current():
    from lowvol_trend.features import LaggedFeatureStore

    cfg = load_config()
    cfg.universe.min_history_days = 20
    panel = _make_panel(n_dates=80, n_symbols=2)
    features = build_features(panel, cfg)
    lagged = LaggedFeatureStore(features, lag=1)
    np.testing.assert_array_equal(lagged.arr("raw_close"), features.arr("raw_close"))
    np.testing.assert_allclose(lagged.arr("close")[1:], features.arr("close")[:-1], equal_nan=True)
    assert np.isnan(lagged.arr("entry_score")[0]).all()


@dataclass
class _StubPosition:
    position: Dict[str, Any]
    stock_value: float
    stocks: List[str]

    def calculate_stock_value(self) -> float:
        return self.stock_value

    def get_stock_list(self) -> List[str]:
        return self.stocks


def test_frequency_stats():
    cfg = load_config()
    dates = pd.bdate_range("2020-01-01", periods=300)
    positions = {}
    for i, dt in enumerate(dates):
        held = i % 2 == 0
        positions[dt] = _StubPosition(
            position={"cash": 10.0, "now_account_value": 100.0},
            stock_value=50.0 if held else 5.0,
            stocks=["SH600000", "SH600001", "SH600002"] if held else ["SH600000"],
        )
    stats = compute_frequency_stats(positions, cfg)
    assert stats.n_days == 300
    assert 0.0 < stats.full_effective_share < 1.0
    assert stats.rolling_63_min >= 0.0


def test_features_no_future_leak():
    """Rows up to T must be identical when only future prices are changed."""

    cfg = load_config()
    cfg.universe.min_history_days = 20
    cfg.universe.recent_window = 20
    cfg.universe.min_recent_bars = 5
    panel_a = _make_panel()
    panel_b = _make_panel()
    # Divergence starts at row 100; earlier rows are identical.
    panel_b.fields["close"] = panel_b.fields["close"].copy()
    panel_b.fields["close"][100:] *= 1.5
    panel_b.fields["high"] = panel_b.fields["high"].copy()
    panel_b.fields["high"][100:] *= 1.5
    panel_b.fields["low"] = panel_b.fields["low"].copy()
    panel_b.fields["low"][100:] *= 1.0
    f_a = build_features(panel_a, cfg)
    f_b = build_features(panel_b, cfg)
    for name in ("entry_signal", "base_pass", "rs_rank", "score"):
        a = f_a.arr(name)[:99]
        b = f_b.arr(name)[:99]
        if a.dtype == bool:
            np.testing.assert_array_equal(a, b, err_msg=name)
        else:
            np.testing.assert_allclose(a, b, equal_nan=True, err_msg=name)
