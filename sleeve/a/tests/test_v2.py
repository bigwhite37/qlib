"""Unit tests for the V2 policy simulator, gates and lean feature builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lowvol_trend.config import load_config
from lowvol_trend.features import build_features, row_pct_rank
from lowvol_trend.portfolio import select_new_candidates
from lowvol_trend.v2 import v2_base_mask
from lowvol_trend.v2_data import build_v2_features
from lowvol_trend.v2_policy import ExitPolicy, policy_stats, simulate_entries
from lowvol_trend.v2_strategy import add_composite_score, apply_gate

from test_core import _make_panel


def _cfg():
    return load_config(None)


def test_stop_fraction_clips_and_handles_nan():
    policy = ExitPolicy(stop_atr_mult=2.0, stop_min=0.04, stop_max=0.08)
    atr = np.array([0.0, 0.5, 2.0, np.nan], dtype=np.float64)
    close = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    out = policy.stop_fraction(atr, close)
    assert out[0] == pytest.approx(0.04)
    assert out[1] == pytest.approx(0.08)  # 2*0.5/10 = 0.10 -> clipped to max
    assert out[2] == pytest.approx(0.08)
    assert out[3] == pytest.approx(0.08)  # NaN falls back to the maximum stop


def test_lean_feature_builder_matches_full_builder():
    panel = _make_panel(n_dates=200, n_symbols=5)
    cfg = _cfg()
    full = build_features(panel, cfg)
    lean = build_v2_features(panel, cfg)
    for name in ("close", "raw_close", "ma20", "ma60", "atr20", "ret1", "vol60", "amount_rank", "adtv20_shares"):
        a = np.asarray(full.arr(name), dtype=np.float64)
        b = np.asarray(lean.arr(name), dtype=np.float64)
        np.testing.assert_allclose(
            np.nan_to_num(a, nan=-1.0), np.nan_to_num(b, nan=-1.0), rtol=1e-5, atol=1e-6, err_msg=name
        )


def test_policy_simulator_uses_next_day_close_and_price_cap():
    panel = _make_panel(n_dates=60, n_symbols=2, seed=3)
    cfg = _cfg()
    features = build_v2_features(panel, cfg)
    entry = np.zeros(panel.valid.shape, dtype=bool)
    entry[10, 0] = True
    policy = ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=None, max_hold_days=5,
                        exit_on_market_extreme=False)
    frame = simulate_entries(features, cfg, entry, policy)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert int(row["entry_index"]) == 11
    assert bool(row["entry_filled"])
    raw_entry = float(features.arr("raw_close")[11, 0])
    raw_prev = float(features.arr("raw_close")[10, 0])
    assert raw_entry <= raw_prev * (1.0 + cfg.execution.buy_premium) + 1e-9
    # exit signal at entry + max_hold_days, filled on the next tradable close
    assert int(row["exit_signal_index"]) == 16
    assert int(row["exit_fill_index"]) == 17
    assert row["exit_reason"] == "time_exit"


def test_policy_simulator_records_unfilled_entries():
    panel = _make_panel(n_dates=40, n_symbols=1, seed=11)
    cfg = _cfg()
    features = build_v2_features(panel, cfg)
    # force the T+1 close 20% above the decision close so the 3% cap rejects it
    raw = features.arr("raw_close").copy()
    features.arrays["raw_close"] = raw
    raw[11, 0] = raw[10, 0] * 1.2
    entry = np.zeros(panel.valid.shape, dtype=bool)
    entry[10, 0] = True
    frame = simulate_entries(features, cfg, entry, ExitPolicy(max_hold_days=5))
    assert len(frame) == 1
    assert not bool(frame.iloc[0]["entry_filled"])
    assert frame.iloc[0]["exit_reason"] == "not_filled"
    assert np.isnan(frame.iloc[0]["net_return"])
    stats = policy_stats(frame)
    assert stats["n"] == 0
    assert stats["n_attempt"] == 1


def test_profit_target_and_time_exit_reasons():
    panel = _make_panel(n_dates=60, n_symbols=1, seed=5)
    cfg = _cfg()
    features = build_v2_features(panel, cfg)
    close = features.arr("close")
    entry = np.zeros(panel.valid.shape, dtype=bool)
    entry[10, 0] = True
    # force a +10% jump two days after entry so the 5% target triggers
    close[13, 0] = close[11, 0] * 1.10
    policy = ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=0.05, max_hold_days=20,
                        exit_on_market_extreme=False)
    frame = simulate_entries(features, cfg, entry, policy)
    assert frame.iloc[0]["exit_reason"] == "profit_target"
    assert int(frame.iloc[0]["exit_signal_index"]) == 13
    assert int(frame.iloc[0]["exit_fill_index"]) == 14


def test_apply_gate_respects_both_thresholds():
    panel = _make_panel(n_dates=30, n_symbols=3, seed=2)
    cfg = _cfg()
    features = build_v2_features(panel, cfg)
    base = np.ones(panel.valid.shape, dtype=bool)
    rel = np.array([[0.02, 0.005, 0.03]], dtype=np.float32).repeat(30, axis=0)
    win = np.array([[0.60, 0.60, 0.40]], dtype=np.float32).repeat(30, axis=0)
    gated = apply_gate(features, base, rel, win, mu_min=0.01, p_min=0.50)
    signal = gated.arr("entry_signal")
    assert signal[0, 0] and not signal[0, 1] and not signal[0, 2]
    np.testing.assert_allclose(np.nan_to_num(gated.arr("entry_score"), nan=-9.0), np.where(signal, rel, -9.0))


def test_row_pct_rank_matches_pandas_reference():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(40, 25)).astype(np.float32)
    mask = np.ones_like(values, dtype=bool)
    mask[3, :] = False
    values[7, 2] = np.nan
    out = row_pct_rank(values, mask)
    reference = row_pct_rank(values, mask, block=1000)
    np.testing.assert_allclose(np.nan_to_num(out, nan=-1.0), np.nan_to_num(reference, nan=-1.0), atol=1e-6)

    import pandas as pd

    masked = np.where(mask & np.isfinite(values), values, np.nan)
    expected = pd.DataFrame(masked).rank(axis=1, pct=True, na_option="keep").to_numpy(dtype=np.float64)
    np.testing.assert_allclose(np.nan_to_num(out, nan=-1.0), np.nan_to_num(expected, nan=-1.0), atol=1e-6)


def test_composite_score_is_bounded_and_masked():
    panel = _make_panel(n_dates=200, n_symbols=6, seed=9)
    cfg = _cfg()
    features = build_v2_features(panel, cfg)
    base = np.ones(panel.valid.shape, dtype=bool)
    base[:, 5] = False
    add_composite_score(features, base)
    composite = features.arr("composite_rank")
    assert np.isnan(composite[:, 5]).all()
    finite = composite[np.isfinite(composite)]
    assert finite.size > 0
    assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_close_weakest_trim_mode_is_opt_in():
    cfg = _cfg()
    assert cfg.strategy.risk_trim_mode == "scale"
    cfg.strategy.risk_trim_mode = "close_weakest"
    assert cfg.strategy.risk_trim_mode == "close_weakest"
    assert cfg.strategy.min_trim_weight > 0


def test_row_pct_rank_matches_pandas_reference():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(40, 25)).astype(np.float32)
    mask = np.ones_like(values, dtype=bool)
    mask[3, :] = False
    values[7, 2] = np.nan
    out = row_pct_rank(values, mask)
    reference = row_pct_rank(values, mask, block=1000)
    np.testing.assert_allclose(np.nan_to_num(out, nan=-1.0), np.nan_to_num(reference, nan=-1.0), atol=1e-6)
    masked = np.where(mask & np.isfinite(values), values, np.nan)
    expected = pd.DataFrame(masked).rank(axis=1, pct=True, na_option="keep").to_numpy(dtype=np.float64)
    np.testing.assert_allclose(np.nan_to_num(out, nan=-1.0), np.nan_to_num(expected, nan=-1.0), atol=1e-6)


def test_composite_score_is_bounded_and_masked():
    panel = _make_panel(n_dates=200, n_symbols=6, seed=9)
    cfg = _cfg()
    features = build_v2_features(panel, cfg)
    base = np.ones(panel.valid.shape, dtype=bool)
    base[:, 5] = False
    add_composite_score(features, base)
    composite = features.arr("composite_rank")
    assert np.isnan(composite[:, 5]).all()
    finite = composite[np.isfinite(composite)]
    assert finite.size > 0
    assert finite.min() >= 0.0 and finite.max() <= 1.0


def test_risk_trim_mode_defaults_to_proportional_scaling():
    cfg = _cfg()
    assert cfg.strategy.risk_trim_mode == "scale"
    assert 0.0 < cfg.strategy.vol_rebalance_band <= 1.0
    assert 0.0 < cfg.strategy.gross_rebalance_band <= 1.0
    assert cfg.strategy.min_trim_weight > 0



def test_risk_control_defaults():
    cfg = _cfg()
    assert cfg.strategy.extreme_liquidates is True
    assert cfg.strategy.min_trim_weight == pytest.approx(0.02)
    assert cfg.strategy.risk_trim_mode == "scale"



def test_position_budget_is_enforced_without_correlation_filter():
    """Regression: the V2 shortcut must not bypass the holdings limit."""

    panel = _make_panel(n_dates=60, n_symbols=5, seed=3)
    cfg = _cfg()
    cfg.strategy.use_corr_filter = False
    cfg.strategy.max_positions = 2
    cfg.strategy.max_new_per_day = 3
    features = build_v2_features(panel, cfg)
    t = 40
    signal = np.zeros(panel.valid.shape, dtype=bool)
    signal[t, :] = True
    features.arrays["entry_signal"] = signal
    features.arrays["entry_score"] = np.where(signal, 1.0, np.nan).astype(np.float32)
    symbols = list(panel.symbols)
    at_limit, _ = select_new_candidates(features, t, symbols[:2], {}, cfg)
    assert at_limit == []
    below_limit, _ = select_new_candidates(features, t, symbols[:1], {}, cfg)
    assert len(below_limit) == 1
    empty_book, _ = select_new_candidates(features, t, [], {}, cfg)
    assert len(empty_book) == 2  # capped by max_positions, not by max_new_per_day


def test_exiting_positions_free_their_slot():
    panel = _make_panel(n_dates=60, n_symbols=5, seed=3)
    cfg = _cfg()
    cfg.strategy.use_corr_filter = False
    cfg.strategy.max_positions = 2
    cfg.strategy.max_new_per_day = 3
    features = build_v2_features(panel, cfg)
    t = 40
    signal = np.zeros(panel.valid.shape, dtype=bool)
    signal[t, :] = True
    features.arrays["entry_signal"] = signal
    features.arrays["entry_score"] = np.where(signal, 1.0, np.nan).astype(np.float32)
    symbols = list(panel.symbols)
    exits = {symbols[0]: "time_exit"}
    selected, _ = select_new_candidates(features, t, symbols[:2], exits, cfg)
    assert len(selected) == 1

