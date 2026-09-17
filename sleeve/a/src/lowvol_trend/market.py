"""Market-state indicators and the four-state position cap."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MarketConfig

STATE_NAMES = ("extreme", "weak", "neutral", "strong")
STATE_TO_CAP = {"extreme": 0.0, "weak": 0.25, "neutral": 0.55, "strong": 0.90}
CAP_TO_STATE = {v: k for k, v in STATE_TO_CAP.items()}


@dataclass
class MarketState:
    median_return: np.ndarray
    proxy: np.ndarray
    proxy_ma: np.ndarray
    proxy_short_ma: np.ndarray
    breadth_60: np.ndarray
    breadth_20: np.ndarray
    drop_diffusion: np.ndarray
    raw_state: np.ndarray  # object array of state names
    raw_cap: np.ndarray
    effective_state: np.ndarray
    effective_cap: np.ndarray
    weak_recovery_ok: np.ndarray
    entry_allowed: np.ndarray
    proxy_return_60: np.ndarray

    def date_labels(self, dates: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(self.effective_state, index=dates)


def _pct_change_skipna(series: pd.Series, periods: int) -> pd.Series:
    base = series.shift(periods)
    return series / base - 1.0


def _cap_to_state(cap: float) -> str:
    best = min(STATE_TO_CAP, key=lambda name: abs(STATE_TO_CAP[name] - float(cap)))
    return best


def compute_market_state(
    ret: np.ndarray,
    close: np.ndarray,
    ma60: np.ndarray,
    ma20: np.ndarray,
    eligible: np.ndarray,
    cfg: MarketConfig,
) -> MarketState:
    """Compute the market proxy, breadth, drop diffusion and effective state."""

    n_dates = ret.shape[0]
    ret_masked = np.where(eligible & np.isfinite(ret), ret, np.nan)
    # Suppress warnings for the very first warm-up rows where no stock is
    # eligible yet; those rows are excluded from the backtest anyway.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median_ret = np.nanmedian(ret_masked, axis=1)
        breadth_60 = np.nanmean(np.where(eligible & np.isfinite(ma60), close > ma60, np.nan), axis=1)
        breadth_20 = np.nanmean(np.where(eligible & np.isfinite(ma20), close > ma20, np.nan), axis=1)
        drop_diffusion = np.nanmean(np.where(eligible & np.isfinite(ret), ret < cfg.drop_threshold, np.nan), axis=1)
    median_ret = np.nan_to_num(median_ret, nan=0.0)
    breadth_60 = np.nan_to_num(breadth_60, nan=0.0)
    breadth_20 = np.nan_to_num(breadth_20, nan=0.0)
    drop_diffusion = np.nan_to_num(drop_diffusion, nan=0.0)

    proxy = np.ones(n_dates, dtype=np.float64)
    for t in range(1, n_dates):
        proxy[t] = proxy[t - 1] * (1.0 + float(median_ret[t]))
    proxy_series = pd.Series(proxy)
    proxy_ma = proxy_series.rolling(cfg.ma_window, min_periods=cfg.ma_window).mean().to_numpy()
    proxy_short_ma = proxy_series.rolling(cfg.short_ma_window, min_periods=cfg.short_ma_window).mean().to_numpy()
    ret5 = np.full(n_dates, np.nan)
    if n_dates > cfg.lookback_5d:
        ret5[cfg.lookback_5d :] = proxy[cfg.lookback_5d :] / proxy[:-cfg.lookback_5d] - 1.0
    ret60 = _pct_change_skipna(proxy_series, 60).to_numpy()

    raw_state: List[str] = []
    raw_cap = np.zeros(n_dates, dtype=np.float64)
    for t in range(n_dates):
        if np.isfinite(ret5[t]) and ret5[t] <= cfg.extreme_5d_return:
            state = "extreme"
        elif breadth_60[t] < cfg.extreme_breadth and drop_diffusion[t] > cfg.extreme_drop_diffusion:
            state = "extreme"
        elif (
            np.isfinite(proxy_ma[t])
            and proxy[t] > proxy_ma[t]
            and np.isfinite(proxy_short_ma[t])
            and t >= 5
            and np.isfinite(proxy_short_ma[t - 5])
            and proxy_short_ma[t] > proxy_short_ma[t - 5]
            and breadth_60[t] >= cfg.strong_breadth
        ):
            state = "strong"
        elif np.isfinite(proxy_ma[t]) and proxy[t] < proxy_ma[t] and breadth_60[t] < cfg.weak_breadth:
            state = "weak"
        else:
            state = "neutral"
        raw_state.append(state)
        raw_cap[t] = STATE_TO_CAP[state]

    effective_cap = np.zeros(n_dates, dtype=np.float64)
    upgrade_streak = 0
    for t in range(n_dates):
        if t == 0:
            effective_cap[t] = raw_cap[t]
            continue
        prev = effective_cap[t - 1]
        if raw_cap[t] < prev - 1e-12:
            # Risk deterioration takes effect on the next trading day.
            effective_cap[t] = raw_cap[t]
            upgrade_streak = 0
        elif raw_cap[t] > prev + 1e-12:
            upgrade_streak += 1
            if upgrade_streak >= cfg.upgrade_confirm_days:
                effective_cap[t] = raw_cap[t]
            else:
                effective_cap[t] = prev
        else:
            effective_cap[t] = prev
            upgrade_streak = 0

    state_by_cap = np.asarray([_cap_to_state(c) for c in effective_cap], dtype=object)
    raw_state_arr = np.asarray(raw_state, dtype=object)

    weak_recovery = np.zeros(n_dates, dtype=bool)
    lb = int(cfg.weak_recovery_lookback)
    if n_dates > lb:
        weak_recovery[lb:] = (breadth_20[lb:] - breadth_20[:-lb]) >= cfg.weak_recovery_gap
    entry_allowed = effective_cap > 0
    weak_mask = state_by_cap == "weak"
    entry_allowed = entry_allowed & (~weak_mask | weak_recovery)
    entry_allowed[: cfg.min_history_days_market] = False

    return MarketState(
        median_return=median_ret.astype(np.float32),
        proxy=proxy.astype(np.float64),
        proxy_ma=proxy_ma.astype(np.float64),
        proxy_short_ma=proxy_short_ma.astype(np.float64),
        breadth_60=breadth_60.astype(np.float32),
        breadth_20=breadth_20.astype(np.float32),
        drop_diffusion=drop_diffusion.astype(np.float32),
        raw_state=raw_state_arr,
        raw_cap=raw_cap.astype(np.float32),
        effective_state=state_by_cap,
        effective_cap=effective_cap.astype(np.float32),
        weak_recovery_ok=weak_recovery,
        entry_allowed=entry_allowed,
        proxy_return_60=ret60.astype(np.float32),
    )


def market_dataframe(state: MarketState, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "median_return": state.median_return,
            "proxy": state.proxy,
            "proxy_ma60": state.proxy_ma,
            "proxy_ma20": state.proxy_short_ma,
            "breadth_60": state.breadth_60,
            "breadth_20": state.breadth_20,
            "drop_diffusion": state.drop_diffusion,
            "raw_state": state.raw_state,
            "raw_cap": state.raw_cap,
            "effective_state": state.effective_state,
            "effective_cap": state.effective_cap,
            "weak_recovery_ok": state.weak_recovery_ok,
            "entry_allowed": state.entry_allowed,
        },
        index=dates,
    )
