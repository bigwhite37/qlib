"""Stock-level features, daily universe filters and rule-based scores.

Everything is computed from information up to and including date ``T``.  The
backtest engine is responsible for the T -> T+1 timing; nothing in this module
looks forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data import PanelData
from .market import MarketState, compute_market_state


def row_pct_rank(values: np.ndarray, mask: np.ndarray, block: int = 256) -> np.ndarray:
    """Cross-sectional percentile rank (1.0 = largest) with NaN in masked-out cells.

    Computed in date blocks so the temporary float64 frame never covers the whole
    panel at once (the panel-wide version needed roughly 200 MB per call).
    """

    values = np.asarray(values)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    n_rows = values.shape[0]
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        masked = np.where(mask[start:stop] & np.isfinite(values[start:stop]), values[start:stop], np.nan)
        ranked = pd.DataFrame(masked).rank(axis=1, pct=True, na_option="keep").to_numpy(dtype=np.float64)
        out[start:stop] = ranked.astype(np.float32)
    return out


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


class LaggedFeatureStore:
    """Shift signal/exit features by ``lag`` trading days for stress testing.

    Execution-related fields (raw close, factor, volume, ADTV) stay at the
    decision day so order quantity and price constraints remain exactly the
    same as the frozen plan.  This approximates a delayed decision/exit plan.
    """

    EXEMPT = {"raw_close", "factor", "volume", "amount", "adtv20_shares", "amount20_yuan"}

    def __init__(self, base: "FeatureStore", lag: int = 1):
        self.base = base
        self.lag = int(lag)
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def panel(self):
        return self.base.panel

    @property
    def market(self):
        return self.base.market

    @property
    def dates(self):
        return self.base.dates

    @property
    def symbols(self):
        return self.base.symbols

    @property
    def date_to_index(self):
        return self.base.date_to_index

    @property
    def symbol_to_index(self):
        return self.base.symbol_to_index

    @property
    def diagnostics(self):
        return self.base.diagnostics

    def arr(self, name: str) -> np.ndarray:
        if self.lag <= 0 or name in self.EXEMPT:
            return self.base.arr(name)
        if name in self._cache:
            return self._cache[name]
        source = self.base.arr(name)
        if source.dtype == bool:
            out = np.zeros_like(source, dtype=bool)
        else:
            out = np.full_like(source, np.nan, dtype=source.dtype)
        if self.lag < source.shape[0]:
            out[self.lag :] = source[: -self.lag]
        self._cache[name] = out
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arr(name)

    def df(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.arr(name), index=self.dates, columns=self.symbols)

    def decision_index(self, execution_time) -> int:
        return self.base.decision_index(execution_time)


@dataclass
class FeatureStore:
    panel: PanelData
    cfg: Config
    market: MarketState
    arrays: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def arr(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature {name!r}; available={sorted(self.arrays)}") from exc

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arr(name)

    def df(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.arr(name), index=self.panel.dates, columns=self.panel.symbols)

    @property
    def date_to_index(self) -> Dict[pd.Timestamp, int]:
        if not hasattr(self, "_date_to_index"):
            self._date_to_index = {pd.Timestamp(d).normalize(): i for i, d in enumerate(self.dates)}
        return self._date_to_index

    @property
    def symbol_to_index(self) -> Dict[str, int]:
        if not hasattr(self, "_symbol_to_index"):
            self._symbol_to_index = {s: i for i, s in enumerate(self.symbols)}
        return self._symbol_to_index

    def decision_index(self, execution_time) -> int:
        """Return the decision (previous trading day) index for an execution date."""

        execution_date = pd.Timestamp(execution_time).normalize()
        idx = self.date_to_index.get(execution_date)
        if idx is None:
            # Fall back to the last date not after execution_time.
            pos = int(np.searchsorted(self.dates, execution_date, side="right")) - 1
            if pos < 0:
                return -1
            idx = pos
        return idx - 1

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates

    @property
    def symbols(self) -> List[str]:
        return self.panel.symbols


def build_features(panel: PanelData, cfg: Config) -> FeatureStore:
    dates = panel.dates
    symbols = panel.symbols
    n_dates, n_symbols = panel.n_dates, panel.n_symbols

    close = pd.DataFrame(panel.field("close"), index=dates, columns=symbols, copy=False)
    high = pd.DataFrame(panel.field("high"), index=dates, columns=symbols, copy=False)
    low = pd.DataFrame(panel.field("low"), index=dates, columns=symbols, copy=False)
    factor = pd.DataFrame(panel.field("factor"), index=dates, columns=symbols, copy=False)
    volume = pd.DataFrame(panel.field("volume"), index=dates, columns=symbols, copy=False)
    amount = pd.DataFrame(panel.field("amount"), index=dates, columns=symbols, copy=False)
    valid = pd.DataFrame(panel.valid, index=dates, columns=symbols, copy=False)

    raw_close = close / factor
    raw_close[~np.isfinite(raw_close)] = np.nan
    raw_volume_shares = volume * factor * 100.0
    raw_volume_shares[~np.isfinite(raw_volume_shares)] = np.nan
    del volume
    amount_yuan = amount * 1000.0
    amount_yuan[~np.isfinite(amount_yuan)] = np.nan
    del amount

    ret1 = close.pct_change(fill_method=None)
    prev_close = close.shift(1)

    ma_fast = close.rolling(cfg.signal.ma_fast, min_periods=cfg.signal.ma_fast).mean()
    ma_mid = close.rolling(cfg.signal.ma_mid, min_periods=cfg.signal.ma_mid).mean()
    ma_slow = close.rolling(cfg.signal.ma_slow, min_periods=cfg.signal.ma_slow).mean()
    ma_slow_prev = ma_slow.shift(5)
    ma_fast_prev = ma_fast.shift(1)
    prev_close_adj = close.shift(1)

    true_range = np.maximum(
        (high - low).to_numpy(dtype=np.float64),
        np.maximum(
            (high - prev_close).abs().to_numpy(dtype=np.float64),
            (low - prev_close).abs().to_numpy(dtype=np.float64),
        ),
    )
    atr = pd.DataFrame(true_range, index=dates, columns=symbols, copy=False).rolling(
        cfg.signal.atr_window, min_periods=cfg.signal.atr_window
    ).mean()

    high20 = close.rolling(cfg.signal.drawdown_lookback, min_periods=cfg.signal.drawdown_lookback).max()
    dist_high20 = high20 - close

    vol60 = ret1.rolling(cfg.signal.rs_window, min_periods=cfg.signal.rs_window).std()
    negative = ret1.where(ret1 < 0.0, 0.0)
    downvol20 = (
        negative.pow(2)
        .rolling(cfg.signal.atr_window, min_periods=cfg.signal.atr_window)
        .mean()
        .pow(0.5)
    )
    del negative

    net_move = close - close.shift(cfg.signal.rs_window)
    swing = (close - close.shift(1)).abs().rolling(cfg.signal.rs_window, min_periods=cfg.signal.rs_window).sum()
    trend_quality = _safe_ratio(net_move.to_numpy(dtype=np.float64), swing.to_numpy(dtype=np.float64))
    del net_move, swing
    trend_quality = pd.DataFrame(trend_quality, index=dates, columns=symbols)

    ret60 = close.pct_change(cfg.signal.rs_window, fill_method=None)
    adtv20 = raw_volume_shares.rolling(
        cfg.execution.volume_lookback, min_periods=10
    ).mean()
    del raw_volume_shares
    amount20 = amount_yuan.rolling(cfg.execution.volume_lookback, min_periods=10).mean()
    del amount_yuan

    below_ma20 = close < ma_mid
    two_below_ma20 = below_ma20 & below_ma20.shift(1, fill_value=False)
    del below_ma20

    valid_count = valid.cumsum()
    valid_recent = valid.rolling(cfg.universe.recent_window, min_periods=1).sum()

    # ------------------------------------------------------------------
    # Market state (needed before relative strength is meaningful)
    # ------------------------------------------------------------------
    market_eligible = panel.valid & (valid_count.to_numpy(dtype=np.float32) >= cfg.market.min_history_days_market)
    market = compute_market_state(
        ret=ret1.to_numpy(dtype=np.float32),
        close=close.to_numpy(dtype=np.float32),
        ma60=ma_slow.to_numpy(dtype=np.float32),
        ma20=ma_mid.to_numpy(dtype=np.float32),
        eligible=market_eligible,
        cfg=cfg.market,
    )
    del market_eligible
    proxy_ret60 = market.proxy_return_60.astype(np.float32)[:, None]
    rs60 = ret60.to_numpy(dtype=np.float32) - proxy_ret60
    rs60[~np.isfinite(ret60.to_numpy(dtype=np.float32))] = np.nan

    # ------------------------------------------------------------------
    # Base tradable universe (design section 3)
    # ------------------------------------------------------------------
    base = panel.valid.copy()
    base &= valid_count.to_numpy(dtype=np.float64) >= cfg.universe.min_history_days
    base &= valid_recent.to_numpy(dtype=np.float64) >= cfg.universe.min_recent_bars
    base &= np.isfinite(raw_close.to_numpy(dtype=np.float32)) & (
        raw_close.to_numpy(dtype=np.float32) >= cfg.universe.min_raw_price
    )
    base &= np.isfinite(amount20.to_numpy(dtype=np.float32)) & (amount20.to_numpy(dtype=np.float32) > 0)
    base &= np.isfinite(vol60.to_numpy(dtype=np.float32))
    base &= np.isfinite(ma_fast.to_numpy(dtype=np.float32))
    base &= np.isfinite(ma_mid.to_numpy(dtype=np.float32))
    base &= np.isfinite(ma_slow.to_numpy(dtype=np.float32))
    base &= np.isfinite(atr.to_numpy(dtype=np.float32)) & (atr.to_numpy(dtype=np.float32) > 0)

    amount_rank = row_pct_rank(amount20.to_numpy(dtype=np.float32), np.isfinite(amount20.to_numpy(dtype=np.float32)))
    vol_rank = row_pct_rank(vol60.to_numpy(dtype=np.float32), np.isfinite(vol60.to_numpy(dtype=np.float32)))
    base &= amount_rank >= (1.0 - cfg.universe.liquidity_keep_quantile)
    base &= vol_rank <= cfg.universe.vol_keep_quantile

    # ------------------------------------------------------------------
    # Cross-sectional ranks and composite score (design section 5.2)
    # ------------------------------------------------------------------
    rs_rank = row_pct_rank(rs60, base)
    tq_rank = row_pct_rank(trend_quality.to_numpy(dtype=np.float32), base)
    dv_rank = row_pct_rank(-downvol20.to_numpy(dtype=np.float32), base)
    score = (
        cfg.signal.score_w_rs * rs_rank
        + cfg.signal.score_w_trend_quality * tq_rank
        + cfg.signal.score_w_low_downvol * dv_rank
    ).astype(np.float32)
    del tq_rank, dv_rank
    score[~base] = np.nan
    score_rank = row_pct_rank(score, base)
    top_half = base & (score_rank >= 0.5)

    # ------------------------------------------------------------------
    # Entry signal (design section 5.1)
    # ------------------------------------------------------------------
    close_np = close.to_numpy(dtype=np.float32)
    prev_close_np = prev_close_adj.to_numpy(dtype=np.float32)
    ma_fast_np = ma_fast.to_numpy(dtype=np.float32)
    ma_fast_prev_np = ma_fast_prev.to_numpy(dtype=np.float32)
    ma_mid_np = ma_mid.to_numpy(dtype=np.float32)
    ma_slow_np = ma_slow.to_numpy(dtype=np.float32)
    ma_slow_prev_np = ma_slow_prev.to_numpy(dtype=np.float32)
    atr_np = atr.to_numpy(dtype=np.float32)
    dist_np = dist_high20.to_numpy(dtype=np.float32)
    ret1_np = ret1.to_numpy(dtype=np.float32)

    del ma_slow_prev, ma_fast_prev, prev_close_adj, prev_close
    trend_ok = (
        np.isfinite(ma_slow_prev_np)
        & (close_np > ma_slow_np)
        & (ma_mid_np > ma_slow_np)
        & (ma_slow_np > ma_slow_prev_np)
    )
    rs_ok = np.isfinite(rs_rank) & (rs_rank >= cfg.signal.rs_top_quantile)
    drawdown_ok = (
        np.isfinite(dist_np)
        & np.isfinite(atr_np)
        & (dist_np >= cfg.signal.min_drawdown_atr * atr_np)
        & (dist_np <= cfg.signal.max_drawdown_atr * atr_np)
    )
    repair_ok = (
        np.isfinite(ret1_np)
        & (ret1_np > 0.0)
        & (ret1_np <= cfg.signal.max_daily_gain)
        & np.isfinite(ma_fast_np)
        & np.isfinite(ma_fast_prev_np)
        & np.isfinite(prev_close_np)
        & (close_np > ma_fast_np)
        & (prev_close_np <= ma_fast_prev_np)
    )
    entry_signal = base & trend_ok & rs_ok & drawdown_ok & repair_ok
    entry_score = np.where(entry_signal, score, np.nan).astype(np.float32)
    del high, low

    # ------------------------------------------------------------------
    # Persist float32 arrays; the temporary dataframes can now be released.
    # ------------------------------------------------------------------
    arrays: Dict[str, np.ndarray] = {
        "close": close_np,
        "raw_close": raw_close.to_numpy(dtype=np.float32),
        "factor": factor.to_numpy(dtype=np.float32),
        "ma5": ma_fast_np,
        "ma20": ma_mid_np,
        "ma60": ma_slow_np,
        "ma60_prev5": ma_slow_prev_np,
        "atr20": atr_np,
        "high20": high20.to_numpy(dtype=np.float32),
        "dist_high20": dist_np,
        "ret1": ret1_np,
        "ret60": ret60.to_numpy(dtype=np.float32),
        "rs60": rs60.astype(np.float32),
        "vol60": vol60.to_numpy(dtype=np.float32),
        "downvol20": downvol20.to_numpy(dtype=np.float32),
        "trend_quality60": trend_quality.to_numpy(dtype=np.float32),
        "adtv20_shares": adtv20.to_numpy(dtype=np.float32),
        "amount20_yuan": amount20.to_numpy(dtype=np.float32),
        "two_below_ma20": two_below_ma20.to_numpy(dtype=bool),
        "valid_count": valid_count.to_numpy(dtype=np.int32),
        "valid_recent20": valid_recent.to_numpy(dtype=np.int16),
        "base_pass": base.astype(bool),
        "rs_rank": rs_rank.astype(np.float32),
        "score": score.astype(np.float32),
        "score_rank": score_rank.astype(np.float32),
        "top_half": top_half.astype(bool),
        "entry_signal": entry_signal.astype(bool),
        "entry_score": entry_score.astype(np.float32),
        "amount_rank": amount_rank.astype(np.float32),
        "vol_rank": vol_rank.astype(np.float32),
    }

    diagnostics = {
        "n_base_pass_last": int(base[-1].sum()),
        "n_entry_last": int(entry_signal[-1].sum()),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
        "n_symbols": n_symbols,
        "n_dates": n_dates,
    }
    return FeatureStore(panel=panel, cfg=cfg, market=market, arrays=arrays, diagnostics=diagnostics)
