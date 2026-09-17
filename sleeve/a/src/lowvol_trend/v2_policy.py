"""Policy-consistent entry/exit simulation shared by labels and the strategy.

The V2 design requires the label generator and the live strategy to use one
single exit policy implementation.  This module owns that simulator: given an
entry mask it returns, for every attempted entry, whether it filled, when it
exited, why, and the net return after fees.

Everything is causal: the decision uses data up to and including the signal
close T; the entry fills at the T+1 close under the frozen price cap; exit
signals are raised on closes and filled on the next tradable close, respecting
limit-down and suspension blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureStore
from .v2 import _build_limit_matrix, _fees_buy, _fees_sell

REASON_NAMES = np.array(
    ["not_filled", "stop_loss", "profit_target", "trailing_stop", "market_extreme", "time_exit", "stale_loser", "profit_lock"],
    dtype=object,
)
REASON_CODES = {name: i for i, name in enumerate(REASON_NAMES)}


@dataclass
class ExitPolicy:
    """Frozen single-stock exit policy.

    Prices are compared on the adjusted close path (dividend adjusted), while
    the realised P&L is computed on raw closes, exactly like the account does.
    """

    stop_atr_mult: float = 2.0
    stop_min: float = 0.04
    stop_max: float = 0.08
    use_stop: bool = True
    profit_target_pct: Optional[float] = None
    # Express the profit target in multiples of the position ATR at entry instead
    # of a flat percentage, clamped to a fraction of the entry price.  0 keeps the
    # flat profit_target_pct.
    profit_target_atr_mult: float = 0.0
    profit_target_atr_floor: float = 0.02
    profit_target_atr_cap: float = 0.30
    use_trailing: bool = True
    trailing_activate_mult: float = 1.5
    trailing_distance_mult: float = 1.0
    max_hold_days: int = 20
    max_fill_wait_days: int = 15
    exit_on_market_extreme: bool = True
    reference_notional: float = 10_000.0
    exit_below_ma20_days: int = 0
    exit_below_ma60: bool = False
    # Cut positions that have not moved above their entry price after this many
    # trading days: the time cap otherwise forces the losers out at their worst.
    exit_below_entry_days: int = 0
    # Once the position has traded above the profit target, give back at most this
    # fraction from its running maximum instead of selling at the first touch.
    profit_lock_pct: float = 0.0

    def stop_fraction(self, atr20: np.ndarray, close: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = self.stop_atr_mult * atr20 / close
        out = np.clip(raw, self.stop_min, self.stop_max)
        return np.where(np.isfinite(out), out, self.stop_max)

    def describe(self) -> Dict[str, object]:
        return {
            "stop_atr_mult": self.stop_atr_mult,
            "stop_min": self.stop_min,
            "stop_max": self.stop_max,
            "use_stop": self.use_stop,
            "profit_target_pct": self.profit_target_pct,
            "use_trailing": self.use_trailing,
            "trailing_activate_mult": self.trailing_activate_mult,
            "trailing_distance_mult": self.trailing_distance_mult,
            "max_hold_days": self.max_hold_days,
            "exit_on_market_extreme": self.exit_on_market_extreme,
            "exit_below_ma20_days": self.exit_below_ma20_days,
            "exit_below_ma60": self.exit_below_ma60,
            "exit_below_entry_days": self.exit_below_entry_days,
            "profit_lock_pct": self.profit_lock_pct,
        }


def simulate_entries(
    features: FeatureStore,
    cfg: Config,
    entry_mask: np.ndarray,
    policy: ExitPolicy,
    market_extreme: Optional[np.ndarray] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    buy_premium: Optional[float] = None,
    return_frame: bool = True,
) -> "pd.DataFrame | Dict[str, np.ndarray]":
    """Simulate every (signal day, stock) pair flagged by the entry mask.

    Returns one row per attempted entry, including unfilled attempts
    (entry_filled == False), because the design requires failed orders to be
    recorded separately instead of being silently treated as losses.
    """

    panel = features.panel
    close_adj = features.arr("close").astype(np.float64)
    raw_close = features.arr("raw_close").astype(np.float64)
    atr20 = features.arr("atr20").astype(np.float64)
    ma20 = features.arr("ma20").astype(np.float64)
    ma60 = features.arr("ma60").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    valid = np.asarray(panel.valid)
    limit_matrix = _build_limit_matrix(panel, cfg)
    n_dates = len(panel.dates)
    end_index = n_dates if end_index is None else min(int(end_index), n_dates)
    start_index = max(0, int(start_index))
    last_index = end_index - 1
    premium = cfg.execution.buy_premium if buy_premium is None else float(buy_premium)

    if market_extreme is None:
        extreme_flags = np.zeros(n_dates, dtype=bool)
    else:
        extreme_flags = np.asarray(market_extreme, dtype=bool)

    horizon = int(policy.max_hold_days) + int(policy.max_fill_wait_days) + 5
    offsets = np.arange(1, horizon + 1)
    symbols = np.asarray(panel.symbols, dtype=object)

    # Per-day numpy chunks keep peak memory near 1/4 of a list-of-objects
    # table for the 8M-row label build.
    rows: Dict[str, List[np.ndarray]] = {
        "signal_index": [],
        "entry_index": [],
        "symbol_index": [],
        "entry_filled": [],
        "exit_fill_index": [],
        "exit_signal_index": [],
        "exit_reason_code": [],
        "net_return": [],
        "gross_return": [],
        "hold_days": [],
        "stop_pct": [],
    }

    for t in range(start_index, last_index):
        entry_idx = t + 1
        cand = np.flatnonzero(entry_mask[t])
        if cand.size == 0:
            continue
        prev_raw = raw_close[t, cand]
        e_raw = raw_close[entry_idx, cand]
        e_adj = close_adj[entry_idx, cand]
        entry_ok = (
            valid[entry_idx, cand]
            & np.isfinite(e_raw)
            & np.isfinite(prev_raw)
            & (prev_raw > 0)
            & (e_raw <= prev_raw * (1.0 + premium))
            & np.isfinite(e_adj)
            & (e_adj > 0)
        )
        limit_pct = limit_matrix[entry_idx, cand]
        entry_ok &= ~(e_raw >= prev_raw * (1.0 + limit_pct) - 0.001)

        n_all = cand.size
        rows["signal_index"].append(np.full(n_all, t, dtype=np.int64))
        rows["entry_index"].append(np.full(n_all, entry_idx, dtype=np.int64))
        rows["symbol_index"].append(cand)
        if not entry_ok.any():
            rows["entry_filled"].append(np.zeros(n_all, dtype=bool))
            rows["exit_fill_index"].append(np.full(n_all, -1, dtype=np.int64))
            rows["exit_signal_index"].append(np.full(n_all, -1, dtype=np.int64))
            rows["exit_reason_code"].append(np.zeros(n_all, dtype=np.int8))
            rows["net_return"].append(np.full(n_all, np.nan))
            rows["gross_return"].append(np.full(n_all, np.nan))
            rows["hold_days"].append(np.full(n_all, -1, dtype=np.int64))
            rows["stop_pct"].append(np.full(n_all, np.nan))
            continue

        idx = cand[entry_ok]
        n = idx.size
        entry_indices = np.full(n, entry_idx, dtype=np.int64)
        entry_adj = close_adj[entry_idx, idx]
        entry_raw = raw_close[entry_idx, idx]
        stop_pct = policy.stop_fraction(atr20[t, idx], close_adj[t, idx])

        path = entry_idx + offsets[:, None]
        path_clipped = np.minimum(path, last_index)
        path_valid = path <= last_index
        adj_path = np.where(path_valid, close_adj[path_clipped, idx], np.nan)
        raw_path = np.where(path_valid, raw_close[path_clipped, idx], np.nan)
        vol_path = np.where(path_valid, volume[path_clipped, idx], 0.0)
        valid_path = np.where(path_valid, valid[path_clipped, idx], False)

        if policy.use_stop:
            stop_hit = adj_path <= entry_adj[None, :] * (1.0 - stop_pct[None, :])
        else:
            stop_hit = np.zeros_like(adj_path, dtype=bool)
        if policy.profit_target_pct is not None:
            target_hit = adj_path >= entry_adj[None, :] * (1.0 + float(policy.profit_target_pct))
        else:
            target_hit = np.zeros_like(adj_path, dtype=bool)
        if policy.use_trailing or policy.profit_lock_pct:
            run_max = np.fmax.accumulate(np.where(np.isfinite(adj_path), adj_path, -np.inf), axis=0)
            armed = run_max >= entry_adj[None, :] * (1.0 + policy.trailing_activate_mult * stop_pct[None, :])
            trailing_hit = armed & (adj_path <= run_max * (1.0 - policy.trailing_distance_mult * stop_pct[None, :]))
        else:
            trailing_hit = np.zeros_like(adj_path, dtype=bool)

        if policy.exit_on_market_extreme:
            extreme_path = extreme_flags[path_clipped] & path_valid
        else:
            extreme_path = np.zeros_like(adj_path, dtype=bool)

        ma20_path = np.where(path_valid, ma20[path_clipped, idx], np.nan)
        ma60_path = np.where(path_valid, ma60[path_clipped, idx], np.nan)
        if policy.exit_below_ma20_days and policy.exit_below_ma20_days > 0:
            below = adj_path < ma20_path
            k = int(policy.exit_below_ma20_days)
            below_run = np.zeros_like(below, dtype=np.int32)
            for r in range(below.shape[0]):
                prev = below_run[r - 1] if r else 0
                below_run[r] = np.where(below[r], prev + 1, 0)
            ma20_hit = below_run >= k
        else:
            ma20_hit = np.zeros_like(adj_path, dtype=bool)
        if policy.exit_below_ma60:
            ma60_hit = adj_path < ma60_path
        else:
            ma60_hit = np.zeros_like(adj_path, dtype=bool)

        if policy.exit_below_entry_days and policy.exit_below_entry_days > 0:
            stale_hit = (offsets[:, None] >= int(policy.exit_below_entry_days)) & (adj_path < entry_adj[None, :])
        else:
            stale_hit = np.zeros_like(adj_path, dtype=bool)

        time_hit = np.zeros_like(adj_path, dtype=bool)
        time_row = int(policy.max_hold_days) - 1
        if 0 <= time_row < horizon:
            time_hit[time_row, :] = path_valid[time_row, :]

        if policy.profit_lock_pct and policy.profit_target_pct is not None:
            lock_armed = run_max >= entry_adj[None, :] * (1.0 + float(policy.profit_target_pct))
            lock_hit = lock_armed & (adj_path <= run_max * (1.0 - float(policy.profit_lock_pct)))
        else:
            lock_hit = np.zeros_like(adj_path, dtype=bool)
        if policy.profit_lock_pct and policy.profit_target_pct is not None:
            target_hit = np.zeros_like(target_hit, dtype=bool)
        triggers = stop_hit | target_hit | trailing_hit | extreme_path | time_hit | ma20_hit | ma60_hit | stale_hit | lock_hit
        any_trigger = triggers.any(axis=0)
        first_trigger = np.argmax(triggers, axis=0)

        code = np.zeros(n, dtype=np.int8)
        code = np.where(stale_hit.any(axis=0), 6, code)
        code = np.where(lock_hit.any(axis=0), 7, code)
        code = np.where(time_hit.any(axis=0), 5, code)
        code = np.where(extreme_path.any(axis=0), 4, code)
        code = np.where(ma20_hit.any(axis=0) | ma60_hit.any(axis=0), 5, code)
        code = np.where(trailing_hit.any(axis=0), 3, code)
        code = np.where(target_hit.any(axis=0), 2, code)
        code = np.where(stop_hit.any(axis=0), 1, code)

        fill_ok = valid_path & np.isfinite(raw_path) & (vol_path > 0)
        prev_path_raw = np.where(path_valid, raw_close[np.maximum(path_clipped - 1, 0), idx], np.nan)
        limit_down_locked = raw_path <= prev_path_raw * (1.0 - limit_matrix[path_clipped, idx]) + 0.001
        fill_ok &= ~limit_down_locked
        fill_pos = np.full(n, -1, dtype=np.int64)
        row_indices = np.arange(horizon)[:, None]
        for k in range(horizon):
            eligible = (row_indices[k] > first_trigger) & fill_ok[k] & (fill_pos < 0) & any_trigger
            fill_pos[eligible] = k
        filled = any_trigger & (fill_pos >= 0)
        exit_signal_index = np.where(
            any_trigger, np.take_along_axis(path_clipped, first_trigger[None, :], axis=0)[0], -1
        )
        exit_fill_index = np.where(
            filled, np.take_along_axis(path_clipped, np.maximum(fill_pos, 0)[None, :], axis=0)[0], -1
        )

        notional = policy.reference_notional
        buy_fee = _fees_buy(np.full(n, notional, dtype=np.float64), cfg)
        sell_price = np.where(filled, raw_close[np.maximum(exit_fill_index, 0), idx], np.nan)
        qty = notional / np.where(np.isfinite(entry_raw) & (entry_raw > 0), entry_raw, np.nan)
        sell_notional = qty * sell_price
        sell_fee = np.zeros(n, dtype=np.float64)
        if filled.any():
            date_ints = panel.date_ints[np.maximum(exit_fill_index, 0)]
            sell_fee = _fees_sell(np.nan_to_num(sell_notional, nan=0.0), date_ints, cfg)
        net = np.where(
            filled & np.isfinite(sell_notional),
            (sell_notional - sell_fee - notional - buy_fee) / notional,
            np.nan,
        )
        gross = np.where(filled & np.isfinite(sell_price), sell_price / entry_raw - 1.0, np.nan)
        hold = np.where(filled, exit_fill_index - entry_idx + 1, -1)

        rows["entry_filled"].append(entry_ok)
        fill_out = np.full(n_all, -1, dtype=np.int64)
        fill_out[entry_ok] = exit_fill_index
        rows["exit_fill_index"].append(fill_out)
        sig_out = np.full(n_all, -1, dtype=np.int64)
        sig_out[entry_ok] = exit_signal_index
        rows["exit_signal_index"].append(sig_out)
        reason_out = np.zeros(n_all, dtype=np.int8)
        reason_out[entry_ok] = code
        rows["exit_reason_code"].append(reason_out)
        net_out = np.full(n_all, np.nan)
        net_out[entry_ok] = net
        rows["net_return"].append(net_out)
        gross_out = np.full(n_all, np.nan)
        gross_out[entry_ok] = gross
        rows["gross_return"].append(gross_out)
        hold_out = np.full(n_all, -1, dtype=np.int64)
        hold_out[entry_ok] = hold
        rows["hold_days"].append(hold_out)
        stop_out = np.full(n_all, np.nan)
        stop_out[entry_ok] = stop_pct
        rows["stop_pct"].append(stop_out)

    columns = {}
    for key, value in rows.items():
        columns[key] = np.concatenate(value) if value else np.array([])
        rows[key] = None
    if not return_frame:
        return columns
    frame = pd.DataFrame(columns)
    del columns
    if frame.empty:
        return frame
    frame["exit_reason"] = REASON_NAMES[frame["exit_reason_code"].to_numpy(dtype=np.int64)]
    frame["signal_date"] = panel.dates[frame["signal_index"].to_numpy()]
    frame["entry_date"] = panel.dates[frame["entry_index"].to_numpy()]
    frame["instrument"] = symbols[frame["symbol_index"].to_numpy()]
    fill_idx = frame["exit_fill_index"].to_numpy(dtype=np.int64)
    date_values = panel.dates.to_numpy(dtype="datetime64[ns]")
    exit_dates = np.full(len(fill_idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_exit = fill_idx >= 0
    if valid_exit.any():
        exit_dates[valid_exit] = date_values[fill_idx[valid_exit]]
    frame["exit_date"] = exit_dates
    frame["matrix_row"] = np.arange(len(frame), dtype=np.int64)
    frame["win"] = np.where(frame["net_return"].notna(), (frame["net_return"] > 0).astype(float), np.nan)
    return frame


def policy_stats(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {"n": 0, "n_attempt": 0, "win": float("nan"), "ret": float("nan")}
    filled = frame[frame["entry_filled"] & frame["net_return"].notna()]
    if filled.empty:
        return {"n": 0, "n_attempt": int(len(frame)), "win": float("nan"), "ret": float("nan")}
    wins = filled[filled["net_return"] > 0]["net_return"]
    losses = filled[filled["net_return"] <= 0]["net_return"]
    span_years = max(1e-9, (int(filled["signal_index"].max()) - int(filled["signal_index"].min()) + 1) / 243.0)
    return {
        "n": int(len(filled)),
        "n_attempt": int(len(frame)),
        "fill_rate": float(frame["entry_filled"].mean()),
        "win": float((filled["net_return"] > 0).mean()),
        "ret": float(filled["net_return"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "hold": float(filled["hold_days"].mean()),
        "per_year": float(len(filled) / span_years),
    }
