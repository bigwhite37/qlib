"""V2 research pipeline: policy-consistent labels, market state and models.

Implements the pieces of ``docs/chats_002.md`` that differ most from V1:

* a light base universe (no volatility / RS / moving-average hard gates);
* a policy-consistent single-stock exit simulator used for both labels and the
  live strategy;
* two model targets (net policy return and win indicator);
* the V2 market-state ladder with an explicit extreme-risk recovery rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureStore


# ---------------------------------------------------------------------------
# Base universe
# ---------------------------------------------------------------------------
def v2_base_mask(features: FeatureStore, cfg: Config) -> np.ndarray:
    """Light universe constraints from chats_002 section 5.1."""

    panel = features.panel
    base = panel.valid.copy()
    base &= features.arr("valid_count") >= 120
    base &= features.arr("valid_recent20") >= 18
    raw_close = features.arr("raw_close")
    base &= np.isfinite(raw_close) & (raw_close >= 3.0)
    amount20 = features.arr("amount20_yuan")
    base &= np.isfinite(amount20) & (amount20 > 0)
    amount_rank = features.arr("amount_rank")
    floor = float(getattr(cfg.universe, "v2_min_amount_rank", 0.20))
    base &= np.isfinite(amount_rank) & (amount_rank >= floor)
    return base


# ---------------------------------------------------------------------------
# V2 market state
# ---------------------------------------------------------------------------
@dataclass
class V2MarketState:
    proxy_return: np.ndarray
    proxy: np.ndarray
    breadth_20: np.ndarray
    breadth_60: np.ndarray
    drop_diffusion: np.ndarray
    raw_state: np.ndarray
    raw_cap: np.ndarray
    effective_state: np.ndarray
    effective_cap: np.ndarray
    recovery_phase: np.ndarray


def compute_v2_market_state(
    features: FeatureStore,
    base_mask: np.ndarray,
    cfg: Config,
) -> V2MarketState:
    panel = features.panel
    ret = features.arr("ret1")
    close = features.arr("close")
    ma20 = features.arr("ma20")
    ma60 = features.arr("ma60")
    valid = panel.valid
    n_dates = len(panel.dates)

    eligible = base_mask & valid
    masked_ret = np.where(eligible & np.isfinite(ret), ret, np.nan)
    with np.errstate(all="ignore"):
        mean_ret = np.nanmean(masked_ret, axis=1)
        breadth_20 = np.nanmean(np.where(eligible & np.isfinite(ma20), close > ma20, np.nan), axis=1)
        breadth_60 = np.nanmean(np.where(eligible & np.isfinite(ma60), close > ma60, np.nan), axis=1)
        drop_diffusion = np.nanmean(np.where(eligible & np.isfinite(ret), ret < -0.03, np.nan), axis=1)
    mean_ret = np.nan_to_num(mean_ret, nan=0.0)
    breadth_20 = np.nan_to_num(breadth_20, nan=0.0)
    breadth_60 = np.nan_to_num(breadth_60, nan=0.0)
    drop_diffusion = np.nan_to_num(drop_diffusion, nan=0.0)

    proxy = np.ones(n_dates, dtype=np.float64)
    for t in range(1, n_dates):
        proxy[t] = proxy[t - 1] * (1.0 + float(mean_ret[t]))
    proxy_series = pd.Series(proxy)
    ret3 = (proxy_series / proxy_series.shift(3) - 1.0).to_numpy()
    ret5 = (proxy_series / proxy_series.shift(5) - 1.0).to_numpy()
    ret20 = (proxy_series / proxy_series.shift(20) - 1.0).to_numpy()
    ret60 = (proxy_series / proxy_series.shift(60) - 1.0).to_numpy()
    b20_prev5 = np.full(n_dates, np.nan)
    b20_prev5[5:] = breadth_20[:-5]

    raw_state = np.empty(n_dates, dtype=object)
    raw_cap = np.zeros(n_dates, dtype=np.float64)
    for t in range(n_dates):
        extreme = (
            (np.isfinite(ret5[t]) and ret5[t] <= -0.06)
            or (breadth_60[t] < 0.20 and drop_diffusion[t] > 0.35)
        )
        if extreme:
            state, cap = "extreme", 0.0
        elif np.isfinite(ret60[t]) and ret60[t] > 0.0 and breadth_60[t] >= 0.55:
            state, cap = "strong", 0.90
        elif (np.isfinite(ret20[t]) and ret20[t] > 0.0) or breadth_60[t] >= 0.40:
            state, cap = "neutral", 0.70
        else:
            state, cap = "weak", 0.35
        raw_state[t] = state
        raw_cap[t] = cap

    # Extreme-risk recovery protocol:
    #   1) after extreme clears, wait for two days with M3>0 and B20 > B20(-5)
    #      before restoring a 35% cap;
    #   2) then keep 35% for three more non-extreme days before returning to
    #      the normal state table.  No account-NAV recovery condition.
    effective_state = np.empty(n_dates, dtype=object)
    effective_cap = np.zeros(n_dates, dtype=np.float64)
    recovery_phase = np.zeros(n_dates, dtype=np.int8)
    phase = 0
    recovery_streak = 0
    restore_streak = 0
    for t in range(n_dates):
        if raw_state[t] == "extreme":
            phase = 0
            recovery_streak = 0
            restore_streak = 0
            effective_state[t] = "extreme"
            effective_cap[t] = 0.0
            continue
        recovery_signal = (
            np.isfinite(ret3[t])
            and ret3[t] > 0.0
            and np.isfinite(b20_prev5[t])
            and breadth_20[t] > b20_prev5[t]
        )
        if phase == 0:
            effective_state[t] = raw_state[t]
            effective_cap[t] = raw_cap[t]
            if t > 0 and raw_state[t - 1] == "extreme":
                phase = 1
                recovery_streak = 1 if recovery_signal else 0
                effective_state[t] = "weak"
                effective_cap[t] = 0.35
        elif phase == 1:
            recovery_streak = recovery_streak + 1 if recovery_signal else 0
            effective_state[t] = "weak"
            effective_cap[t] = 0.35
            if recovery_streak >= 2:
                phase = 2
                restore_streak = 0
        else:
            restore_streak += 1
            if restore_streak >= 3:
                phase = 0
                effective_state[t] = raw_state[t]
                effective_cap[t] = raw_cap[t]
            else:
                effective_state[t] = "weak"
                effective_cap[t] = 0.35
        recovery_phase[t] = phase
    return V2MarketState(
        proxy_return=mean_ret.astype(np.float32),
        proxy=proxy,
        breadth_20=breadth_20.astype(np.float32),
        breadth_60=breadth_60.astype(np.float32),
        drop_diffusion=drop_diffusion.astype(np.float32),
        raw_state=raw_state,
        raw_cap=raw_cap.astype(np.float32),
        effective_state=effective_state,
        effective_cap=effective_cap.astype(np.float32),
        recovery_phase=recovery_phase,
    )


# ---------------------------------------------------------------------------
# Policy-consistent labels
# ---------------------------------------------------------------------------
@dataclass
class PolicyConfig:
    stop_atr_mult: float = 2.0
    stop_min: float = 0.04
    stop_max: float = 0.08
    trailing_activate_mult: float = 1.5
    trailing_distance_mult: float = 1.0
    max_hold_days: int = 20
    max_fill_wait_days: int = 15
    reference_notional: float = 10_000.0


def _build_limit_matrix(panel, cfg: Config) -> np.ndarray:
    """Board/date price-limit matrix [n_dates, n_symbols]."""

    boards = panel.instrument_meta["board"].reindex(panel.symbols).fillna("unknown").to_numpy(dtype=object)
    n_dates, n_symbols = len(panel.dates), len(panel.symbols)
    dates = panel.date_ints
    limit = np.full((n_dates, n_symbols), 0.10, dtype=np.float32)
    for j, board in enumerate(boards):
        if board == "star":
            limit[:, j] = 0.20
        elif board == "bse":
            limit[:, j] = 0.30
        elif board == "chinext":
            limit[:, j] = np.where(dates >= 20200824, 0.20, 0.10).astype(np.float32)
        elif board in ("sh_main", "sz_main"):
            limit[:, j] = 0.10
        else:
            limit[:, j] = 0.10
    if cfg.execution.unknown_st_conservative:
        main = np.char.startswith(boards.astype(str), "sh_main") | np.char.startswith(boards.astype(str), "sz_main")
        limit[:, main] = np.minimum(limit[:, main], cfg.execution.conservative_limit_pct)
    return limit


def _fees_buy(notional: np.ndarray, cfg: Config) -> np.ndarray:
    commission = np.maximum(notional * cfg.execution.commission_rate, cfg.execution.min_commission)
    extra = notional * (cfg.execution.transfer_fee_rate + cfg.execution.friction_bps_per_side / 10_000.0)
    return commission + extra


def _fees_sell(notional: np.ndarray, date_ints: np.ndarray, cfg: Config) -> np.ndarray:
    commission = np.maximum(notional * cfg.execution.commission_rate, cfg.execution.min_commission)
    change = int(pd.Timestamp(cfg.execution.stamp_tax_change_date).strftime("%Y%m%d"))
    stamp_rate = np.where(
        date_ints < change,
        cfg.execution.stamp_tax_rate_before_2023_08_28,
        cfg.execution.stamp_tax_rate_after_2023_08_28,
    )
    stamp = notional * stamp_rate
    transfer = notional * cfg.execution.transfer_fee_rate
    friction = notional * cfg.execution.friction_bps_per_side / 10_000.0
    return commission + stamp + transfer + friction


def simulate_policy_labels(
    features: FeatureStore,
    market: V2MarketState,
    cfg: Config,
    policy: Optional[PolicyConfig] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> pd.DataFrame:
    """Vectorised policy-consistent labels for the V2 base universe.

    For every signal date T and eligible stock, try to buy at T+1 close under
    the 3% price cap, then simulate the frozen single-stock exit policy (stop,
    trailing protection, 20-day time exit, market-extreme exit).  Exit signals
    are generated at close and filled at the next tradable close, respecting
    limit-down and suspension blocks.
    """

    panel = features.panel
    policy = policy or PolicyConfig()
    boards_all = panel.instrument_meta["board"].reindex(panel.symbols).fillna("unknown").to_numpy(dtype=object)
    base = v2_base_mask(features, cfg)
    close_adj = features.arr("close").astype(np.float64)
    raw_close = features.arr("raw_close").astype(np.float64)
    atr20 = features.arr("atr20").astype(np.float64)
    volume = panel.field("volume").astype(np.float64)
    valid = panel.valid
    limit_matrix = _build_limit_matrix(panel, cfg)
    n_dates = len(panel.dates)
    end_index = n_dates if end_index is None else min(int(end_index), n_dates)
    start_index = max(0, int(start_index))
    last_index = end_index - 1

    market_extreme = market.effective_state == "extreme"
    rows: Dict[str, List] = {
        "signal_index": [],
        "entry_index": [],
        "exit_signal_index": [],
        "exit_fill_index": [],
        "instrument": [],
        "entry_filled": [],
        "status": [],
        "exit_reason": [],
        "policy_return_net": [],
        "policy_win": [],
    }
    horizon = int(policy.max_hold_days) + int(policy.max_fill_wait_days) + 5
    offsets = np.arange(1, horizon + 1)

    for t in range(start_index, last_index):
        entry_idx = t + 1
        if entry_idx > last_index:
            break
        cand = np.flatnonzero(base[t])
        if cand.size == 0:
            continue
        e_raw = raw_close[entry_idx, cand]
        prev_raw = raw_close[t, cand]
        entry_ok = (
            valid[entry_idx, cand]
            & np.isfinite(e_raw)
            & np.isfinite(prev_raw)
            & (prev_raw > 0)
            & (e_raw <= prev_raw * 1.03)
        )
        if entry_ok.any():
            limit_pct = limit_matrix[entry_idx, cand]
            entry_ok &= ~(e_raw >= prev_raw * (1.0 + limit_pct) - 0.001)
        idx = cand[entry_ok]
        if idx.size == 0:
            continue
        n = idx.size
        entry_indices = np.full(n, entry_idx, dtype=np.int64)
        entry_adj = close_adj[entry_idx, idx]
        entry_raw = raw_close[entry_idx, idx]
        stop_pct = np.clip(
            policy.stop_atr_mult * atr20[t, idx] / np.where(close_adj[t, idx] > 0, close_adj[t, idx], np.nan),
            policy.stop_min,
            policy.stop_max,
        )
        stop_pct = np.where(np.isfinite(stop_pct), stop_pct, policy.stop_max)

        path = entry_idx + offsets[:, None]
        path_clipped = np.minimum(path, last_index)
        path_valid = path <= last_index
        adj_path = np.where(path_valid, close_adj[path_clipped, idx], np.nan)
        raw_path = np.where(path_valid, raw_close[path_clipped, idx], np.nan)
        vol_path = np.where(path_valid, volume[path_clipped, idx], 0.0)
        valid_path = np.where(path_valid, valid[path_clipped, idx], False)

        stop_hit = adj_path <= entry_adj[None, :] * (1.0 - stop_pct[None, :])
        run_max = np.fmax.accumulate(np.where(np.isfinite(adj_path), adj_path, -np.inf), axis=0)
        armed = run_max >= entry_adj[None, :] * (1.0 + policy.trailing_activate_mult * stop_pct[None, :])
        trailing_hit = armed & (adj_path <= run_max * (1.0 - policy.trailing_distance_mult * stop_pct[None, :]))
        extreme_path = market_extreme[path_clipped] & path_valid
        time_hit = np.zeros_like(stop_hit, dtype=bool)
        time_row = policy.max_hold_days - 1
        if 0 <= time_row < horizon:
            time_hit[time_row, :] = path_valid[time_row, :]
        triggers = stop_hit | trailing_hit | extreme_path | time_hit
        any_trigger = triggers.any(axis=0)
        if not any_trigger.any():
            continue
        first_trigger = np.argmax(triggers, axis=0)
        stop_any = stop_hit.any(axis=0)
        tr_any = trailing_hit.any(axis=0)
        ex_any = extreme_path.any(axis=0)
        tm_any = time_hit.any(axis=0)
        reason_code = np.zeros(n, dtype=np.int8)
        reason_code = np.where(stop_any, 1, reason_code)
        reason_code = np.where(~stop_any & tr_any, 2, reason_code)
        reason_code = np.where(~stop_any & ~tr_any & ex_any, 3, reason_code)
        reason_code = np.where(~stop_any & ~tr_any & ~ex_any & tm_any, 4, reason_code)
        reason_names = np.array(["open", "stop_loss", "trailing_stop", "market_extreme", "time_exit"], dtype=object)
        reason = reason_names[reason_code]

        # First tradable close strictly after the trigger signal.
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
            any_trigger,
            np.take_along_axis(path_clipped, first_trigger[None, :], axis=0)[0],
            -1,
        )
        exit_fill_index = np.where(
            filled,
            np.take_along_axis(path_clipped, np.maximum(fill_pos, 0)[None, :], axis=0)[0],
            -1,
        )

        buy_notional = policy.reference_notional
        buy_fee = _fees_buy(np.full(n, buy_notional, dtype=np.float64), cfg)
        sell_price = np.where(filled, raw_close[np.maximum(exit_fill_index, 0), idx], np.nan)
        qty = buy_notional / np.where(np.isfinite(entry_raw) & (entry_raw > 0), entry_raw, np.nan)
        sell_notional = qty * sell_price
        sell_fee = np.zeros(n, dtype=np.float64)
        if filled.any():
            date_ints = panel.date_ints[np.maximum(exit_fill_index, 0)]
            sell_fee = _fees_sell(np.nan_to_num(sell_notional, nan=0.0), date_ints, cfg)
        policy_return = np.where(
            filled & np.isfinite(sell_notional),
            (sell_notional - sell_fee - buy_notional - buy_fee) / buy_notional,
            np.nan,
        )
        status = np.where(filled, "closed", "open")
        win = np.where(filled & np.isfinite(policy_return), policy_return > 0, np.nan)

        keep = any_trigger
        rows["signal_index"].extend(np.full(int(keep.sum()), t, dtype=np.int64).tolist())
        rows["entry_index"].extend(entry_indices[keep].tolist())
        rows["exit_signal_index"].extend(exit_signal_index[keep].tolist())
        rows["exit_fill_index"].extend(exit_fill_index[keep].tolist())
        rows["instrument"].extend(np.asarray(panel.symbols, dtype=object)[idx][keep].tolist())
        rows["entry_filled"].extend(np.ones(int(keep.sum()), dtype=bool).tolist())
        rows["status"].extend(status[keep].tolist())
        rows["exit_reason"].extend(reason[keep].tolist())
        rows["policy_return_net"].extend(policy_return[keep].tolist())
        rows["policy_win"].extend(win[keep].tolist())

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["signal_date"] = panel.dates[frame["signal_index"].to_numpy()]
    frame["entry_date"] = panel.dates[frame["entry_index"].to_numpy()]
    frame["exit_signal_date"] = [
        panel.dates[int(v)] if v is not None and int(v) >= 0 else pd.NaT for v in frame["exit_signal_index"]
    ]
    frame["exit_fill_date"] = [
        panel.dates[int(v)] if v is not None and int(v) >= 0 else pd.NaT for v in frame["exit_fill_index"]
    ]
    frame["label_end_time"] = frame["exit_fill_date"]
    frame.loc[frame["label_end_time"].isna(), "label_end_time"] = panel.dates[-1]
    frame["label_status"] = np.where(frame["status"] == "closed", "matured", "open_not_matured")
    return frame[
        [
            "signal_date",
            "entry_date",
            "instrument",
            "entry_filled",
            "status",
            "exit_reason",
            "exit_signal_date",
            "exit_fill_date",
            "label_end_time",
            "label_status",
            "policy_return_net",
            "policy_win",
            "signal_index",
            "entry_index",
            "exit_signal_index",
            "exit_fill_index",
        ]
    ]
