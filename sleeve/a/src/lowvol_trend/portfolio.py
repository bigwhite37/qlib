"""Target portfolio construction and exit rules (design sections 5-6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureStore
from .market import MarketState
from .models import Position


@dataclass
class TargetPlan:
    target_weights: Dict[str, float]
    exit_reasons: Dict[str, str]
    new_symbols: List[str]
    gross_cap: float
    drawdown_cap: float
    preliminary_vol: float
    scaled_vol: float
    vol_scale: float
    cash_weight_budget: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def drawdown_cap(nav: float, peak_nav: float, tiers: Sequence[Tuple[float, float]], default: float = 1.0) -> float:
    if peak_nav <= 0 or nav <= 0:
        return default
    drawdown = nav / peak_nav - 1.0
    cap = default
    # Tiers are written as (threshold, cap); the highest crossed threshold
    # applies and therefore wins.
    for threshold, tier_cap in sorted(tiers):
        if -drawdown >= threshold - 1e-12:
            cap = min(cap, tier_cap)
    return cap


def _pairwise_corr(a: np.ndarray, b: np.ndarray, min_obs: int = 20) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < min_obs:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _fallback_vol(weights: Mapping[str, float], vol60_row: np.ndarray, symbol_index: Mapping[str, int]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    variance = 0.0
    for symbol, weight in weights.items():
        j = symbol_index[symbol]
        v = vol60_row[j]
        if np.isfinite(v):
            variance += (weight * float(v)) ** 2
    return float(np.sqrt(variance) * np.sqrt(252.0))


def estimate_annual_vol(
    ret_window: np.ndarray,
    weights: Mapping[str, float],
    symbol_index: Mapping[str, int],
    vol60_row: np.ndarray,
    cfg: Config,
) -> float:
    """Estimate annualised portfolio vol from the last 60 daily returns.

    Qlib's ``ShrinkCovEstimator`` is used with fractional returns, therefore
    ``is_price=False`` and ``scale_return=False`` are passed explicitly.  If
    the estimator is unavailable or there are too few observations we fall back
    to the diagonal (vol60) estimate.
    """

    active = {s: w for s, w in weights.items() if abs(w) > 1e-12 and s in symbol_index}
    if not active:
        return 0.0
    if ret_window.shape[0] < 20:
        return _fallback_vol(active, vol60_row, symbol_index)
    try:
        from qlib.model.riskmodel.shrink import ShrinkCovEstimator  # noqa: WPS433

        columns = list(active.keys())
        idx = [symbol_index[s] for s in columns]
        frame = pd.DataFrame(ret_window[:, idx], columns=columns)
        estimator = ShrinkCovEstimator(
            alpha=cfg.strategy.shrink_alpha,
            target=cfg.strategy.shrink_target,
            nan_option="fill",
            assume_centered=False,
            scale_return=False,
        )
        cov = estimator.predict(frame, is_price=False)
        w = np.asarray([active[s] for s in columns], dtype=np.float64)
        sigma_daily = float(np.sqrt(max(0.0, float(w @ cov.to_numpy(dtype=np.float64) @ w))))
        if not np.isfinite(sigma_daily) or sigma_daily <= 0:
            return _fallback_vol(active, vol60_row, symbol_index)
        return sigma_daily * np.sqrt(252.0)
    except Exception:
        return _fallback_vol(active, vol60_row, symbol_index)


def average_pairwise_corr(
    features: FeatureStore, t: int, j: int, others: Sequence[int], window: int
) -> float:
    """Mean correlation of one candidate with a set of already-held names."""

    if not others:
        return 0.0
    ret = features.arr("ret1")
    start = max(0, t - window + 1)
    candidate = ret[start : t + 1, j]
    values = []
    for k in others:
        corr = _pairwise_corr(candidate, ret[start : t + 1, k])
        if np.isfinite(corr):
            values.append(corr)
    if not values:
        return 0.0
    return float(np.mean(values))


def correlation_penalty(avg_corr: float, strength: float, base: float, floor: float) -> float:
    """Size multiplier applied to a candidate that duplicates existing exposure."""

    excess = max(0.0, float(avg_corr) - float(base))
    return float(max(floor, min(1.0, 1.0 - strength * excess)))


def update_position_observations(
    features: FeatureStore,
    t: int,
    positions: Mapping[str, Position],
    cfg: Config,
) -> None:
    """Update high-water marks and rank-exit counters using date ``t`` data."""

    close = features.arr("close")
    ma20 = features.arr("ma20")
    top_half = features.arr("top_half")
    symbol_index = {s: j for j, s in enumerate(features.symbols)}
    for symbol, pos in positions.items():
        if pos.shares <= 0:
            continue
        j = symbol_index.get(symbol)
        if j is None:
            continue
        price = close[t, j]
        if np.isfinite(price):
            pos.high_adj = max(pos.high_adj, float(price))
            if np.isfinite(ma20[t, j]) and float(price) > float(ma20[t, j]):
                pos.seen_above_ma20 = True
            if float(price) >= pos.entry_adj_price + cfg.strategy.trailing_activate_atr * pos.entry_atr:
                pos.trailing_armed = True
            if bool(top_half[t, j]):
                pos.below_top_half_days = 0
            else:
                pos.below_top_half_days += 1


def evaluate_exits(
    features: FeatureStore,
    t: int,
    positions: Mapping[str, Position],
    market: MarketState,
    cfg: Config,
) -> Dict[str, str]:
    """Return ``{symbol: reason}`` for positions that must exit at the next close."""

    exits: Dict[str, str] = {}
    close = features.arr("close")
    ma60 = features.arr("ma60")
    two_below = features.arr("two_below_ma20")
    symbol_index = {s: j for j, s in enumerate(features.symbols)}
    market_extreme = (
        cfg.strategy.exit_market_extreme
        and cfg.strategy.use_market_timing
        and float(market.effective_cap[t]) <= 1e-12
    )
    for symbol, pos in positions.items():
        if pos.shares <= 0:
            continue
        if market_extreme:
            exits[symbol] = "market_extreme"
            continue
        j = symbol_index.get(symbol)
        if j is None:
            continue
        price = close[t, j]
        if not np.isfinite(price):
            continue
        price = float(price)
        held_days = t - pos.entry_index
        if cfg.strategy.profit_target_pct > 0 and held_days >= cfg.strategy.profit_target_min_hold:
            target_pct = float(cfg.strategy.profit_target_pct)
            if (
                cfg.strategy.profit_target_atr_mult > 0
                and pos.entry_atr > 0
                and pos.entry_adj_price > 0
            ):
                raw_pct = (
                    float(cfg.strategy.profit_target_atr_mult)
                    * float(pos.entry_atr)
                    / float(pos.entry_adj_price)
                )
                target_pct = float(
                    np.clip(
                        raw_pct,
                        cfg.strategy.profit_target_atr_floor,
                        cfg.strategy.profit_target_atr_cap,
                    )
                )
            if price >= pos.entry_adj_price * (1.0 + target_pct):
                exits[symbol] = "profit_target"
                continue
        if price <= pos.entry_stop_adj_price:
            exits[symbol] = "stop_loss"
            continue
        if cfg.strategy.exit_trend_ma60 and np.isfinite(ma60[t, j]) and price < float(ma60[t, j]):
            exits[symbol] = "trend_fail_ma60"
            continue
        if cfg.strategy.exit_trend_ma20 and bool(two_below[t, j]) and pos.seen_above_ma20:
            exits[symbol] = "trend_fail_ma20"
            continue
        if pos.trailing_armed and price <= pos.high_adj - cfg.strategy.trailing_atr_mult * pos.entry_atr:
            exits[symbol] = "trailing_stop"
            continue
        if held_days >= cfg.strategy.max_hold_days:
            exits[symbol] = "time_exit"
            continue
        if (
            cfg.strategy.exit_rank
            and held_days >= cfg.strategy.rank_exit_min_hold
            and pos.below_top_half_days >= cfg.strategy.rank_exit_confirm_days
        ):
            exits[symbol] = "rank_exit"
            continue
    return exits


def _planned_stop_pct(features: FeatureStore, t: int, j: int, cfg: Config) -> float:
    atr = features.arr("atr20")[t, j]
    price = features.arr("close")[t, j]
    if not np.isfinite(atr) or not np.isfinite(price) or price <= 0:
        return cfg.strategy.stop_max
    raw = cfg.strategy.stop_atr_mult * float(atr) / float(price)
    return float(np.clip(raw, cfg.strategy.stop_min, cfg.strategy.stop_max))


def select_new_candidates(
    features: FeatureStore,
    t: int,
    held: Iterable[str],
    exits: Mapping[str, str],
    cfg: Config,
) -> Tuple[List[Tuple[str, int, float, float]], Dict[str, str]]:
    """Select new names with correlation filtering.

    Returns ``(selected, replacement_exits)``.  When a candidate is highly
    correlated with an existing holding but has a higher V0 score, the existing
    holding is marked for replacement, which implements the design rule
    "prefer the higher-score stock" without ever averaging down.
    """

    entry_signal = features.arr("entry_signal")
    entry_score = features.arr("entry_score")
    score = features.arr("score")
    ret = features.arr("ret1")
    symbol_index = {s: j for j, s in enumerate(features.symbols)}

    day_gate = features.arrays.get("entry_day_ok")
    if day_gate is not None and not bool(day_gate[t]):
        # Patience: today's best candidate is not unusually good relative to the
        # trailing distribution of daily best candidates, so wait.
        return [], {}
    order = np.flatnonzero(entry_signal[t])
    if order.size == 0:
        return [], {}
    order = order[np.argsort(-np.nan_to_num(entry_score[t, order], nan=-np.inf), kind="stable")]
    if cfg.strategy.head_rerank_k > 1 and order.size > 1:
        # Second ranking stage over the head of the list only.  The breadth
        # experiment showed that everything below the first one or two names is
        # noise, so a re-rank is only meaningful at the top.
        head = order[: cfg.strategy.head_rerank_k]
        key = features.arr(cfg.strategy.head_rerank_key)[t, head]
        sortable = np.nan_to_num(key, nan=np.inf) * float(cfg.strategy.head_rerank_sign)
        order = np.concatenate([head[np.argsort(sortable, kind="stable")], order[cfg.strategy.head_rerank_k :]])
        del head, key, sortable
    held_all = set(held)
    held_set = {s for s in held if s not in exits}
    selected: List[Tuple[str, int, float, float]] = []
    selected_syms: List[str] = []
    replacement_exits: Dict[str, str] = {}
    window_start = max(0, t - cfg.strategy.corr_window + 1)
    for j in order:
        if len(selected) >= cfg.strategy.max_new_per_day:
            break
        score_value = entry_score[t, j]
        if not np.isfinite(score_value):
            continue
        min_score = float(getattr(cfg.strategy, "entry_min_score", 0.0) or 0.0)
        if min_score > 0.0 and float(score_value) < min_score:
            # Patience: wait for a better candidate rather than filling the slot.
            continue
        symbol = features.symbols[j]
        if symbol in held_set or symbol in exits or symbol in selected_syms:
            continue
        # The position budget counts the names that will still be held after
        # today's sells: positions already queued for exit (policy exits or
        # correlation replacements) free their slot for the same close, and the
        # cash they release is not needed because the buy budget is cash-only.
        # This check must come before any other acceptance path, otherwise the
        # position budget is silently ignored.
        pending_exit = set(exits) | set(replacement_exits)
        if len(selected_syms) + len(held_all - pending_exit) >= cfg.strategy.max_positions:
            break
        candidate_ret = ret[window_start : t + 1, j]
        correlated_with: Optional[str] = None
        if not cfg.strategy.use_corr_filter:
            selected.append((symbol, int(j), float(score_value), _planned_stop_pct(features, t, j, cfg)))
            selected_syms.append(symbol)
            continue
        for other in sorted(held_set) + selected_syms:
            k = symbol_index[other]
            corr = _pairwise_corr(candidate_ret, ret[window_start : t + 1, k])
            if np.isfinite(corr) and corr > cfg.strategy.corr_threshold:
                correlated_with = other
                break
        if correlated_with is not None:
            existing_is_held = correlated_with in held_set
            existing_score = float(score[t, symbol_index[correlated_with]]) if existing_is_held else -np.inf
            if (
                cfg.strategy.correlation_replace
                and existing_is_held
                and float(score_value) > existing_score
                and correlated_with not in replacement_exits
            ):
                replacement_exits[correlated_with] = "corr_replace"
                held_set.discard(correlated_with)
            else:
                continue
        stop_pct = _planned_stop_pct(features, t, j, cfg)
        selected.append((symbol, int(j), float(score_value), stop_pct))
        selected_syms.append(symbol)
    return selected, replacement_exits


def construct_targets(
    features: FeatureStore,
    cfg: Config,
    t: int,
    current_weights: Mapping[str, float],
    held_symbols: Sequence[str],
    held_caps: Mapping[str, float],
    nav: float,
    peak_nav: float,
    cash: float,
    market: MarketState,
    exits: Mapping[str, str],
    scale_factors: "Mapping[str, float] | None" = None,
) -> TargetPlan:
    """Build target weights for execution at ``t + 1``.

    ``current_weights`` are the decision-date close weights of the actual
    account position (the caller derives them from Qlib's ``Position``).  The
    rules implemented here are intentionally conservative:

    * existing winners are never averaged down / added to;
    * all risk reductions (vol target, drawdown tier, market downgrade) are
      implemented as proportional trims of the current portfolio;
    * new positions are sized with inverse-vol weights and a per-trade stop
      risk cap;
    * buys are budgeted from cash only, never from same-close sell proceeds.
    """

    symbol_index = {s: j for j, s in enumerate(features.symbols)}
    vol60 = features.arr("vol60")
    ret = features.arr("ret1")

    market_cap = float(market.effective_cap[t]) if cfg.strategy.use_market_timing else cfg.strategy.max_gross
    dd_cap = (
        drawdown_cap(nav, peak_nav, cfg.strategy.drawdown_tiers)
        if cfg.strategy.use_drawdown_control
        else 1.0
    )
    gross_cap = float(min(market_cap, dd_cap, cfg.strategy.max_gross))
    entry_allowed = bool(market.entry_allowed[t]) if cfg.strategy.use_market_timing else True
    entry_allowed = entry_allowed and gross_cap > 0

    keep_weights: Dict[str, float] = {}
    for symbol, weight in current_weights.items():
        if symbol in exits:
            continue
        cap = float(held_caps.get(symbol, cfg.strategy.max_weight))
        weight_value = max(0.0, min(float(weight), cap))
        if scale_factors:
            weight_value *= float(scale_factors.get(symbol, 1.0))
        keep_weights[symbol] = weight_value
    # Positions that have decayed below the minimum weight lock a slot without
    # meaningfully contributing exposure; close them so the 12-position budget
    # remains available for qualified candidates.
    small_positions = {
        symbol: weight
        for symbol, weight in keep_weights.items()
        if 0.0 < weight < cfg.strategy.min_position_weight
    }
    if small_positions:
        exits = {**exits, **{symbol: "small_position_cleanup" for symbol in small_positions}}
        for symbol in small_positions:
            keep_weights.pop(symbol, None)
    if cfg.strategy.rebalance_inverse_vol and keep_weights:
        # Pull every held weight toward the inverse-vol target at the same total.
        vol_row = vol60[t]
        inv: Dict[str, float] = {}
        for symbol, weight in keep_weights.items():
            j = symbol_index.get(symbol)
            v = float(vol_row[j]) if j is not None and np.isfinite(vol_row[j]) else 0.03
            inv[symbol] = 1.0 / max(v, 1e-6)
        inv_total = sum(inv.values())
        total_now = sum(keep_weights.values())
        if inv_total > 0 and total_now > 0:
            strength = float(np.clip(cfg.strategy.rebalance_strength, 0.0, 1.0))
            for symbol, weight in list(keep_weights.items()):
                cap = float(held_caps.get(symbol, cfg.strategy.max_weight))
                target_weight = total_now * inv[symbol] / inv_total
                blended = (1.0 - strength) * weight + strength * target_weight
                keep_weights[symbol] = float(min(max(blended, 0.0), cap))
    target: Dict[str, float] = dict(keep_weights)

    # Trim for the actual current portfolio volatility even before considering
    # new names.  This makes market/drawdown risk reductions and the 10% vol
    # target apply to the whole account, not only to new buys.
    window_start = max(0, t - cfg.strategy.vol_estimate_window + 1)
    ret_window = ret[window_start : t + 1]
    vol60_row = vol60[t]
    proportional = cfg.strategy.risk_trim_mode != "close_weakest"
    pre_vol = estimate_annual_vol(ret_window, target, symbol_index, vol60_row, cfg)
    vol_scale = 1.0
    if proportional and cfg.strategy.use_vol_target and cfg.strategy.vol_target > 0:
        band = max(0.5, min(1.0, float(cfg.strategy.vol_rebalance_band)))
        edge = cfg.strategy.vol_target / band
        if pre_vol > edge and pre_vol > 0:
            vol_scale = edge / pre_vol
            target = {s: w * vol_scale for s, w in target.items()}
    total = sum(target.values())

    # Gross exposure cap (market state + drawdown tier), also banded so that a
    # market-state wobble does not force an immediate partial liquidation.
    if gross_cap <= 0:
        if cfg.strategy.extreme_liquidates:
            target = {}
        total = sum(target.values())
    elif proportional and total > gross_cap > 0:
        gross_band = max(0.5, min(1.0, float(cfg.strategy.gross_rebalance_band)))
        if total > gross_cap / gross_band:
            factor = (gross_cap / gross_band) / total
            target = {s: w * factor for s, w in target.items()}
            total = sum(target.values())

    cash_weight_budget = max(0.0, cash / nav - 0.002) if nav > 0 else 0.0
    remaining_budget = max(0.0, gross_cap - total)
    new_symbols: List[str] = []
    selected: List[Tuple[str, int, float, float]] = []
    if entry_allowed and remaining_budget > 0 and cash_weight_budget > 0:
        selected, replacement_exits = select_new_candidates(features, t, held_symbols, exits, cfg)
        if replacement_exits:
            exits = {**exits, **replacement_exits}
            # Remove replaced positions from the kept sleeve.
            for symbol in replacement_exits:
                target.pop(symbol, None)
            total = sum(target.values())
            remaining_budget = max(0.0, gross_cap - total)
        allocation_budget = min(remaining_budget, cash_weight_budget)
        if selected:
            inv_vols: List[float] = []
            caps: List[float] = []
            held_indices = [
                symbol_index[s] for s in held_symbols if s in symbol_index and s not in exits
            ]
            for symbol, j, score, stop_pct in selected:
                v = float(vol60[t, j])
                if not np.isfinite(v) or v <= 1e-8:
                    v = 0.03
                inv_vols.append(1.0 / v)
                risk_cap = cfg.strategy.risk_per_trade / max(stop_pct, 1e-6)
                cap = min(cfg.strategy.max_weight, risk_cap)
                if cfg.strategy.corr_sizing:
                    # The correlation *filter* was rejected because it threw the
                    # candidate away; sizing keeps the opportunity and only trims
                    # the duplicate exposure it would add.
                    avg_corr = average_pairwise_corr(
                        features, t, int(j), held_indices, cfg.strategy.corr_window
                    )
                    cap *= correlation_penalty(
                        avg_corr,
                        cfg.strategy.corr_sizing_strength,
                        cfg.strategy.corr_sizing_base,
                        cfg.strategy.corr_sizing_floor,
                    )
                caps.append(cap)
            inv_sum = sum(inv_vols)
            weights = [0.0] * len(selected)
            if inv_sum > 0:
                # Allocate inversely to vol, then cap; redistribute leftover to
                # uncapped candidates for a few passes.
                for _ in range(4):
                    open_slots = [i for i, w in enumerate(weights) if w < caps[i] - 1e-12]
                    if not open_slots:
                        break
                    allocated = sum(weights)
                    budget_left = allocation_budget - allocated
                    if budget_left <= 1e-12:
                        break
                    local_sum = sum(inv_vols[i] for i in open_slots)
                    if local_sum <= 0:
                        break
                    changed = False
                    for i in open_slots:
                        add = budget_left * inv_vols[i] / local_sum
                        new_w = min(caps[i], weights[i] + add)
                        if new_w > weights[i] + 1e-12:
                            changed = True
                        weights[i] = new_w
                    if not changed:
                        break
            for (symbol, j, score, stop_pct), weight in zip(selected, weights):
                if weight <= 1e-6:
                    continue
                target[symbol] = target.get(symbol, 0.0) + weight
                new_symbols.append(symbol)
            total = sum(target.values())

    # ------------------------------------------------------------------
    # Risk reduction on the existing sleeve.
    # ------------------------------------------------------------------
    scaled_vol = estimate_annual_vol(ret_window, target, symbol_index, vol60_row, cfg)
    if cfg.strategy.risk_trim_mode == "close_weakest" and target:
        entry_score = features.arr("entry_score")
        edge = 1e9
        if cfg.strategy.use_vol_target and cfg.strategy.vol_target > 0:
            band = max(0.5, min(1.0, float(cfg.strategy.vol_rebalance_band)))
            edge = cfg.strategy.vol_target / band
        gross_limit = gross_cap if gross_cap > 0 else 0.0
        dropped: Dict[str, float] = {}
        while len(target) > 1:
            total_now = sum(target.values())
            vol_now = estimate_annual_vol(ret_window, target, symbol_index, vol60_row, cfg)
            over_vol = vol_now > edge
            over_gross = total_now > gross_limit > 0
            if not over_vol and not over_gross:
                break
            # drop the lowest-scoring remaining position as a whole
            worst = None
            worst_score = np.inf
            for symbol in target:
                j = symbol_index.get(symbol)
                score = float(entry_score[t, j]) if j is not None and np.isfinite(entry_score[t, j]) else -np.inf
                weight = float(target[symbol])
                if weight < cfg.strategy.min_trim_weight and worst is None:
                    worst, worst_score = symbol, score
                    continue
                if score < worst_score:
                    worst, worst_score = symbol, score
            if worst is None:
                break
            dropped[worst] = "risk_trim"
            target.pop(worst, None)
        if dropped:
            exits = {**exits, **dropped}
        scaled_vol = estimate_annual_vol(ret_window, target, symbol_index, vol60_row, cfg)
        total = sum(target.values())
    if proportional:
        if cfg.strategy.use_vol_target and cfg.strategy.vol_target > 0:
            band = max(0.5, min(1.0, float(cfg.strategy.vol_rebalance_band)))
            edge = cfg.strategy.vol_target / band
            if scaled_vol > edge and scaled_vol > 0:
                factor = edge / scaled_vol
                target = {s: w * factor for s, w in target.items()}
                total = sum(target.values())
        if total > gross_cap > 0:
            gross_band = max(0.5, min(1.0, float(cfg.strategy.gross_rebalance_band)))
            if total > gross_cap / gross_band:
                factor = (gross_cap / gross_band) / total
                target = {s: w * factor for s, w in target.items()}
                total = sum(target.values())

    diagnostics = {
        "n_selected": len(selected),
        "remaining_budget_initial": remaining_budget,
        "cash_weight_budget": cash_weight_budget,
        "gross_cap": gross_cap,
        "market_cap": market_cap,
        "drawdown_cap": dd_cap,
        "pre_vol": pre_vol,
        "scaled_vol": scaled_vol,
    }
    return TargetPlan(
        target_weights=target,
        exit_reasons=dict(exits),
        new_symbols=new_symbols,
        gross_cap=gross_cap,
        drawdown_cap=dd_cap,
        preliminary_vol=pre_vol,
        scaled_vol=scaled_vol,
        vol_scale=vol_scale,
        cash_weight_budget=cash_weight_budget,
        diagnostics=diagnostics,
    )
