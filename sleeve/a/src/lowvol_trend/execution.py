"""A-share close auction matching, limit rules, fees and order generation.

This module deliberately does not use Qlib's ``TopkDropoutStrategy`` order
generator.  Orders are frozen on the decision date and only the matching layer
sees the execution date's close/volume.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .models import Fill, Order, Position

MAIN_BOARDS = {"sh_main", "sz_main"}


def board_limit_pct(board: str, date_int: int) -> float:
    """Price-limit percentage for the board and date.

    The dates come from public exchange rules:

    * STAR Market: 20% since 2019-07-22.
    * ChiNext: 10% before 2020-08-24, 20% from 2020-08-24.
    * Beijing Stock Exchange: 30% since first trading (2021-11-15).
    * Shanghai/Shenzhen main boards: 10%.

    Historical ST status cannot be recovered from daily bars, so the caller can
    request a conservative ``min(limit, 5%)`` restriction as a stress test.
    """

    if board == "star":
        return 0.20
    if board == "chinext":
        return 0.20 if date_int >= 20200824 else 0.10
    if board == "bse":
        return 0.30
    if board in MAIN_BOARDS:
        return 0.10
    return 0.10


def effective_limit_pct(board: str, date_int: int, cfg: Config) -> float:
    limit = board_limit_pct(board, date_int)
    if cfg.execution.unknown_st_conservative and board in MAIN_BOARDS:
        limit = min(limit, cfg.execution.conservative_limit_pct)
    return limit


def compute_fees(
    side: str,
    notional: float,
    date_int: int,
    cfg: Config,
) -> Dict[str, float]:
    exec_cfg = cfg.execution
    commission = max(notional * exec_cfg.commission_rate, exec_cfg.min_commission)
    stamp = 0.0
    if side == "SELL":
        change = int(pd.Timestamp(exec_cfg.stamp_tax_change_date).strftime("%Y%m%d"))
        stamp_rate = (
            exec_cfg.stamp_tax_rate_before_2023_08_28
            if date_int < change
            else exec_cfg.stamp_tax_rate_after_2023_08_28
        )
        stamp = notional * stamp_rate
    transfer = notional * exec_cfg.transfer_fee_rate
    friction = notional * exec_cfg.friction_bps_per_side / 10_000.0
    total = commission + stamp + transfer + friction
    return {
        "commission": float(commission),
        "stamp_tax": float(stamp),
        "transfer_fee": float(transfer),
        "friction": float(friction),
        "total_fee": float(total),
    }


def max_affordable_quantity(cash: float, price: float, cfg: Config) -> int:
    """Maximum buy quantity in board lots affordable including estimated fees."""

    if cash <= 0 or price <= 0:
        return 0
    lot = int(cfg.execution.lot_size)
    # Conservative fee buffer (commission + friction + transfer).
    buffer_rate = (
        cfg.execution.commission_rate
        + cfg.execution.transfer_fee_rate
        + cfg.execution.friction_bps_per_side / 10_000.0
        + 0.0005
    )
    qty = int(math.floor(cash / (price * (1.0 + buffer_rate)) / lot)) * lot
    while qty > 0:
        fees = compute_fees("BUY", qty * price, 29990101, cfg)
        if qty * price + fees["total_fee"] <= cash + 1e-9:
            return qty
        qty -= lot
    return 0


def _bar_values(features, t: int, j: int) -> Tuple[float, float, float, float]:
    raw_close = features.arr("raw_close")
    volume = features.panel.field("volume")
    factor = features.panel.field("factor")
    close = float(raw_close[t, j]) if np.isfinite(raw_close[t, j]) else float("nan")
    raw_volume = float(volume[t, j] * factor[t, j] * 100.0) if np.isfinite(volume[t, j]) and np.isfinite(factor[t, j]) else 0.0
    return close, raw_volume, float(raw_close[t, j]), float(volume[t, j])


def generate_orders(
    features,
    cfg: Config,
    t: int,
    plan,
    positions: Mapping[str, Position],
    nav: float,
    cash: float,
    instrument_boards: Mapping[str, str],
) -> List[Order]:
    """Freeze orders for execution at ``t + 1``.

    Buy quantities are computed from the *decision-day* raw close with the
    3% maximum-buy-price premium, so no execution-day information is used.
    Sell quantities are share amounts decided from the decision-day close.
    """

    symbol_index = {s: j for j, s in enumerate(features.symbols)}
    raw_close = features.arr("raw_close")
    adtv20 = features.arr("adtv20_shares")
    lot = int(cfg.execution.lot_size)
    orders: List[Order] = []
    order_id = 0

    # Sells first in the order book logically, but the matcher still spends
    # start-of-day cash for buys.  This ordering only affects order ids.
    for symbol, pos in positions.items():
        if pos.shares <= 0:
            continue
        j = symbol_index.get(symbol)
        if j is None or not np.isfinite(raw_close[t, j]) or raw_close[t, j] <= 0:
            continue
        decision_price = float(raw_close[t, j])
        current_value = pos.shares * decision_price
        target_w = float(plan.target_weights.get(symbol, 0.0))
        if symbol in plan.exit_reasons:
            target_w = 0.0
        target_value = target_w * nav
        delta_value = current_value - target_value
        if delta_value <= max(0.001 * nav, 1e-9):
            continue
        if target_w <= 1e-9:
            qty = pos.shares
        else:
            desired_shares = int(math.floor(target_value / decision_price / lot)) * lot
            qty = pos.shares - desired_shares
            qty = int(math.floor(qty / lot) * lot)
            if qty <= 0:
                continue
        qty = min(qty, pos.shares)
        planned_cap = int(math.floor(cfg.execution.participation_rate * float(adtv20[t, j]) / lot) * lot)
        if planned_cap <= 0:
            continue
        qty = min(qty, planned_cap)
        if qty <= 0:
            continue
        reason = plan.exit_reasons.get(symbol, "trim_to_target")
        orders.append(
            Order(
                order_id=order_id,
                symbol=symbol,
                side="SELL",
                quantity=int(qty),
                decision_index=t,
                decision_date=str(features.dates[t].date()),
                execute_index=t + 1,
                execute_date=str(features.dates[t + 1].date()),
                reason=reason,
                decision_price_raw=decision_price,
                target_weight=target_w,
                cash_budget_basis=cash,
            )
        )
        order_id += 1

    # New buys only.  Existing positions are never averaged down.
    new_symbols = [s for s in plan.new_symbols if s not in positions]
    buy_budget = max(0.0, cash)
    desired_total = 0.0
    desired: Dict[str, float] = {}
    for symbol in new_symbols:
        desired[symbol] = max(0.0, float(plan.target_weights.get(symbol, 0.0)) * nav)
        desired_total += desired[symbol]
    if desired_total > 0:
        scale = min(1.0, buy_budget / desired_total)
    else:
        scale = 0.0
    for symbol in new_symbols:
        j = symbol_index.get(symbol)
        if j is None or not np.isfinite(raw_close[t, j]) or raw_close[t, j] <= 0:
            continue
        decision_price = float(raw_close[t, j])
        max_buy_price = decision_price * (1.0 + cfg.execution.buy_premium)
        budget = desired[symbol] * scale
        if budget <= 1e-6:
            continue
        qty = max_affordable_quantity(budget, max_buy_price, cfg)
        if qty <= 0:
            continue
        planned_cap = int(math.floor(cfg.execution.participation_rate * float(adtv20[t, j]) / lot) * lot)
        qty = min(qty, planned_cap)
        if qty <= 0:
            continue
        # Keep the actual buy budget bookkeeping conservative: reserve the
        # maximum price plus a fee buffer.
        buy_budget -= qty * max_buy_price
        orders.append(
            Order(
                order_id=order_id,
                symbol=symbol,
                side="BUY",
                quantity=int(qty),
                decision_index=t,
                decision_date=str(features.dates[t].date()),
                execute_index=t + 1,
                execute_date=str(features.dates[t + 1].date()),
                reason="new_entry",
                decision_price_raw=decision_price,
                max_buy_price=float(max_buy_price),
                target_weight=float(plan.target_weights.get(symbol, 0.0)),
                cash_budget_basis=float(budget),
            )
        )
        order_id += 1
    return orders


def match_orders(
    orders: Sequence[Order],
    features,
    cfg: Config,
    execute_index: int,
    positions: MutableMapping[str, Position],
    cash: float,
    instrument_boards: Mapping[str, str],
) -> Tuple[float, List[Fill]]:
    """Match frozen orders against the execution day's close/volume.

    Buys are processed before sells and can only use the cash available at the
    start of the day.  Sell proceeds are added to cash afterwards but are not
    usable for the same close's buys, matching the design's conservative
    settlement convention.
    """

    raw_close = features.arr("raw_close")
    volume = features.panel.field("volume")
    factor = features.panel.field("factor")
    date_int = int(features.panel.date_ints[execute_index])
    lot = int(cfg.execution.lot_size)
    fills: List[Fill] = []

    def _planned_participation_qty(self, order: Order) -> int:
        j = features.symbols.index(order.symbol)
        adv = float(features.arr("adtv20_shares")[order.decision_index, j])
        if not np.isfinite(adv) or adv <= 0:
            return 0
        return int(math.floor(cfg.execution.participation_rate * adv / lot) * lot)

    # ---------------- buys: start-of-day cash only ----------------
    for order in [o for o in orders if o.side == "BUY"]:
        j = features.symbols.index(order.symbol)
        close = float(raw_close[execute_index, j]) if np.isfinite(raw_close[execute_index, j]) else float("nan")
        vol = float(volume[execute_index, j]) if np.isfinite(volume[execute_index, j]) else 0.0
        fac = float(factor[execute_index, j]) if np.isfinite(factor[execute_index, j]) else 0.0
        if not np.isfinite(close) or close <= 0 or vol <= 0 or fac <= 0:
            order.status = "REJECTED"
            order.reject_reason = "suspended_or_missing_bar"
            continue
        limit_pct = effective_limit_pct(instrument_boards.get(order.symbol, "unknown"), date_int, cfg)
        prev_close = order.decision_price_raw
        if np.isfinite(prev_close) and prev_close > 0 and close >= prev_close * (1.0 + limit_pct) - cfg.execution.price_tolerance:
            order.status = "REJECTED"
            order.reject_reason = "limit_up_locked_or_unknown_st_conservative"
            continue
        if order.max_buy_price is not None and close > order.max_buy_price + cfg.execution.price_tolerance:
            order.status = "REJECTED"
            order.reject_reason = "close_above_frozen_max_buy_price"
            continue
        raw_volume_shares = vol * fac * 100.0
        fill_cap = int(math.floor(cfg.execution.participation_rate * raw_volume_shares / lot) * lot)
        qty = min(int(order.quantity), fill_cap)
        qty = min(qty, max_affordable_quantity(cash, close, cfg))
        if qty <= 0:
            order.status = "REJECTED"
            order.reject_reason = "no_cash_or_no_volume_capacity"
            continue
        notional = qty * close
        fees = compute_fees("BUY", notional, date_int, cfg)
        total = notional + fees["total_fee"]
        if total > cash + 1e-6:
            qty = max_affordable_quantity(cash, close, cfg)
            if qty <= 0:
                order.status = "REJECTED"
                order.reject_reason = "insufficient_cash"
                continue
            notional = qty * close
            fees = compute_fees("BUY", notional, date_int, cfg)
            total = notional + fees["total_fee"]
        cash -= total
        order.filled_quantity = qty
        order.fill_price = close
        order.status = "FILLED" if qty >= order.quantity else "PARTIAL"
        if qty < order.quantity:
            order.reject_reason = "volume_participation_cap"
        pos = positions.get(order.symbol)
        if pos is None:
            positions[order.symbol] = Position(
                symbol=order.symbol,
                shares=qty,
                entry_index=execute_index,
                entry_date=order.execute_date,
                entry_raw_price=close,
                entry_adj_price=float(features.arr("close")[execute_index, j]),
                entry_atr=float(features.arr("atr20")[order.decision_index, j]),
                stop_pct=float(
                    np.clip(
                        cfg.strategy.stop_atr_mult
                        * float(features.arr("atr20")[order.decision_index, j])
                        / float(features.arr("close")[order.decision_index, j]),
                        cfg.strategy.stop_min,
                        cfg.strategy.stop_max,
                    )
                ),
                cost_basis_total=total,
                buy_fees_paid=fees["total_fee"],
                high_adj=float(features.arr("close")[execute_index, j]),
            )
        else:
            pos.shares += qty
            pos.cost_basis_total += total
            pos.buy_fees_paid += fees["total_fee"]
        fills.append(
            Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side="BUY",
                quantity=qty,
                price=close,
                date=order.execute_date,
                trade_index=execute_index,
                notional=notional,
                commission=fees["commission"],
                stamp_tax=fees["stamp_tax"],
                transfer_fee=fees["transfer_fee"],
                friction=fees["friction"],
                total_fee=fees["total_fee"],
                reason=order.reason,
            )
        )

    # ---------------- sells: proceeds added after buys ----------------
    for order in [o for o in orders if o.side == "SELL"]:
        j = features.symbols.index(order.symbol)
        pos = positions.get(order.symbol)
        if pos is None or pos.shares <= 0:
            order.status = "REJECTED"
            order.reject_reason = "no_position"
            continue
        close = float(raw_close[execute_index, j]) if np.isfinite(raw_close[execute_index, j]) else float("nan")
        vol = float(volume[execute_index, j]) if np.isfinite(volume[execute_index, j]) else 0.0
        fac = float(factor[execute_index, j]) if np.isfinite(factor[execute_index, j]) else 0.0
        if not np.isfinite(close) or close <= 0 or vol <= 0 or fac <= 0:
            order.status = "REJECTED"
            order.reject_reason = "suspended_or_missing_bar"
            continue
        limit_pct = effective_limit_pct(instrument_boards.get(order.symbol, "unknown"), date_int, cfg)
        prev_close = order.decision_price_raw
        if np.isfinite(prev_close) and prev_close > 0 and close <= prev_close * (1.0 - limit_pct) + cfg.execution.price_tolerance:
            order.status = "REJECTED"
            order.reject_reason = "limit_down_locked_or_unknown_st_conservative"
            continue
        raw_volume_shares = vol * fac * 100.0
        fill_cap = int(math.floor(cfg.execution.participation_rate * raw_volume_shares / lot) * lot)
        qty = min(int(order.quantity), int(pos.shares))
        if qty < pos.shares:
            qty = min(qty, fill_cap)
            qty = int(math.floor(qty / lot) * lot)
        else:
            qty = min(qty, max(fill_cap, lot)) if fill_cap > 0 else 0
        if qty <= 0:
            order.status = "REJECTED"
            order.reject_reason = "no_volume_capacity"
            continue
        notional = qty * close
        fees = compute_fees("SELL", notional, date_int, cfg)
        cash += notional - fees["total_fee"]
        pos.shares -= qty
        proportion = qty / (qty + pos.shares) if (qty + pos.shares) > 0 else 1.0
        pos.cost_basis_total *= 1.0 - proportion
        if pos.cost_basis_total < 1e-9:
            pos.cost_basis_total = 0.0
        order.filled_quantity = qty
        order.fill_price = close
        order.status = "FILLED" if qty >= order.quantity else "PARTIAL"
        fills.append(
            Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side="SELL",
                quantity=qty,
                price=close,
                date=order.execute_date,
                trade_index=execute_index,
                notional=notional,
                commission=fees["commission"],
                stamp_tax=fees["stamp_tax"],
                transfer_fee=fees["transfer_fee"],
                friction=fees["friction"],
                total_fee=fees["total_fee"],
                reason=order.reason,
            )
        )
    return cash, fills
