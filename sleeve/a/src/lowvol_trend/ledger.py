"""Execution ledger built from Qlib executor results.

Qlib keeps the authoritative account (cash, adjusted amount, valuation) and
saves it in the standard ``positions_normal.pkl`` report.  This ledger adds the
strategy-specific observations that Qlib does not keep: raw share amounts,
entry prices/ATR, trailing highs, exit reasons and complete round trips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureStore
from .models import Fill, Order as AuditOrder, Position, RoundTrip


@dataclass
class OrderRecord:
    order_ref: Any
    symbol: str
    side: str
    reason: str
    decision_index: int
    decision_date: str
    execute_index: int
    execute_date: str
    planned_amount_adj: float
    planned_raw_shares: float
    max_raw_buy_price: Optional[float]
    min_raw_sell_price: Optional[float]
    status: str = "PENDING"
    reject_reason: str = ""
    filled_amount_adj: float = 0.0
    filled_raw_shares: float = 0.0
    fill_raw_price: Optional[float] = None
    trade_value: float = 0.0
    total_fee: float = 0.0


class ExecutionLedger:
    def __init__(self, features: FeatureStore, cfg: Config):
        self.features = features
        self.cfg = cfg
        self.positions: Dict[str, Position] = {}
        self.orders: List[OrderRecord] = []
        self.fills: List[Fill] = []
        self.round_trips: List[RoundTrip] = []
        self._order_map: Dict[int, OrderRecord] = {}
        self._last_execute_index: Optional[int] = None

    # ------------------------------------------------------------------
    # order bookkeeping
    # ------------------------------------------------------------------
    def record_orders(self, orders: Sequence[Any], decision_index: int) -> None:
        executing_index = decision_index + 1
        for order in orders:
            factor = float(self.features.arr("factor")[decision_index, self.features.symbol_to_index[order.stock_id]])
            planned_adj = float(getattr(order, "amount", 0.0))
            planned_raw = planned_adj * factor
            record = OrderRecord(
                order_ref=order,
                symbol=order.stock_id,
                side="BUY" if int(order.direction) == 1 else "SELL",
                reason=str(getattr(order, "reason", "")),
                decision_index=decision_index,
                decision_date=str(self.features.dates[decision_index].date()),
                execute_index=executing_index,
                execute_date=str(self.features.dates[executing_index].date()),
                planned_amount_adj=planned_adj,
                planned_raw_shares=planned_raw,
                max_raw_buy_price=getattr(order, "max_raw_buy_price", None),
                min_raw_sell_price=getattr(order, "min_raw_sell_price", None),
            )
            self.orders.append(record)
            self._order_map[id(order)] = record

    def record_execution(self, execute_result: Optional[Sequence[Any]]) -> None:
        """Consume ``[(order, trade_val, trade_cost, trade_price), ...]``."""

        if not execute_result:
            return
        for item in execute_result:
            order, trade_val, trade_cost, trade_price = item
            record = self._order_map.get(id(order))
            if record is None:
                continue
            factor = getattr(order, "factor", None)
            j = self.features.symbol_to_index.get(order.stock_id)
            if factor is None or not np.isfinite(factor) or factor <= 0:
                factor = float(self.features.arr("factor")[record.execute_index, j]) if j is not None else 1.0
            factor = float(factor)
            trade_val = float(trade_val or 0.0)
            trade_cost = float(trade_cost or 0.0)
            trade_price = float(trade_price) if trade_price is not None and np.isfinite(trade_price) else float("nan")
            deal_amount_adj = float(getattr(order, "deal_amount", 0.0) or 0.0)
            raw_shares = deal_amount_adj * factor
            raw_price = trade_price / factor if factor > 0 and np.isfinite(trade_price) else trade_price
            if deal_amount_adj <= 1e-12 or trade_val <= 0 or not np.isfinite(raw_price):
                record.status = "REJECTED"
                record.reject_reason = getattr(order, "reject_reason", "") or "no_fill"
                continue
            record.status = "FILLED" if np.isclose(deal_amount_adj, record.planned_amount_adj, rtol=1e-9, atol=1e-12) else "PARTIAL"
            record.filled_amount_adj = deal_amount_adj
            record.filled_raw_shares = raw_shares
            record.fill_raw_price = raw_price
            record.trade_value = trade_val
            record.total_fee = trade_cost
            self.fills.append(
                Fill(
                    order_id=len(self.orders),
                    symbol=order.stock_id,
                    side=record.side,
                    quantity=int(round(raw_shares)),
                    price=raw_price,
                    date=record.execute_date,
                    trade_index=record.execute_index,
                    notional=trade_val,
                    commission=0.0,
                    stamp_tax=0.0,
                    transfer_fee=0.0,
                    friction=0.0,
                    total_fee=trade_cost,
                    reason=record.reason,
                )
            )
            if record.side == "BUY":
                self._apply_buy(record, deal_amount_adj, raw_shares, raw_price, trade_val, trade_cost)
            else:
                self._apply_sell(record, deal_amount_adj, raw_shares, raw_price, trade_val, trade_cost)

    # ------------------------------------------------------------------
    # position bookkeeping
    # ------------------------------------------------------------------
    def _apply_buy(
        self,
        record: OrderRecord,
        amount_adj: float,
        raw_shares: float,
        raw_price: float,
        trade_val: float,
        cost: float,
    ) -> None:
        pos = self.positions.get(record.symbol)
        j = self.features.symbol_to_index.get(record.symbol)
        adj_price = float(self.features.arr("close")[record.execute_index, j]) if j is not None else np.nan
        if pos is None:
            atr = np.nan
            decision_j = self.features.symbol_to_index.get(record.symbol)
            if decision_j is not None:
                atr = float(self.features.arr("atr20")[record.decision_index, decision_j])
            ref_price = (
                float(self.features.arr("close")[record.decision_index, decision_j])
                if decision_j is not None and np.isfinite(self.features.arr("close")[record.decision_index, decision_j])
                else adj_price
            )
            stop_pct = cfg_stop_pct(self.cfg, atr, ref_price)
            ma20_at_entry = (
                float(self.features.arr("ma20")[record.execute_index, j])
                if j is not None and np.isfinite(self.features.arr("ma20")[record.execute_index, j])
                else np.nan
            )
            pos = Position(
                symbol=record.symbol,
                shares=0,
                entry_index=record.execute_index,
                entry_date=record.execute_date,
                entry_raw_price=raw_price,
                entry_adj_price=adj_price if np.isfinite(adj_price) else raw_price,
                entry_atr=atr if np.isfinite(atr) else 0.0,
                stop_pct=stop_pct,
                cost_basis_total=0.0,
                buy_fees_paid=0.0,
                high_adj=adj_price if np.isfinite(adj_price) else raw_price,
                seen_above_ma20=bool(
                    np.isfinite(adj_price) and np.isfinite(ma20_at_entry) and adj_price > ma20_at_entry
                ),
            )
            self.positions[record.symbol] = pos
        # ``shares`` mirrors Qlib's adjusted amount; this avoids phantom raw
        # share residues when the adjustment factor changes between fills.
        pos.shares += amount_adj
        pos.total_buy_shares += raw_shares
        pos.cost_basis_total += trade_val + cost
        pos.buy_fees_paid += cost
        pos.total_buy_cost += trade_val + cost

    def _apply_sell(
        self,
        record: OrderRecord,
        amount_adj: float,
        raw_shares: float,
        raw_price: float,
        trade_val: float,
        cost: float,
    ) -> None:
        pos = self.positions.get(record.symbol)
        if pos is None or pos.shares <= 0:
            return
        sold = min(float(pos.shares), float(amount_adj))
        before = float(pos.shares)
        proportion = sold / before if before > 0 else 1.0
        pos.shares = max(0.0, before - sold)
        pos.cost_basis_total *= max(0.0, 1.0 - proportion)
        pos.total_sell_gross += trade_val
        pos.total_sell_net += trade_val - cost
        pos.total_sell_shares += raw_shares
        if pos.shares <= 1e-9 or not np.isfinite(pos.cost_basis_total):
            # Complete round trip.
            is_full_exit = record.reason.startswith(("stop", "trend", "trailing", "time", "rank", "market"))
            if pos.shares <= 1e-9 or is_full_exit:
                profit = pos.total_sell_net - pos.total_buy_cost
                entry_price = pos.total_buy_cost / pos.total_buy_shares if pos.total_buy_shares > 0 else np.nan
                exit_price = (
                    pos.total_sell_gross / pos.total_sell_shares if pos.total_sell_shares > 0 else np.nan
                )
                self.round_trips.append(
                    RoundTrip(
                        symbol=record.symbol,
                        entry_date=pos.entry_date,
                        exit_date=record.execute_date,
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        shares=int(round(pos.total_buy_shares)),
                        holding_days=int(record.execute_index - pos.entry_index),
                        buy_cost_total=float(pos.total_buy_cost),
                        sell_proceeds_net=float(pos.total_sell_net),
                        profit=float(profit),
                        return_pct=float(profit / pos.total_buy_cost) if pos.total_buy_cost > 0 else np.nan,
                        exit_reason=record.reason or "unknown",
                    )
                )
                del self.positions[record.symbol]

    # ------------------------------------------------------------------
    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.fills])

    def orders_frame(self) -> pd.DataFrame:
        return pd.DataFrame([o.__dict__ for o in self.orders]).drop(columns=["order_ref"], errors="ignore")

    def round_trips_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.round_trips])

    def open_positions_frame(self, mark_prices: Optional[np.ndarray] = None, current_index: Optional[int] = None) -> pd.DataFrame:
        rows = []
        for symbol, pos in self.positions.items():
            if pos.shares <= 1e-9:
                continue
            j = self.features.symbol_to_index.get(symbol)
            factor = np.nan
            if current_index is not None and j is not None:
                factor = float(self.features.arr("factor")[current_index, j])
            price = pos.entry_raw_price
            if mark_prices is not None and j is not None and np.isfinite(mark_prices[j]):
                price = float(mark_prices[j])
            raw_shares = pos.shares * factor if np.isfinite(factor) and factor > 0 else np.nan
            holding_days = None if current_index is None else int(current_index - pos.entry_index)
            rows.append(
                {
                    "symbol": symbol,
                    "shares": int(round(raw_shares)) if np.isfinite(raw_shares) else np.nan,
                    "entry_date": pos.entry_date,
                    "entry_price": pos.entry_raw_price,
                    "mark_price": price,
                    "market_value": raw_shares * price if np.isfinite(raw_shares) else pos.shares * pos.entry_raw_price,
                    "unrealized_pnl": (raw_shares * price if np.isfinite(raw_shares) else pos.shares * pos.entry_raw_price)
                    - pos.cost_basis_total,
                    "holding_days": holding_days,
                    "stop_pct": pos.stop_pct,
                }
            )
        return pd.DataFrame(rows)


def cfg_stop_pct(cfg: Config, atr: float, ref_price: float) -> float:
    if not np.isfinite(atr) or not np.isfinite(ref_price) or ref_price <= 0:
        return cfg.strategy.stop_max
    return float(
        np.clip(
            cfg.strategy.stop_atr_mult * atr / ref_price,
            cfg.strategy.stop_min,
            cfg.strategy.stop_max,
        )
    )
