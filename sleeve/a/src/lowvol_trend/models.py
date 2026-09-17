"""Small data objects shared by the strategy and execution layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Position:
    symbol: str
    shares: float  # Qlib adjusted amount (raw shares = amount * factor)
    entry_index: int
    entry_date: str
    entry_raw_price: float
    entry_adj_price: float
    entry_atr: float
    stop_pct: float
    cost_basis_total: float  # remaining shares, including buy fees
    buy_fees_paid: float
    high_adj: float
    trailing_armed: bool = False
    below_top_half_days: int = 0
    seen_above_ma20: bool = False
    last_exit_reason: Optional[str] = None
    total_buy_cost: float = 0.0
    total_sell_gross: float = 0.0
    total_sell_net: float = 0.0
    total_buy_shares: float = 0.0
    total_sell_shares: float = 0.0
    # Scale-out bookkeeping: set when the first profit target has been reached, and
    # again once the reduced target weight has been applied to the order plan.
    scaled_out: bool = False
    scale_applied: bool = False

    @property
    def entry_stop_adj_price(self) -> float:
        return self.entry_adj_price * (1.0 - self.stop_pct)


@dataclass
class Order:
    order_id: int
    symbol: str
    side: str  # BUY / SELL
    quantity: int
    decision_index: int
    decision_date: str
    execute_index: int
    execute_date: str
    reason: str
    decision_price_raw: float
    max_buy_price: Optional[float] = None
    min_sell_price: Optional[float] = None
    target_weight: float = 0.0
    filled_quantity: int = 0
    fill_price: Optional[float] = None
    status: str = "PENDING"
    reject_reason: str = ""
    future_price_used: bool = False
    cash_budget_basis: float = 0.0


@dataclass
class Fill:
    order_id: int
    symbol: str
    side: str
    quantity: int
    price: float
    date: str
    trade_index: int
    notional: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    friction: float
    total_fee: float
    reason: str


@dataclass
class RoundTrip:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    holding_days: int
    buy_cost_total: float
    sell_proceeds_net: float
    profit: float
    return_pct: float
    exit_reason: str


@dataclass
class DailyRecord:
    date: str
    trade_index: int
    nav: float
    cash: float
    market_value: float
    gross_exposure: float
    n_positions: int
    effective_holding: bool
    peak_nav: float
    drawdown: float
    market_state: str
    market_cap: float
    gross_cap: float
    n_candidates: int
    n_orders: int
    n_fills: int
    turnover: float
    costs: float
