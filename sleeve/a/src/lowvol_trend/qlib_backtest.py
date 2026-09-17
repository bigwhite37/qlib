"""Qlib backtest integration for the sleeve/a strategy.

The strategy logic (signals, market state, exits, target weights) is computed
from the panel prepared by :mod:`lowvol_trend.data` / :mod:`lowvol_trend.features`.
Execution, account keeping, portfolio metrics and the standard observability
records are delegated to Qlib:

* ``Exchange`` subclass for close-auction matching with A-share limit rules,
  frozen max-buy prices and exact fees;
* a custom ``Quote`` implementation so Qlib reads our DuckDB panel directly
  instead of issuing thousands of individual expression queries;
* ``SimulatorExecutor(trade_type="parallel")`` so buys cannot spend the same
  close's sell proceeds;
* ``WeightStrategyBase`` + a frozen-order ``OrderGenerator``;
* ``qlib.workflow.R`` / ``risk_analysis`` for the observable experiment
  artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .bootstrap import ensure_local_qlib

ensure_local_qlib()

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO  # noqa: E402
from qlib.backtest.exchange import Exchange  # noqa: E402
from qlib.backtest.high_performance_ds import BaseQuote, NumpyQuote  # noqa: E402
from qlib.backtest.signal import Signal  # noqa: E402
from qlib.contrib.strategy.order_generator import OrderGenerator  # noqa: E402
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase  # noqa: E402
from qlib.utils.index_data import SingleData  # noqa: E402

from .config import Config  # noqa: E402
from .data import infer_board  # noqa: E402
from .execution import board_limit_pct, compute_fees, effective_limit_pct  # noqa: E402
from .features import FeatureStore  # noqa: E402
from .ledger import ExecutionLedger  # noqa: E402
from .portfolio import construct_targets, evaluate_exits, update_position_observations  # noqa: E402


@dataclass
class ConstrainedOrder(Order):
    """Qlib order with the extra constraints required by the design."""

    max_raw_buy_price: Optional[float] = None
    min_raw_sell_price: Optional[float] = None
    reason: str = ""


class _LazyPanelField:
    """Panel-backed array that pages in only when Qlib actually reads it."""

    def __init__(self, panel, name: str):
        self._panel = panel
        self._name = name
        self._data = None

    def _materialise(self) -> np.ndarray:
        if self._data is None:
            self._data = np.asarray(self._panel.field(self._name), dtype=np.float32)
        return self._data

    def __getitem__(self, key):
        return self._materialise()[key]

    def __array__(self, dtype=None):
        data = self._materialise()
        return data if dtype is None else data.astype(dtype)


class PanelQuote(BaseQuote):
    """Serve Qlib's quote interface straight from the in-memory panel."""

    _FIELD_ALIASES = {
        "close": "$close",
        "factor": "$factor",
        "volume": "$volume",
        "change": "$change",
        "open": "$open",
        "high": "$high",
        "low": "$low",
    }

    def __init__(self, features: FeatureStore, freq: str = "day"):
        super().__init__(quote_df=pd.DataFrame(), freq=freq)
        self.features = features
        self.panel = features.panel
        valid = self.panel.valid
        close = self.panel.field("close")
        # Suspended days must look like missing $close to Qlib.
        nan32 = np.float32(np.nan)
        self._fields: Dict[str, np.ndarray] = {
            "$close": np.where(valid, close, nan32).astype(np.float32, copy=False),
            "$factor": np.asarray(self.panel.field("factor"), dtype=np.float32),
            "$volume": np.where(valid, self.panel.field("volume"), np.float32(0.0)).astype(np.float32, copy=False),
            "$change": np.where(valid, features.arr("ret1"), nan32).astype(np.float32, copy=False),
            "$open": _LazyPanelField(self.panel, "open"),
            "$high": _LazyPanelField(self.panel, "high"),
            "$low": _LazyPanelField(self.panel, "low"),
        }
        # Forward-fill factor so Qlib can round adjusted amounts to A-share lots
        # even after a data gap (done in place to avoid extra copies).
        factor_data = np.array(self.panel.field("factor"), dtype=np.float32, copy=True)
        factor_frame = pd.DataFrame(factor_data, copy=False)
        factor_frame.ffill(inplace=True)
        self._fields["$factor"] = factor_frame.to_numpy(dtype=np.float32)
        self._symbols = list(features.symbols)
        self._symbol_to_j = {s: j for j, s in enumerate(self._symbols)}
        self._dates = features.dates

    def get_all_stock(self):
        return self._symbols

    def _resolve_field(self, field: str) -> Optional[str]:
        if field in self._fields:
            return field
        return self._FIELD_ALIASES.get(str(field).lstrip("$"))

    def _slice(self, stock_id: str, start_time, end_time, field: str):
        name = self._resolve_field(field)
        if name is None or stock_id not in self._symbol_to_j:
            return None
        j = self._symbol_to_j[stock_id]
        start = pd.Timestamp(start_time).normalize()
        end = pd.Timestamp(end_time).normalize()
        lo = int(np.searchsorted(self._dates.values, np.datetime64(start), side="left"))
        hi = int(np.searchsorted(self._dates.values, np.datetime64(end), side="right")) - 1
        lo = max(0, min(lo, len(self._dates) - 1))
        hi = max(0, min(hi, len(self._dates) - 1))
        if hi < lo:
            return None
        return self._fields[name][lo : hi + 1, j], self._dates[lo : hi + 1]

    def get_data(
        self,
        stock_id: str,
        start_time,
        end_time,
        field: str,
        method: Optional[str] = None,
    ):
        sliced = self._slice(stock_id, start_time, end_time, field)
        if sliced is None:
            return None
        values, index = sliced
        values = np.asarray(values)
        if method is None:
            return SingleData(pd.Series(values, index=index, dtype=float))
        if method in ("ts_data_last", "last"):
            finite = values[np.isfinite(values)]
            return float(finite[-1]) if finite.size else None
        if method == "sum":
            return float(np.nansum(values))
        if method == "mean":
            finite = values[np.isfinite(values)]
            return float(finite.mean()) if finite.size else None
        if method == "max":
            finite = values[np.isfinite(values)]
            return float(finite.max()) if finite.size else None
        if method == "min":
            finite = values[np.isfinite(values)]
            return float(finite.min()) if finite.size else None
        if method in ("all", "any"):
            # Return the raw series; Qlib's limit-checking path is replaced by
            # our own ``check_stock_limit``, so this is only a safe fallback.
            return SingleData(pd.Series(values, index=index, dtype=float))
        raise ValueError(f"Unsupported quote method {method!r} for field {field!r}")


class LowVolExchange(Exchange):
    """Qlib exchange with A-share board/date limits and frozen-price checks."""

    def __init__(self, features: FeatureStore, cfg: Config, *args, **kwargs):
        self._features = features
        self._cfg = cfg
        self._panel = features.panel
        self._boards = {s: str(features.panel.instrument_meta.loc[s, "board"]) for s in features.symbols}
        self._raw_close = self._panel.raw_close()
        mark_data = np.array(self._raw_close, dtype=np.float32, copy=True)
        mark_frame = pd.DataFrame(mark_data, copy=False)
        mark_frame.ffill(inplace=True)
        self._raw_mark = mark_frame.to_numpy(dtype=np.float32)
        self._valid = self._panel.valid
        self._factor = self._panel.field("factor")
        self._volume = self._panel.field("volume")
        kwargs.setdefault("quote_cls", NumpyQuote)
        super().__init__(*args, **kwargs)
        # Replace Qlib's D.features-backed quote with our panel-backed quote.
        self.quote = PanelQuote(features, freq=self.freq)

    def get_quote_from_qlib(self) -> None:
        """Skip Qlib's per-expression data loading; the panel is already loaded."""

        fields = sorted({self.buy_price, self.sell_price, "$close", "$change", "$factor", "$volume"})
        self.all_fields = fields
        index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype=object), pd.DatetimeIndex([])],
            names=["instrument", "datetime"],
        )
        self.quote_df = pd.DataFrame({name: pd.Series(dtype=float) for name in fields}, index=index)
        self.trade_w_adj_price = False

    # ------------------------------------------------------------------
    # A-share trading limits
    # ------------------------------------------------------------------
    def _execution_index(self, time_value) -> Optional[int]:
        date = pd.Timestamp(time_value).normalize()
        return self._features.date_to_index.get(date)

    def _limit_flags(self, stock_id: str, time_value) -> Optional[Tuple[bool, bool]]:
        t = self._execution_index(time_value)
        j = self._features.symbol_to_index.get(stock_id)
        if t is None or j is None or t <= 0:
            return None
        if not bool(self._valid[t, j]):
            return True, True
        raw_cur = float(self._raw_close[t, j]) if np.isfinite(self._raw_close[t, j]) else np.nan
        raw_prev = float(self._raw_mark[t - 1, j]) if np.isfinite(self._raw_mark[t - 1, j]) else np.nan
        if not np.isfinite(raw_cur) or not np.isfinite(raw_prev) or raw_prev <= 0:
            return True, True
        date_int = int(self._panel.date_ints[t])
        board = self._boards.get(stock_id, "unknown")
        limit = effective_limit_pct(board, date_int, self._cfg)
        tol = self._cfg.execution.price_tolerance

        def _tick(value: float) -> float:
            # A-share price tick is 0.01 yuan; exchanges round the limit price
            # half-up from the previous close.
            return float(np.floor(value * 100.0 + 0.5) / 100.0)

        prev_tick = _tick(raw_prev)
        cur_tick = _tick(raw_cur)
        limit_up = _tick(prev_tick * (1.0 + limit))
        limit_down = _tick(prev_tick * (1.0 - limit))
        buy_locked = cur_tick >= limit_up - tol
        sell_locked = cur_tick <= limit_down + tol
        return bool(buy_locked), bool(sell_locked)

    def check_stock_limit(self, stock_id: str, start_time, end_time, direction: int | None = None) -> bool:
        flags = self._limit_flags(stock_id, start_time)
        if flags is None:
            return True
        buy_locked, sell_locked = flags
        if direction is None:
            return bool(buy_locked or sell_locked)
        return bool(buy_locked) if int(direction) == int(OrderDir.BUY) else bool(sell_locked)

    # ------------------------------------------------------------------
    # Frozen max-buy price / minimum sell price
    # ------------------------------------------------------------------
    def _execution_raw_price(self, order: Order) -> Optional[float]:
        price = self.get_deal_price(
            order.stock_id,
            order.start_time,
            order.end_time,
            direction=order.direction,
        )
        factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        if price is None or factor is None or not np.isfinite(price) or not np.isfinite(factor) or factor <= 0:
            return None
        return float(price) / float(factor)

    def check_order(self, order: Order) -> bool:
        if not super().check_order(order):
            return False
        raw_price = self._execution_raw_price(order)
        if raw_price is None:
            return False
        tol = self._cfg.execution.price_tolerance
        max_raw = getattr(order, "max_raw_buy_price", None)
        if max_raw is not None and int(order.direction) == int(OrderDir.BUY) and raw_price > float(max_raw) + tol:
            return False
        min_raw = getattr(order, "min_raw_sell_price", None)
        if min_raw is not None and int(order.direction) == int(OrderDir.SELL) and raw_price < float(min_raw) - tol:
            return False
        return True

    # ------------------------------------------------------------------
    # Volume participation + exact fees
    # ------------------------------------------------------------------
    def _execution_volume_cap_adj(self, order: Order) -> float:
        t = self._execution_index(order.start_time)
        j = self._features.symbol_to_index.get(order.stock_id)
        if t is None or j is None:
            return 0.0
        if not bool(self._valid[t, j]):
            return 0.0
        # Qlib amount unit is raw shares / factor.  Raw shares are
        # qlib_volume * factor * 100, so the cap in qlib amount units is
        # participation * qlib_volume * 100.
        volume_qlib = float(self._volume[t, j])
        if not np.isfinite(volume_qlib) or volume_qlib <= 0:
            return 0.0
        return max(0.0, self._cfg.execution.participation_rate * volume_qlib * 100.0)

    def _calc_trade_info_by_order(self, order: Order, position, dealt_order_amount):
        cap = self._execution_volume_cap_adj(order)
        already = float(dealt_order_amount.get(order.stock_id, 0.0))
        remaining_cap = max(0.0, cap - already)
        # Exchange._calc_trade_info_by_order resets deal_amount from
        # order.amount, so the cap must be applied to order.amount itself.
        order.amount = min(float(order.amount), remaining_cap)
        trade_price, trade_val, trade_cost = super()._calc_trade_info_by_order(
            order, position, dealt_order_amount
        )
        if trade_val is not None and np.isfinite(trade_val) and trade_val > 1e-5:
            t = self._execution_index(order.start_time)
            date_int = int(self._panel.date_ints[t]) if t is not None else 29990101
            side = "BUY" if int(order.direction) == int(OrderDir.BUY) else "SELL"
            fees = compute_fees(side, float(trade_val), date_int, self._cfg)
            trade_cost = float(fees["total_fee"])
        return trade_price, trade_val, trade_cost


class RuleSignal(Signal):
    """Signal wrapper so the strategy can reuse ``WeightStrategyBase``."""

    def __init__(self, features: FeatureStore):
        self.features = features

    def get_signal(self, start_time, end_time):
        # ``pred_start_time`` is the decision day itself.  ``pred_end_time``
        # may fall on a weekend/holiday (e.g. Friday decision for Monday
        # execution) for which there is no panel row, so keying the lookup on
        # the start time is both correct and robust.
        date = pd.Timestamp(start_time).normalize()
        t = self.features.date_to_index.get(date)
        if t is None:
            return None
        return pd.Series(
            self.features.arr("entry_score")[t],
            index=self.features.symbols,
            name="score",
        )


class FrozenOrderGenerator(OrderGenerator):
    """Generate quantity-frozen orders using only decision-day information."""

    def __init__(self, strategy: "LowVolTrendStrategy"):
        self.strategy = strategy
        self.features = strategy.features
        self.cfg = strategy.cfg

    def generate_order_list_from_target_weight_position(
        self,
        current,
        trade_exchange,
        target_weight_position,
        risk_degree,
        pred_start_time,
        pred_end_time,
        trade_start_time,
        trade_end_time,
    ) -> List[Order]:
        if not target_weight_position:
            target_weight_position = {}
        features = self.features
        t = features.date_to_index.get(pd.Timestamp(pred_start_time).normalize())
        if t is None:
            return []
        nav = float(current.calculate_value())
        cash = float(current.get_cash(include_settle=False))
        if nav <= 0:
            return []
        current_amounts = {k: float(v) for k, v in current.get_stock_amount_dict().items()}
        symbols = sorted(set(target_weight_position) | set(current_amounts))
        buy_candidates: List[Tuple[float, Order]] = []
        sell_orders: List[Order] = []
        remaining_cash = cash
        fee_buffer = 0.0025

        for symbol in symbols:
            j = features.symbol_to_index.get(symbol)
            if j is None:
                continue
            adj_price = trade_exchange.get_close(symbol, pred_start_time, pred_end_time)
            factor = trade_exchange.get_factor(symbol, pred_start_time, pred_end_time)
            if (
                adj_price is None
                or factor is None
                or not np.isfinite(adj_price)
                or not np.isfinite(factor)
                or float(adj_price) <= 0
                or float(factor) <= 0
            ):
                continue
            adj_price = float(adj_price)
            factor = float(factor)
            current_amount = current_amounts.get(symbol, 0.0)
            target_w = float(target_weight_position.get(symbol, 0.0))
            desired_value = max(0.0, target_w) * nav
            desired_amount = desired_value / adj_price
            delta = desired_amount - current_amount
            raw_decision = float(features.arr("raw_close")[t, j]) if np.isfinite(features.arr("raw_close")[t, j]) else np.nan
            if delta > 1e-12:
                # Never average down an existing position unless the target
                # system explicitly asks for more; the portfolio constructor
                # does not, so this is a safety guard.
                if current_amount > 1e-9:
                    continue
                if not np.isfinite(raw_decision) or raw_decision <= 0:
                    continue
                adv = float(features.arr("adtv20_shares")[t, j])
                if np.isfinite(adv) and adv > 0:
                    cap_raw_shares = self.cfg.execution.participation_rate * adv
                    cap_amount = cap_raw_shares / factor
                    delta = min(delta, cap_amount)
                max_raw_buy = raw_decision * (1.0 + self.cfg.execution.buy_premium)
                # Cash budget: use the frozen maximum price and a fee buffer.
                affordable_raw_shares = remaining_cash / (max_raw_buy * (1.0 + fee_buffer))
                affordable_amount = affordable_raw_shares / factor
                delta = min(delta, affordable_amount)
                delta = trade_exchange.round_amount_by_trade_unit(delta, factor)
                if delta <= 1e-12:
                    continue
                remaining_cash = max(
                    0.0,
                    remaining_cash - delta * max_raw_buy * factor * (1.0 + fee_buffer),
                )
                buy_candidates.append(
                    (
                        float(target_w),
                        ConstrainedOrder(
                            stock_id=symbol,
                            amount=float(delta),
                            direction=Order.BUY,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            max_raw_buy_price=float(max_raw_buy),
                            reason="new_entry",
                        ),
                    )
                )
            elif delta < -1e-12:
                reason = self.strategy.current_exits.get(symbol, "trim_to_target")
                full_exit = target_w <= 1e-9 or symbol in self.strategy.current_exits
                if not full_exit:
                    # A partial trim below the minimum size is not worth the
                    # minimum commission and the extra round-trip accounting:
                    # risk reductions wait until they are large enough or until
                    # the position is closed outright.
                    trim_weight = min(abs(delta), current_amount) * adj_price / nav if nav > 0 else 0.0
                    if trim_weight < self.cfg.strategy.min_trim_weight:
                        continue
                qty = current_amount if full_exit else min(abs(delta), current_amount)
                # Design rule: no order may exceed the planned share of the
                # previous 20-day average volume.  A capped exit is retried on
                # the next decision day if the signal is still active.
                adv = float(features.arr("adtv20_shares")[t, j])
                if np.isfinite(adv) and adv > 0:
                    planned_cap_raw = self.cfg.execution.participation_rate * adv
                    planned_cap_amount = planned_cap_raw / factor
                    qty = min(qty, planned_cap_amount)
                full_exit_after_cap = full_exit and qty >= current_amount * (1.0 - 1e-9)
                if full_exit_after_cap:
                    # Closing an odd-lot residue is allowed as one order; do not
                    # floor it away into an unsellable zero-lot position.
                    qty = current_amount
                else:
                    qty = trade_exchange.round_amount_by_trade_unit(qty, factor)
                if qty <= 1e-12:
                    continue
                sell_orders.append(
                    ConstrainedOrder(
                        stock_id=symbol,
                        amount=float(qty),
                        direction=Order.SELL,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        min_raw_sell_price=None,
                        reason=reason,
                    )
                )

        buy_candidates.sort(key=lambda item: item[0], reverse=True)
        return [order for _, order in buy_candidates] + sell_orders


class LowVolTrendStrategy(WeightStrategyBase):
    """Rule V0 implemented as a Qlib ``WeightStrategyBase``."""

    def __init__(
        self,
        features: FeatureStore,
        cfg: Config,
        ledger: ExecutionLedger,
        market,
        *,
        signal=None,
        trade_exchange=None,
        **kwargs,
    ):
        self.features = features
        self.cfg = cfg
        self.ledger = ledger
        self._market = market
        self.peak_nav: Optional[float] = None
        self._nav_history: List[float] = []
        self.current_exits: Dict[str, str] = {}
        self.decision_log: List[Dict[str, Any]] = []
        self._pending_orders: List[Order] = []
        if signal is None:
            signal = RuleSignal(features)
        super().__init__(
            signal=signal,
            trade_exchange=trade_exchange,
            order_generator_cls_or_obj=FrozenOrderGenerator(self),
            **kwargs,
        )

    def get_risk_degree(self, trade_step=None):
        # Absolute target weights already include all market/drawdown/vol caps.
        return 1.0

    # ------------------------------------------------------------------
    # Exit-rule hooks.  The default implementation is the V0 rule set; the V2
    # strategy overrides these two methods so that the live exits are generated
    # by the very same ExitPolicy the labels were built from.
    # ------------------------------------------------------------------
    def update_observations(self, t: int) -> None:
        update_position_observations(self.features, t, self.ledger.positions, self.cfg)

    def evaluate_exits(self, t: int) -> Dict[str, str]:
        return evaluate_exits(self.features, t, self.ledger.positions, self.strategy_market(), self.cfg)

    # ------------------------------------------------------------------
    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        t = self.features.decision_index(trade_start_time)
        if t < 0:
            self.current_exits = {}
            return {}
        self.update_observations(t)
        exits = self.evaluate_exits(t)
        self.current_exits = dict(exits)

        amounts = current.get_stock_amount_dict()
        adj_close = self.features.arr("close")[t]
        weights: Dict[str, float] = {}
        stock_value = 0.0
        values: Dict[str, float] = {}
        for symbol, amount in amounts.items():
            j = self.features.symbol_to_index.get(symbol)
            if j is not None and np.isfinite(adj_close[j]):
                price = float(adj_close[j])
            else:
                price = float(current.get_stock_price(symbol))
            value = float(amount) * price
            values[symbol] = value
            stock_value += value
        cash = float(current.get_cash(include_settle=False))
        nav = stock_value + cash
        if nav <= 0:
            self.current_exits = {}
            return {}
        weights = {s: v / nav for s, v in values.items()}
        self._nav_history.append(nav)
        peak_window = int(getattr(self.cfg.strategy, "drawdown_peak_window", 0) or 0)
        if peak_window > 0:
            # Rolling peak: lets a cut-to-cash rule release itself, which an
            # all-time peak can never do while the account sits in cash.
            self.peak_nav = max(self._nav_history[-peak_window:])
        else:
            self.peak_nav = nav if self.peak_nav is None else max(self.peak_nav, nav)

        held_caps: Dict[str, float] = {}
        for symbol in amounts:
            pos = self.ledger.positions.get(symbol)
            stop_pct = pos.stop_pct if pos is not None else self.cfg.strategy.stop_max
            held_caps[symbol] = min(
                self.cfg.strategy.max_weight,
                self.cfg.strategy.risk_per_trade / max(float(stop_pct), 1e-6),
            )

        # Scale-out: positions whose first profit target has been reached have their
        # target weight cut once, and the remainder runs to the second target.
        scale_factors: Dict[str, float] = {}
        scale_fraction = float(getattr(self.cfg.strategy, "scale_out_fraction", 0.0) or 0.0)
        if scale_fraction > 0.0:
            for symbol in amounts:
                pos = self.ledger.positions.get(symbol)
                if pos is not None and pos.scaled_out and not pos.scale_applied:
                    scale_factors[symbol] = max(0.0, 1.0 - scale_fraction)
                    pos.scale_applied = True

        plan = construct_targets(
            features=self.features,
            cfg=self.cfg,
            t=t,
            current_weights=weights,
            held_symbols=list(amounts.keys()),
            held_caps=held_caps,
            nav=nav,
            peak_nav=self.peak_nav,
            cash=cash,
            market=self.strategy_market(),
            exits=exits,
            scale_factors=scale_factors,
        )
        self.decision_log.append(
            {
                "decision_index": t,
                "decision_date": str(self.features.dates[t].date()),
                "nav": nav,
                "cash": cash,
                "market_state": str(self.strategy_market().effective_state[t]),
                "market_cap": float(self.strategy_market().effective_cap[t]),
                "gross_cap": plan.gross_cap,
                "drawdown_cap": plan.drawdown_cap,
                "pre_vol": plan.preliminary_vol,
                "scaled_vol": plan.scaled_vol,
                "n_positions": len(amounts),
                "n_exits": len(exits),
                "exits": dict(exits),
                "n_new": len(plan.new_symbols),
                "new_symbols": list(plan.new_symbols),
                "target_gross": float(sum(plan.target_weights.values())),
                "diagnostics": plan.diagnostics,
            }
        )
        return plan.target_weights

    # ------------------------------------------------------------------
    def generate_trade_decision(self, execute_result=None):
        decision = super().generate_trade_decision(execute_result)
        if isinstance(decision, TradeDecisionWO):
            self._pending_orders = list(decision.get_decision())
            if self.features is not None and self._pending_orders:
                t = self.features.decision_index(self.trade_calendar.get_step_time()[0])
                if t >= 0:
                    self.ledger.record_orders(self._pending_orders, t)
        return decision

    def post_exe_step(self, execute_result: Optional[list]) -> None:
        self.ledger.record_execution(execute_result)

    def strategy_market(self):
        return self._market
