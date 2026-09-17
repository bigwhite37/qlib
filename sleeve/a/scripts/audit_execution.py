#!/usr/bin/env python3
"""Audit a completed sleeve/a run for execution-rule violations.

Checks:
* every fill references an order frozen on the previous trading day;
* buy fill prices never exceed the frozen max buy price;
* no order quantity uses post-decision information (max price identity);
* no same-symbol buy and sell on the same execution day;
* planned buy/sell participation respects the 20-day average volume cap;
* cash reconciles between the fill ledger and Qlib's account report;
* account value equals cash + securities value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "execution_audit.json"
    cfg = load_config(args.config)
    orders = pd.read_csv(run_dir / "orders.csv")
    fills = pd.read_csv(run_dir / "fills.csv")
    report = pd.read_csv(run_dir / "report_normal.csv", parse_dates=["datetime"]).set_index("datetime")
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # 1. decision -> next trading index
    bad_exec = orders[orders["execute_index"] != orders["decision_index"] + 1]
    add("orders_execute_next_step", len(bad_exec) == 0, {"violations": int(len(bad_exec))})

    # 2. max buy price
    buy_fills = fills[fills["side"] == "BUY"].copy()
    buy_orders = orders[orders["side"] == "BUY"].copy()
    merged = buy_fills.merge(
        buy_orders[["symbol", "decision_index", "execute_index", "max_raw_buy_price", "planned_raw_shares", "status"]],
        left_on=["symbol", "trade_index"],
        right_on=["symbol", "execute_index"],
        how="left",
        suffixes=("", "_order"),
    )
    above = merged[merged["price"] > merged["max_raw_buy_price"] + cfg.execution.price_tolerance]
    add("buy_fill_le_frozen_max_price", len(above) == 0, {"violations": int(len(above))})

    # 3. max price identity = decision raw close * (1 + premium)
    features = None
    panel = DuckDBPanelLoader(cfg, use_cache=True).load(refresh_cache=False)
    # Reconstruct decision raw close from the panel and compare.
    bad_price_identity = 0
    raw_close = panel.raw_close()
    for row in buy_orders.itertuples(index=False):
        j = panel.symbols.index(row.symbol)
        expected = float(raw_close[int(row.decision_index), j]) * (1.0 + cfg.execution.buy_premium)
        if not np.isfinite(row.max_raw_buy_price) or abs(expected - float(row.max_raw_buy_price)) > 1e-4:
            bad_price_identity += 1
    add("buy_quantity_price_is_decision_only", bad_price_identity == 0, {"violations": bad_price_identity})

    # 4. no opposite orders same symbol/day
    grouped = orders.groupby(["symbol", "execute_index"])["side"].nunique()
    opposite = grouped[grouped > 1]
    add("no_same_symbol_opposite_order_same_day", len(opposite) == 0, {"violations": int(len(opposite))})

    # 5. participation planned cap using 20-day average shares.
    features = build_features(panel, cfg)
    adtv20 = features.arr("adtv20_shares")
    symbol_index = features.symbol_to_index
    bad_planned = 0
    bad_fill_cap = 0
    full_exit_reasons = {
        "market_extreme",
        "stop_loss",
        "trend_fail_ma20",
        "trend_fail_ma60",
        "trailing_stop",
        "time_exit",
        "rank_exit",
        "small_position_cleanup",
        "corr_replace",
    }
    for row in orders.itertuples(index=False):
        j = symbol_index.get(row.symbol)
        if j is None:
            continue
        adv = float(adtv20[int(row.decision_index), j])
        cap = cfg.execution.participation_rate * adv if np.isfinite(adv) else 0.0
        if float(row.planned_raw_shares) > cap + cfg.execution.lot_size + 1e-6:
            # Full liquidation with an unavailable 20-day ADV falls back to the
            # execution-day volume cap, which is enforced below.
            if row.side == "SELL" and str(row.reason) in full_exit_reasons and not np.isfinite(adv):
                continue
            bad_planned += 1
    raw_volume = panel.raw_volume_shares()
    for row in fills.itertuples(index=False):
        j = symbol_index.get(row.symbol)
        if j is None:
            continue
        exec_vol = float(raw_volume[int(row.trade_index), j])
        cap = cfg.execution.participation_rate * exec_vol if np.isfinite(exec_vol) else 0.0
        if float(row.quantity) > cap + 1.0:
            bad_fill_cap += 1
    add("planned_qty_le_20d_volume_participation", bad_planned == 0, {"violations": bad_planned})
    add("fill_qty_le_execution_volume_participation", bad_fill_cap == 0, {"violations": bad_fill_cap})

    # 6. cash reconciliation
    cash = cfg.execution.initial_cash
    for row in fills.itertuples(index=False):
        if row.side == "BUY":
            cash -= row.notional + row.total_fee
        else:
            cash += row.notional - row.total_fee
    report_cash = float(report["cash"].iloc[-1])
    add("ledger_cash_reconciles", abs(cash - report_cash) < 1e-4, {"ledger": cash, "report": report_cash, "diff": cash - report_cash})

    # 7. account value identity
    identity_error = float(np.max(np.abs(report["account"] - (report["cash"] + report["value"]))))
    add("account_equals_cash_plus_value", identity_error < 1e-4, {"max_abs_error": identity_error})

    # 8. same-close sell proceeds are not needed to fund buys: each execution
    # day's buy cash outflow must fit inside the previous close's available cash.
    fills["date"] = pd.to_datetime(fills["date"])
    daily_buy = (
        fills[fills["side"] == "BUY"].assign(cost=lambda d: d["notional"] + d["total_fee"]).groupby("date")["cost"].sum()
    )
    prev_cash = report["cash"].shift(1).fillna(cfg.execution.initial_cash)
    violations = 0
    max_gap = 0.0
    for date, cost in daily_buy.items():
        available = float(prev_cash.loc[date]) if date in prev_cash.index else cfg.execution.initial_cash
        gap = float(cost) - available
        if gap > 1e-4:
            violations += 1
            max_gap = max(max_gap, gap)
    add("buys_do_not_pre_spend_same_close_sells", violations == 0, {"violations": violations, "max_gap": max_gap})

    result = {
        "run_dir": str(run_dir),
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "counts": {"orders": int(len(orders)), "fills": int(len(fills))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
