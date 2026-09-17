"""Acceptance metrics for the sleeve/a design.

All performance numbers are computed from Qlib's account NAV, not from the sum
of stock returns or from a cumulative-excess benchmark curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class NavMetrics:
    start: str
    end: str
    days: int
    initial_nav: float
    final_nav: float
    total_return: float
    cagr: float
    annual_vol: float
    max_drawdown: float
    best_day: float
    worst_day: float


@dataclass
class RoundTripStats:
    n: int
    win_rate: float
    avg_return: float
    median_return: float
    avg_hold_days: float
    profit_factor: float
    by_reason: Dict[str, Dict[str, float]]


@dataclass
class FrequencyStats:
    n_days: int
    effective_days: int
    full_effective_share: float
    rolling_63_mean: float
    rolling_63_min: float
    rolling_63_share_below: float
    rolling_252_mean: float
    rolling_252_min: float
    rolling_252_share_below: float


@dataclass
class AcceptanceReport:
    metrics: Dict[str, Any]
    checks: List[Dict[str, Any]]
    passed: bool


def nav_series(report: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Return the account NAV including the initial cash point."""

    account = report["account"].astype(float).copy()
    if account.empty:
        return account
    first = account.index[0]
    prepend = pd.Series([float(initial_cash)], index=[first - pd.Timedelta(days=1)])
    nav = pd.concat([prepend, account])
    nav.name = "nav"
    return nav


def compute_nav_metrics(nav: pd.Series) -> NavMetrics:
    if len(nav) < 2:
        raise ValueError("NAV series is too short")
    returns = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    return NavMetrics(
        start=str(pd.Timestamp(nav.index[0]).date()),
        end=str(pd.Timestamp(nav.index[-1]).date()),
        days=int(len(nav) - 1),
        initial_nav=float(nav.iloc[0]),
        final_nav=float(nav.iloc[-1]),
        total_return=float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        cagr=float(cagr),
        annual_vol=float(ann_vol),
        max_drawdown=float(drawdown.min()),
        best_day=float(returns.max()) if len(returns) else 0.0,
        worst_day=float(returns.min()) if len(returns) else 0.0,
    )


def compute_round_trip_stats(round_trips: pd.DataFrame) -> RoundTripStats:
    if round_trips.empty:
        return RoundTripStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    profits = round_trips["profit"].astype(float)
    wins = profits > 0
    gross_profit = profits[wins].sum()
    gross_loss = -profits[~wins].sum()
    by_reason: Dict[str, Dict[str, float]] = {}
    for reason, frame in round_trips.groupby("exit_reason"):
        p = frame["profit"].astype(float)
        by_reason[str(reason)] = {
            "n": int(len(frame)),
            "win_rate": float((p > 0).mean()),
            "avg_return": float(frame["return_pct"].astype(float).mean()),
            "avg_hold_days": float(frame["holding_days"].astype(float).mean()),
        }
    return RoundTripStats(
        n=int(len(round_trips)),
        win_rate=float(wins.mean()),
        avg_return=float(round_trips["return_pct"].astype(float).mean()),
        median_return=float(round_trips["return_pct"].astype(float).median()),
        avg_hold_days=float(round_trips["holding_days"].astype(float).mean()),
        profit_factor=float(gross_profit / gross_loss) if gross_loss > 1e-12 else float("inf"),
        by_reason=by_reason,
    )


def compute_frequency_stats(positions: Mapping[Any, Any], cfg: Config) -> FrequencyStats:
    rows: Dict[pd.Timestamp, Dict[str, float]] = {}
    for dt, pos in positions.items():
        cash = float(pos.position.get("cash", 0.0))
        stock_value = float(pos.calculate_stock_value())
        account = float(pos.position.get("now_account_value", cash + stock_value))
        n_stocks = len(pos.get_stock_list())
        rows[pd.Timestamp(dt)] = {
            "stock_value": stock_value,
            "account": account,
            "gross": stock_value / account if account > 0 else 0.0,
            "n": n_stocks,
        }
    frame = pd.DataFrame(rows).T.sort_index()
    if frame.empty:
        return FrequencyStats(0, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    effective = (frame["gross"] >= 0.20) & (frame["n"] >= 3)
    freq63 = effective.astype(float).rolling(63, min_periods=63).mean()
    freq252 = effective.astype(float).rolling(252, min_periods=252).mean()

    def _safe_min(series: pd.Series) -> float:
        series = series.dropna()
        return float(series.min()) if len(series) else float("nan")

    def _safe_mean(series: pd.Series) -> float:
        series = series.dropna()
        return float(series.mean()) if len(series) else float("nan")

    def _share_below(series: pd.Series, threshold: float) -> float:
        series = series.dropna()
        return float((series < threshold).mean()) if len(series) else 0.0

    return FrequencyStats(
        n_days=int(len(frame)),
        effective_days=int(effective.sum()),
        full_effective_share=float(effective.mean()),
        rolling_63_mean=_safe_mean(freq63),
        rolling_63_min=_safe_min(freq63),
        rolling_63_share_below=_share_below(freq63, 0.40),
        rolling_252_mean=_safe_mean(freq252),
        rolling_252_min=_safe_min(freq252),
        rolling_252_share_below=_share_below(freq252, 0.60),
    )


def build_constraints_frame(positions: Mapping[Any, Any], cfg: Config) -> pd.DataFrame:
    """Daily account-constraint audit table (actual fills, not planned)."""

    rows: List[Dict[str, Any]] = []
    for dt, pos in positions.items():
        cash = float(pos.position.get("cash", 0.0))
        stock_value = float(pos.calculate_stock_value())
        account = float(pos.position.get("now_account_value", cash + stock_value))
        n_stocks = len(pos.get_stock_list())
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "nav": account,
                "cash": cash,
                "stock_value": stock_value,
                "gross_exposure": stock_value / account if account > 0 else 0.0,
                "n_positions": n_stocks,
                "effective_holding": bool(stock_value / account >= 0.20 and n_stocks >= 3) if account > 0 else False,
            }
        )
    frame = pd.DataFrame(rows).set_index("datetime").sort_index()
    if not frame.empty:
        frame["rolling_63_effective"] = frame["effective_holding"].astype(float).rolling(63, min_periods=63).mean()
        frame["rolling_252_effective"] = frame["effective_holding"].astype(float).rolling(252, min_periods=252).mean()
    return frame


def quarterly_returns(nav: pd.Series) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame(columns=["quarter", "return", "complete"])
    quarterly = nav.resample("QE").last()
    quarter_returns = quarterly.pct_change()
    if not np.isclose(quarterly.iloc[0], nav.iloc[0]):
        # The first quarter in the series starts after the initial NAV point;
        # its return is measured from the NAV at the previous boundary.
        quarter_returns.iloc[0] = quarterly.iloc[0] / nav.iloc[0] - 1.0
    rows = []
    for dt, ret in quarter_returns.items():
        period = pd.Period(dt, freq="Q")
        complete = (
            period.start_time >= nav.index[0]
            and period.end_time.normalize() <= nav.index[-1].normalize()
            and dt.month in (3, 6, 9, 12)
        )
        rows.append({"quarter": str(period), "return": float(ret), "complete": bool(complete)})
    return pd.DataFrame(rows)


def segment_metrics(
    nav: pd.Series,
    round_trips: pd.DataFrame,
    positions: Mapping[Any, Any],
    cfg: Config,
    segments: Sequence[Tuple[str, str, str]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, start, end in segments:
        mask = (nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))
        sub = nav[mask]
        if len(sub) < 2:
            continue
        # Include the NAV immediately before the segment so the first segment
        # return is not lost.
        prior = nav[nav.index < sub.index[0]]
        if len(prior):
            sub = pd.concat([prior.iloc[[-1]], sub])
        nm = compute_nav_metrics(sub)
        rt = pd.DataFrame()
        if not round_trips.empty:
            rt = round_trips[
                (round_trips["entry_date"] >= start) & (round_trips["exit_date"] <= end)
            ]
        rts = compute_round_trip_stats(rt)
        subpos = {k: v for k, v in positions.items() if start <= str(pd.Timestamp(k).date()) <= end}
        freq = compute_frequency_stats(subpos, cfg) if subpos else None
        out[name] = {
            "start": start,
            "end": end,
            "days": nm.days,
            "total_return": nm.total_return,
            "cagr": nm.cagr,
            "annual_vol": nm.annual_vol,
            "max_drawdown": nm.max_drawdown,
            "round_trips": rts.n,
            "win_rate": rts.win_rate,
            "avg_round_return": rts.avg_return,
            "avg_hold_days": rts.avg_hold_days,
            "effective_share": freq.full_effective_share if freq else float("nan"),
            "rolling_63_min": freq.rolling_63_min if freq else float("nan"),
            "rolling_252_min": freq.rolling_252_min if freq else float("nan"),
        }
    return out


def acceptance_report(
    nav_metrics: NavMetrics,
    frequency: FrequencyStats,
    round_trip: RoundTripStats,
    quarters: pd.DataFrame,
    cfg: Config,
) -> AcceptanceReport:
    complete = quarters[quarters["complete"]] if not quarters.empty else quarters
    negative_quarters = complete[complete["return"] <= 0] if not complete.empty else complete
    checks = [
        {
            "name": "cagr_ge_25pct",
            "passed": bool(nav_metrics.cagr >= 0.25),
            "value": nav_metrics.cagr,
            "threshold": ">= 0.25",
        },
        {
            "name": "annual_vol_le_10pct",
            "passed": bool(nav_metrics.annual_vol <= 0.10),
            "value": nav_metrics.annual_vol,
            "threshold": "<= 0.10",
        },
        {
            "name": "max_drawdown_ge_minus10pct",
            "passed": bool(nav_metrics.max_drawdown >= -0.10),
            "value": nav_metrics.max_drawdown,
            "threshold": ">= -0.10",
        },
        {
            "name": "round_trip_win_rate_ge_70pct",
            "passed": bool(round_trip.win_rate >= 0.70),
            "value": round_trip.win_rate,
            "threshold": ">= 0.70",
        },
        {
            "name": "all_complete_quarters_positive",
            "passed": bool(len(complete) > 0 and len(negative_quarters) == 0),
            "value": int(len(negative_quarters)),
            "threshold": "0 negative complete quarters",
        },
        {
            "name": "rolling_252_effective_ge_60pct",
            "passed": bool(np.isfinite(frequency.rolling_252_min) and frequency.rolling_252_min >= 0.60),
            "value": frequency.rolling_252_min,
            "threshold": ">= 0.60",
        },
        {
            "name": "rolling_63_effective_ge_40pct",
            "passed": bool(np.isfinite(frequency.rolling_63_min) and frequency.rolling_63_min >= 0.40),
            "value": frequency.rolling_63_min,
            "threshold": ">= 0.40",
        },
    ]
    return AcceptanceReport(
        metrics={
            "nav": nav_metrics.__dict__,
            "round_trip": round_trip.__dict__,
            "frequency": frequency.__dict__,
        },
        checks=checks,
        passed=all(c["passed"] for c in checks),
    )
