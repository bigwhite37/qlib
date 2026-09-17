"""Lean feature builder for the V2 strategy (3 GB memory budget).

The V0 builder materialises ~28 panel-sized frames at once.  V2 reads only a
handful of arrays, so this module computes exactly those, one at a time, and
releases every temporary immediately.  Formulas match
:func:`lowvol_trend.features.build_features` exactly for the arrays they share.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .data import PanelData
from .features import FeatureStore, row_pct_rank
from .market import compute_market_state


def build_v2_features(panel: PanelData, cfg: Config, micro: bool = False) -> FeatureStore:
    """Build the minimal causal feature set the V2 strategy reads."""

    dates = panel.dates
    symbols = panel.symbols
    frames = {
        name: pd.DataFrame(np.asarray(panel.field(name)), index=dates, columns=symbols, copy=False)
        for name in ("close", "high", "low", "volume", "amount", "factor")
    }
    close = frames["close"]
    valid = pd.DataFrame(np.asarray(panel.valid), index=dates, columns=symbols, copy=False)

    arrays: Dict[str, np.ndarray] = {}
    arrays["close"] = np.asarray(panel.field("close"), dtype=np.float32)
    with np.errstate(all="ignore"):
        raw_close = (close / frames["factor"]).to_numpy(dtype=np.float32)
    raw_close[~np.isfinite(raw_close)] = np.nan
    arrays["raw_close"] = raw_close
    arrays["factor"] = np.asarray(panel.field("factor"), dtype=np.float32)

    ret1 = close.pct_change(fill_method=None).to_numpy(dtype=np.float32)
    arrays["ret1"] = ret1
    ret60 = close.pct_change(60, fill_method=None).to_numpy(dtype=np.float32)
    del close

    arrays["ma20"] = frames["close"].rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
    arrays["ma60"] = frames["close"].rolling(60, min_periods=60).mean().to_numpy(dtype=np.float32)

    prev_close = frames["close"].shift(1)
    true_range = np.maximum(
        (frames["high"] - frames["low"]).to_numpy(dtype=np.float64),
        np.maximum(
            (frames["high"] - prev_close).abs().to_numpy(dtype=np.float64),
            (frames["low"] - prev_close).abs().to_numpy(dtype=np.float64),
        ),
    )
    del prev_close
    arrays["atr20"] = (
        pd.DataFrame(true_range, index=dates, columns=symbols, copy=False)
        .rolling(20, min_periods=20)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    del true_range

    arrays["vol60"] = (
        pd.DataFrame(ret1, index=dates, columns=symbols, copy=False)
        .rolling(60, min_periods=60)
        .std()
        .to_numpy(dtype=np.float32)
    )

    # 60-day trend quality: net move divided by the sum of absolute daily moves.
    net_move = (frames["close"] - frames["close"].shift(60)).to_numpy(dtype=np.float64)
    swing = (
        frames["close"].diff().abs().rolling(60, min_periods=60).sum().to_numpy(dtype=np.float64)
    )
    with np.errstate(all="ignore"):
        trend_quality = net_move / swing
    trend_quality[~np.isfinite(trend_quality)] = np.nan
    arrays["trend_quality60"] = trend_quality.astype(np.float32)
    del net_move, swing

    # Range-based volatility and limit-up activity: the two strongest features in
    # the round-6 screen (20-day rank IC -0.104 and -0.087 respectively).
    # They are only needed by the alternative rule composites, so they are
    # opt-in to keep the builder inside the memory budget.
    if micro:
        with np.errstate(all="ignore"):
            parkinson = (np.log(frames["high"] / frames["low"]) ** 2).to_numpy(dtype=np.float32)
        arrays["parkinson20"] = (
            pd.DataFrame(parkinson, index=dates, columns=symbols, copy=False)
            .rolling(20, min_periods=20)
            .mean()
            .to_numpy(dtype=np.float32)
        )
        del parkinson
        raw_frame = pd.DataFrame(raw_close, index=dates, columns=symbols, copy=False)
        boards = panel.instrument_meta["board"].reindex(symbols).fillna("unknown").to_numpy(dtype=object)
        limit = np.full((len(dates), len(symbols)), 0.10, dtype=np.float64)
        for j, board in enumerate(boards):
            if board == "star":
                limit[:, j] = 0.20
            elif board == "bse":
                limit[:, j] = 0.30
            elif board == "chinext":
                limit[:, j] = np.where(panel.date_ints >= 20200824, 0.20, 0.10)
        with np.errstate(all="ignore"):
            change = (raw_frame / raw_frame.shift(1) - 1.0).to_numpy(dtype=np.float64)
        up_limit = (change >= limit - 0.005).astype(np.float32)
        arrays["limit_up_20"] = (
            pd.DataFrame(up_limit, index=dates, columns=symbols, copy=False)
            .rolling(20, min_periods=20)
            .sum()
            .to_numpy(dtype=np.float32)
        )
        arrays["ret_max5"] = (
            pd.DataFrame(ret1, index=dates, columns=symbols, copy=False)
            .rolling(5, min_periods=5)
            .max()
            .to_numpy(dtype=np.float32)
        )
        del change, limit, up_limit, raw_frame

    # VWAP family: the design dropped VWAP features on the assumption that the
    # field was missing, but it is fully populated here.  The 20-day VWAP momentum
    # and the 20-day VWAP dispersion measured rank IC -0.070 and -0.068 against
    # 20-day forward returns (scripts/audit_vwap_ic.py).
    vwap = pd.DataFrame(
        np.asarray(panel.field("vwap"), dtype=np.float32), index=dates, columns=symbols, copy=False
    )
    vwap = vwap.where(vwap > 0)
    arrays["vwap_dev5"] = (
        (frames["close"] / vwap - 1.0).rolling(5, min_periods=5).mean().to_numpy(dtype=np.float32)
    )
    with np.errstate(all="ignore"):
        arrays["vwap_mom20"] = (vwap / vwap.shift(20) - 1.0).to_numpy(dtype=np.float32)
        vwap_mean20 = vwap.rolling(20, min_periods=20).mean()
        arrays["vwap_cv20"] = (
            vwap.rolling(20, min_periods=20).std() / vwap_mean20.where(vwap_mean20 > 0)
        ).to_numpy(dtype=np.float32)
    del vwap, vwap_mean20

    amount_yuan = frames["amount"] * 1000.0
    amount20 = amount_yuan.rolling(20, min_periods=10).mean().to_numpy(dtype=np.float32)
    arrays["amount20_yuan"] = amount20
    # Alternative liquidity measures.  The replicated component attribution
    # (stage-1 round 2) shows amount20 is the engine of the composite and that
    # vol20 contributes little, so the liquidity measure itself is the object
    # worth varying.
    amount5 = amount_yuan.rolling(5, min_periods=3).mean()
    amount60 = amount_yuan.rolling(60, min_periods=30).mean()
    arrays["amount5_yuan"] = amount5.to_numpy(dtype=np.float32)
    arrays["amount60_yuan"] = amount60.to_numpy(dtype=np.float32)
    amount20_safe = np.clip(np.asarray(amount20, dtype=np.float64), 1.0, None)
    with np.errstate(all="ignore"):
        arrays["amount_ratio_5_20"] = (
            amount5.to_numpy(dtype=np.float64) / amount20_safe
        ).astype(np.float32)
        amihud = np.abs(np.asarray(ret1, dtype=np.float64)) / amount20_safe
        arrays["amihud20"] = (
            pd.DataFrame(amihud).rolling(20, min_periods=10).mean().to_numpy(dtype=np.float32)
        )
    del amount5, amount60, amount20_safe, amount_yuan
    arrays["amount_rank"] = row_pct_rank(amount20, np.isfinite(amount20))
    adtv20 = (frames["volume"] * frames["factor"] * 100.0).rolling(20, min_periods=10).mean()
    arrays["adtv20_shares"] = adtv20.to_numpy(dtype=np.float32)
    del amount20, adtv20, frames

    valid_count = valid.cumsum().to_numpy(dtype=np.int32)
    arrays["valid_count"] = valid_count
    arrays["valid_recent20"] = valid.rolling(cfg.universe.recent_window, min_periods=1).sum().to_numpy(dtype=np.int16)
    del valid

    arrays["entry_signal"] = np.zeros(panel.valid.shape, dtype=bool)
    arrays["entry_score"] = np.full(panel.valid.shape, np.nan, dtype=np.float32)
    arrays["score"] = np.zeros((1, 1), dtype=np.float32)
    arrays["base_pass"] = np.zeros(panel.valid.shape, dtype=bool)

    market = compute_market_state(
        ret=arrays["ret1"],
        close=arrays["close"],
        ma60=arrays["ma60"],
        ma20=arrays["ma20"],
        eligible=np.asarray(panel.valid) & (arrays["valid_count"] >= cfg.market.min_history_days_market),
        cfg=cfg.market,
    )
    # Relative strength versus the equal-weighted market proxy.
    proxy = pd.Series(market.proxy, index=dates)
    proxy_ret60 = (proxy / proxy.shift(60) - 1.0).to_numpy(dtype=np.float32)
    with np.errstate(all="ignore"):
        rs60 = ret60 - proxy_ret60[:, None]
    rs60[~np.isfinite(ret60)] = np.nan
    arrays["rs60"] = np.asarray(rs60, dtype=np.float32)
    diagnostics = {
        "builder": "v2_lean",
        "n_dates": len(dates),
        "n_symbols": len(symbols),
        "first_date": str(dates[0].date()),
        "last_date": str(dates[-1].date()),
    }
    return FeatureStore(panel=panel, cfg=cfg, market=market, arrays=arrays, diagnostics=diagnostics)
