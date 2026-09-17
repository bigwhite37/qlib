"""V2 strategy: model-ranked entries with the shared policy exits.

The V2 design keeps Qlib's account, executor and exchange, and only replaces:

* the candidate ranking (two LightGBM models instead of the V0 rule score);
* the single-stock exit policy (one implementation shared with the labels);
* the market ladder (the V2 state machine instead of the V0 ladder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .bootstrap import ensure_local_qlib

ensure_local_qlib()

from .config import Config  # noqa: E402
from .features import FeatureStore  # noqa: E402
from .qlib_backtest import LowVolTrendStrategy  # noqa: E402
from .v2 import V2MarketState  # noqa: E402
from .v2_policy import ExitPolicy  # noqa: E402


# Feature arrays the V2 strategy, ledger, exchange and portfolio constructor
# actually read.  Everything else is released so the whole run fits in a 3 GB
# memory budget.
# Arrays the V2 strategy actually reads at run time.  Everything the lean builder
# computes for the composite or the base mask can be released once those are done.
V2_KEEP_ARRAYS = (
    "close",
    "raw_close",
    "factor",
    "ma20",
    "ma60",
    "atr20",
    "ret1",
    "vol60",
    "amount20_yuan",
    "adtv20_shares",
    "vwap_cv20",
    "vwap_mom20",
    "vwap_dev5",
    "entry_signal",
    "entry_score",
    "composite_rank",
)
# "high" is dropped unless the profit target triggers on an intraday touch (the
# slimming helper checks the flag); "open" and "low" are never read live.
V2_DROP_PANEL_FIELDS = ("open", "low", "amount")


def slim_for_v2(features: FeatureStore) -> FeatureStore:
    """Release feature arrays and panel fields the V2 path never reads."""

    arrays = {k: v for k, v in features.arrays.items() if k in V2_KEEP_ARRAYS}
    # "score" is only read by the correlation filter, which V2 disables.
    arrays["score"] = np.zeros((1, 1), dtype=np.float32)
    # "composite_rank" is only needed when the blend is active.
    if arrays.get("composite_rank") is None:
        arrays.pop("composite_rank", None)
    drop_fields = set(V2_DROP_PANEL_FIELDS)
    if not bool(getattr(features.cfg.strategy, "profit_trigger_on_high", False)):
        # "high" is only read when the profit target triggers on an intraday touch.
        drop_fields.add("high")
    fields = {k: v for k, v in features.panel.fields.items() if k not in drop_fields}
    features.panel.fields = fields
    features.arrays = arrays
    import gc

    gc.collect()
    return features


class V2MarketAdapter:
    """Present the V2 market state through the interface the strategy expects."""

    def __init__(self, state: V2MarketState, median_return: Optional[np.ndarray] = None):
        self.state = state
        self.proxy = state.proxy.astype(np.float64)
        self.effective_state = state.effective_state
        self.raw_state = state.raw_state
        self.effective_cap = state.effective_cap.astype(np.float32)
        self.raw_cap = state.raw_cap.astype(np.float32)
        self.breadth_20 = state.breadth_20
        self.breadth_60 = state.breadth_60
        self.drop_diffusion = state.drop_diffusion
        self.recovery_phase = state.recovery_phase
        self.entry_allowed = self.effective_cap > 1e-12
        self.median_return = median_return
        self.proxy_ma = pd.Series(self.proxy).rolling(60, min_periods=20).mean().to_numpy()
        self.proxy_short_ma = pd.Series(self.proxy).rolling(20, min_periods=10).mean().to_numpy()


def policy_from_config(cfg: Config, **overrides: Any) -> ExitPolicy:
    """Build the frozen V2 exit policy from the strategy config."""

    params: Dict[str, Any] = dict(
        stop_atr_mult=cfg.strategy.stop_atr_mult,
        stop_min=cfg.strategy.stop_min,
        stop_max=cfg.strategy.stop_max,
        use_stop=True,
        profit_target_pct=None,
        profit_target_atr_mult=float(getattr(cfg.strategy, "profit_target_atr_mult", 0.0) or 0.0),
        profit_target_atr_floor=float(getattr(cfg.strategy, "profit_target_atr_floor", 0.02)),
        profit_target_atr_cap=float(getattr(cfg.strategy, "profit_target_atr_cap", 0.30)),
        use_trailing=True,
        trailing_activate_mult=cfg.strategy.trailing_activate_atr / max(cfg.strategy.stop_atr_mult, 1e-9),
        trailing_distance_mult=cfg.strategy.trailing_atr_mult / max(cfg.strategy.stop_atr_mult, 1e-9),
        max_hold_days=cfg.strategy.max_hold_days,
        exit_on_market_extreme=cfg.strategy.exit_market_extreme,
    )
    params.update(overrides)
    return ExitPolicy(**params)


# Rule composites used as the model's ranking partner.  Each entry lists the
# features whose NEGATIVE cross-sectional rank contributes to the score.
COMPOSITE_SPECS = {
    "combo3": ("vol20", "amount20_yuan", "dist_ma60"),
    "combo3p": ("parkinson20", "amount20_yuan", "dist_ma60"),
    "combo4": ("vol20", "parkinson20", "amount20_yuan", "dist_ma60"),
    "combo4l": ("parkinson20", "amount20_yuan", "dist_ma60", "limit_up_20"),
    "combo5l": ("vol20", "parkinson20", "amount20_yuan", "dist_ma60", "limit_up_20"),
    # VWAP family (present in this database; see scripts/audit_vwap_ic.py)
    "combo3v": ("vwap_cv20", "amount20_yuan", "dist_ma60"),
    "combo4v": ("vol20", "amount20_yuan", "dist_ma60", "vwap_cv20"),
    "combo4m": ("vol20", "amount20_yuan", "dist_ma60", "vwap_mom20"),
    "combo5v": ("vol20", "amount20_yuan", "dist_ma60", "vwap_cv20", "vwap_mom20"),
    # Round-19 library: the classic low-risk and lottery anomalies.  Each one
    # replaces vol20 inside combo3, so the account comparison is like for like.
    # Standalone rank IC: scripts/research_r19_factors.py.
    "r19beta": ("beta60", "amount20_yuan", "dist_ma60"),
    "r19ivol": ("ivol60", "amount20_yuan", "dist_ma60"),
    "r19semi": ("semidev20", "amount20_yuan", "dist_ma60"),
    "r19skew": ("skew60", "amount20_yuan", "dist_ma60"),
    "r19dd": ("dd252", "amount20_yuan", "dist_ma60"),
    "r19max": ("maxret20", "amount20_yuan", "dist_ma60"),
    "r19pos": ("pos_days20", "amount20_yuan", "dist_ma60"),
    "r19ac1": ("ac1_20", "amount20_yuan", "dist_ma60"),
    "r19mix": ("vol20", "beta60", "ivol60", "amount20_yuan", "dist_ma60"),
    # Single-component and pair ablations of the winning composite.  The model
    # contributes only ~5% of the information ratio (stage-1 round 1), so the
    # composite is the engine and its three components need to be attributed.
    "solo_vol": ("vol20",),
    "solo_amt": ("amount20_yuan",),
    "solo_ma": ("dist_ma60",),
    "pair_vol_amt": ("vol20", "amount20_yuan"),
    "pair_vol_ma": ("vol20", "dist_ma60"),
    "pair_amt_ma": ("amount20_yuan", "dist_ma60"),
    # Alternative liquidity measures (the replicated attribution says the
    # liquidity component is the engine, so the measure itself is what to vary).
    "solo_amt5": ("amount5_yuan",),
    "solo_amt60": ("amount60_yuan",),
    "solo_amtratio": ("amount_ratio_5_20",),
    "solo_amihud": ("amihud20",),
    "pair_amt5_ma": ("amount5_yuan", "dist_ma60"),
    "pair_amt_ratio_vol": ("amount_ratio_5_20", "vol20"),
}


def equal_weight_market(ret: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Equal-weight market proxy: the mean return of the tradable base universe."""

    with np.errstate(all="ignore"):
        masked = np.where(mask & np.isfinite(ret), ret, np.nan)
        market = np.nanmean(masked, axis=1)
    del masked
    return np.asarray(market, dtype=np.float64)


def blockwise_beta(
    ret: np.ndarray,
    market: np.ndarray,
    kind: str = "beta",
    window: int = 60,
    min_periods: int = 40,
    block: int = 512,
) -> np.ndarray:
    """Rolling market beta (or idiosyncratic volatility) in column blocks.

    A full-frame rolling covariance costs roughly half a gigabyte at this panel
    size, which does not fit the 3 GB budget next to everything else a backtest
    holds.  Working 512 symbols at a time keeps the transient frames at a few
    megabytes and returns exactly the same numbers.
    """

    ret = np.asarray(ret, dtype=np.float64)
    n_dates, n_symbols = ret.shape
    out = np.full((n_dates, n_symbols), np.nan, dtype=np.float32)
    market_frame = pd.Series(np.asarray(market, dtype=np.float64))
    market_mean = market_frame.rolling(window, min_periods=min_periods).mean().to_numpy()
    market_var = market_frame.rolling(window, min_periods=min_periods).var().to_numpy()
    market_std = np.sqrt(market_var)
    for start in range(0, n_symbols, block):
        stop = min(n_symbols, start + block)
        chunk = pd.DataFrame(ret[:, start:stop])
        chunk_mean = chunk.rolling(window, min_periods=min_periods).mean()
        cross = chunk.mul(market_frame, axis=0).rolling(window, min_periods=min_periods).mean()
        cov = (cross - chunk_mean.mul(market_mean, axis=0)).to_numpy()
        if kind == "ivol":
            chunk_std = chunk.rolling(window, min_periods=min_periods).std().to_numpy()
            scale = chunk_std * market_std[:, None]
            with np.errstate(all="ignore"):
                corr = np.where(scale > 0, cov / scale, np.nan)
                values = chunk_std * np.sqrt(np.clip(1.0 - corr ** 2, 0.0, None))
        else:
            with np.errstate(all="ignore"):
                values = np.where(market_var[:, None] > 0, cov / market_var[:, None], np.nan)
        out[:, start:stop] = values.astype(np.float32)
        del chunk, chunk_mean, cross, cov, values
    return out


class RuleFrames:
    """Lazily built rolling frames shared by the rule-composite components.

    Every series here is causal: the value at date t uses returns and prices up
    to and including t only, which is what lets the same composite be used for
    the policy labels, the research sweep and the live strategy.
    """

    def __init__(self, features: FeatureStore, base_mask: np.ndarray) -> None:
        self.features = features
        self.base_mask = base_mask
        self._ret: "pd.DataFrame | None" = None
        self._close: "pd.DataFrame | None" = None
        self._market: "pd.Series | None" = None

    @property
    def ret(self) -> pd.DataFrame:
        if self._ret is None:
            self._ret = pd.DataFrame(
                np.asarray(self.features.arr("ret1"), dtype=np.float64), copy=False
            )
        return self._ret

    @property
    def close(self) -> pd.DataFrame:
        if self._close is None:
            self._close = pd.DataFrame(
                np.asarray(self.features.arr("close"), dtype=np.float64), copy=False
            )
        return self._close

    @property
    def market(self) -> pd.Series:
        """Equal-weight return of the tradable base universe (the market proxy)."""

        if self._market is None:
            self._market = pd.Series(
                equal_weight_market(self.features.arr("ret1"), self.base_mask)
            )
        return self._market

    def values(self, name: str) -> np.ndarray:
        """Raw (unranked) values of one composite component.

        A component name may carry a `_hi` suffix, which only changes the
        direction the composite ranks it in (see `add_composite_score`); the
        values themselves are the same.
        """

        features = self.features
        if name.endswith("_hi"):
            name = name[: -len("_hi")]
        if name == "dist_ma60":
            with np.errstate(all="ignore"):
                return np.asarray(features.arr("close") / features.arr("ma60") - 1.0, dtype=np.float32)
        if name == "vol20":
            return self.ret.rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)
        if name == "skew20":
            return self.ret.rolling(20, min_periods=20).skew().to_numpy(dtype=np.float32)
        if name == "skew60":
            return self.ret.rolling(60, min_periods=40).skew().to_numpy(dtype=np.float32)
        if name == "semidev20":
            with np.errstate(all="ignore"):
                downside = self.ret.clip(upper=0.0)
                out = np.sqrt((downside ** 2).rolling(20, min_periods=20).mean()).to_numpy(dtype=np.float32)
            del downside
            return out
        if name == "dd60":
            close = self.close
            return (close / close.rolling(60, min_periods=40).max() - 1.0).to_numpy(dtype=np.float32)
        if name == "dd252":
            close = self.close
            return (close / close.rolling(252, min_periods=150).max() - 1.0).to_numpy(dtype=np.float32)
        if name == "maxret20":
            return self.ret.rolling(20, min_periods=20).max().to_numpy(dtype=np.float32)
        if name == "pos_days20":
            return (self.ret > 0).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        if name == "ac1_20":
            return self.ret.rolling(20, min_periods=20).corr(self.ret.shift(1)).to_numpy(dtype=np.float32)
        if name in ("beta60", "ivol60"):
            return blockwise_beta(
                self.features.arr("ret1"),
                self.market.to_numpy(),
                kind="ivol" if name == "ivol60" else "beta",
            )
        if name == "mom60":
            close = self.close
            return (close / close.shift(60) - 1.0).to_numpy(dtype=np.float32)
        if name == "dist_ma120":
            close = self.close
            return (close / close.rolling(120, min_periods=80).mean() - 1.0).to_numpy(dtype=np.float32)
        return np.asarray(features.arr(name), dtype=np.float32)


def add_composite_score(
    features: FeatureStore,
    base_mask: np.ndarray,
    spec: str = "combo3",
    weights: "list | tuple | None" = None,
) -> FeatureStore:
    """Store the causal rule composite used as the model's ranking partner.

    The score is the mean cross-sectional rank of the negated components; every
    component is computed only from information up to the signal close and is
    restricted to the light V2 base universe.
    """

    from .features import row_pct_rank

    if spec not in COMPOSITE_SPECS:
        raise KeyError("unknown composite spec " + repr(spec) + "; known=" + str(sorted(COMPOSITE_SPECS)))
    components = COMPOSITE_SPECS[spec]
    if weights:
        if len(weights) != len(components):
            raise ValueError("composite weights must match the number of components")
        component_weights = [float(w) for w in weights]
    else:
        component_weights = [1.0] * len(components)
    frames = RuleFrames(features, base_mask)
    total = None
    weight_sum = 0.0
    for name, component_weight in zip(components, component_weights):
        # A plain component is ranked so that LOW values score high (low
        # volatility, low turnover, below the moving average); a component
        # carrying the _hi suffix is ranked the other way round.
        direction = np.float32(1.0 if name.endswith("_hi") else -1.0)
        values = frames.values(name)
        rank = row_pct_rank(direction * values, base_mask)
        scaled = rank * np.float32(component_weight)
        total = scaled if total is None else total + scaled
        weight_sum += component_weight
        del values, rank, scaled
    composite = (total / float(weight_sum)).astype(np.float32)
    del total, frames
    composite[~base_mask] = np.nan
    features.arrays["composite_rank"] = composite
    return features


def apply_v2_signals(
    features: FeatureStore,
    predictions: pd.DataFrame,
    mu_min: float,
    p_min: float,
    base_mask: np.ndarray,
    rank_by: str = "pred_return",
    blend: float = 0.0,
    extra_rank: "np.ndarray | None" = None,
    extra_weight: float = 0.0,
    force_composite: bool = False,
    blend_mode: str = "mean",
) -> tuple:
    """Turn a prediction table into the entry signal/score matrices.

    The prediction table must contain datetime/instrument plus the model
    columns.  Candidates need both the expected-net-return gate and the
    calibrated win-probability gate; ranking is by expected net return.
    """

    symbol_index = {s: j for j, s in enumerate(features.symbols)}
    n_dates, n_symbols = len(features.dates), len(features.symbols)
    signal = np.zeros((n_dates, n_symbols), dtype=bool)
    score = np.full((n_dates, n_symbols), np.nan, dtype=np.float32)
    empty = FeatureStore(
        panel=features.panel,
        cfg=features.cfg,
        market=features.market,
        arrays={**dict(features.arrays), "entry_signal": signal, "entry_score": score},
        diagnostics=dict(features.diagnostics),
    )
    if predictions is None or predictions.empty:
        return empty, {"n_predictions": 0, "n_signals": 0}
    frame = predictions
    date_values = features.dates.to_numpy(dtype="datetime64[ns]")
    row_dates = pd.to_datetime(frame["datetime"]).to_numpy(dtype="datetime64[ns]")
    t_raw = np.searchsorted(date_values, row_dates)
    t_clipped = np.minimum(t_raw, len(date_values) - 1)
    date_ok = date_values[t_clipped] == row_dates
    if "symbol_index" in frame.columns:
        j_raw = frame["symbol_index"].to_numpy(dtype=np.int64)
    else:
        j_raw = frame["instrument"].map(symbol_index).to_numpy(dtype=np.float64)
    symbol_ok = np.isfinite(j_raw) if j_raw.dtype.kind == "f" else np.ones(len(j_raw), dtype=bool)
    keep = date_ok & symbol_ok
    t_arr = t_clipped[keep]
    j_arr = j_raw[keep].astype(np.int64)
    pred_ret = frame["pred_return"].to_numpy(dtype=np.float64)[keep]
    pred_win = frame["pred_win"].to_numpy(dtype=np.float64)[keep]
    pass_gate = (
        np.isfinite(pred_ret)
        & np.isfinite(pred_win)
        & (pred_ret >= mu_min)
        & (pred_win >= p_min)
        & base_mask[t_arr, j_arr]
    )
    signal[t_arr[pass_gate], j_arr[pass_gate]] = True
    if rank_by in frame:
        rank_col = frame[rank_by].to_numpy(dtype=np.float64)[keep]
    else:
        rank_col = pred_ret
    score[t_arr, j_arr] = np.where(np.isfinite(rank_col), rank_col, pred_ret).astype(np.float32)
    score[~signal] = np.nan
    # Note: with blend == 0 and no extra ranker the composite is *not* used, and the
    # score stays the raw model prediction.  force_composite makes blend == 0 mean
    # "rank by the rule composite alone", which is what a true ablation needs.
    if blend > 0.0 or extra_weight > 0.0 or force_composite:
        composite = features.arrays.get("composite_rank")
        if composite is None:
            raise KeyError("blend requires add_composite_score() to have been called first")
        from .features import row_pct_rank

        model_rank = row_pct_rank(np.where(signal, score, np.nan), signal)
        weight = float(min(max(blend, 0.0), 1.0))
        extra_w = float(min(max(extra_weight, 0.0), 1.0))
        if extra_w > 0.0:
            if extra_rank is None:
                raise KeyError("extra_weight requires an extra_rank matrix")
            blended = weight * model_rank + extra_w * extra_rank + (1.0 - weight - extra_w) * composite
            blended = np.where(
                signal & ~np.isfinite(extra_rank), weight * model_rank + (1.0 - weight) * composite, blended
            )
        elif str(blend_mode) == "min":
            # Hard consensus: a candidate must be liked by BOTH rankers.  The mean
            # blend lets a name ranked first by one arm and 500th by the other win
            # the day; the elementwise minimum does not.
            blended = np.minimum(model_rank, composite)
        else:
            blended = weight * model_rank + (1.0 - weight) * composite
        # Candidates missing the model score (unmatured labels) keep the rule rank.
        blended = np.where(signal & ~np.isfinite(model_rank), composite, blended)
        score = np.where(signal, blended, np.nan).astype(np.float32)
    new_features = FeatureStore(
        panel=features.panel,
        cfg=features.cfg,
        market=features.market,
        arrays={**dict(features.arrays), "entry_signal": signal, "entry_score": score},
        diagnostics=dict(features.diagnostics),
    )
    diagnostics = {
        "n_predictions": int(keep.sum()),
        "n_signals": int(signal.sum()),
        "n_signal_days": int(signal.any(axis=1).sum()),
        "n_candidates_per_day": float(signal.sum(axis=1).mean()),
    }
    return new_features, diagnostics


def prediction_matrices(features: FeatureStore, predictions: pd.DataFrame) -> tuple:
    """Scatter the prediction table into dense (date, symbol) matrices."""

    n_dates, n_symbols = len(features.dates), len(features.symbols)
    rel = np.full((n_dates, n_symbols), np.nan, dtype=np.float32)
    win = np.full((n_dates, n_symbols), np.nan, dtype=np.float32)
    date_index = {d: i for i, d in enumerate(features.dates)}
    t_pos = pd.to_datetime(predictions["datetime"]).map(date_index)
    if "symbol_index" in predictions.columns:
        j_pos = predictions["symbol_index"].astype(np.int64)
    else:
        symbol_index = {s: j for j, s in enumerate(features.symbols)}
        j_pos = predictions["instrument"].map(symbol_index)
    keep = t_pos.notna() & j_pos.notna()
    t_arr = t_pos[keep].to_numpy(dtype=np.int64)
    j_arr = j_pos[keep].to_numpy(dtype=np.int64)
    rel[t_arr, j_arr] = predictions["pred_rel"].to_numpy(dtype=np.float32)[keep.to_numpy()]
    win[t_arr, j_arr] = predictions["pred_win"].to_numpy(dtype=np.float32)[keep.to_numpy()]
    return rel, win


def apply_gate(features: FeatureStore, base_mask: np.ndarray, rel: np.ndarray, win: np.ndarray,
               mu_min: float, p_min: float) -> FeatureStore:
    """Set entry_signal/entry_score from the two model gates."""

    signal = base_mask & np.isfinite(rel) & (rel >= mu_min) & np.isfinite(win) & (win >= p_min)
    score = np.where(signal, rel, np.nan).astype(np.float32)
    return FeatureStore(
        panel=features.panel,
        cfg=features.cfg,
        market=features.market,
        arrays={**dict(features.arrays), "entry_signal": signal, "entry_score": score},
        diagnostics=dict(features.diagnostics),
    )


class V2Strategy(LowVolTrendStrategy):
    """V0 account/execution machinery with V2 policy exits."""

    def __init__(self, *args, policy: Optional[ExitPolicy] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy or policy_from_config(self.cfg)
        self.exit_log: List[Dict[str, Any]] = []

    def _score_percentiles(self, t: int):
        """Cross-sectional percentile rank of the blended entry score on day t."""

        from .features import row_pct_rank

        score_row = self.features.arr("entry_score")[t : t + 1]
        finite = np.isfinite(score_row)
        if not finite.any():
            return None
        ranks = row_pct_rank(np.where(finite, score_row, np.nan), finite)[0]
        return ranks

    def update_observations(self, t: int) -> None:
        close = self.features.arr("close")
        symbol_index = self.features.symbol_to_index
        ranks = None
        if bool(getattr(self.cfg.strategy, "exit_rank", False)):
            ranks = self._score_percentiles(t)
        for symbol, pos in self.ledger.positions.items():
            if pos.shares <= 0 or t <= pos.entry_index:
                continue
            j = symbol_index.get(symbol)
            if j is None:
                continue
            price = close[t, j]
            if np.isfinite(price):
                pos.high_adj = max(float(pos.high_adj), float(price))
                fraction = float(getattr(self.cfg.strategy, "scale_out_fraction", 0.0) or 0.0)
                if (
                    fraction > 0.0
                    and not pos.scaled_out
                    and float(price) >= pos.entry_adj_price * (1.0 + float(self.policy.profit_target_pct))
                ):
                    pos.scaled_out = True
            if ranks is not None:
                rank = ranks[j] if j < len(ranks) else np.nan
                below = (not np.isfinite(rank)) or (
                    rank < float(getattr(self.cfg.strategy, "rank_exit_below_quantile", 0.50))
                )
                if below:
                    pos.below_top_half_days += 1
                else:
                    pos.below_top_half_days = 0

    def _profit_target_pct(self, pos):
        """The position profit target as a fraction of its entry price.

        A flat percentage target means different things for different names -
        7.5% is half a standard deviation for a quiet name and a fifth for a
        volatile one.  When profit_target_atr_mult is set the target is expressed
        in multiples of the position own ATR measured at entry, clamped to a
        fraction of the entry price, so every position exits after the same
        number of normal days.
        """

        base = self.policy.profit_target_pct
        if base is None:
            return None
        if bool(getattr(pos, "scaled_out", False)):
            # After the first target has been banked, the remainder runs to the
            # second target instead.
            return float(getattr(self.cfg.strategy, "profit_target_pct_2", base))
        mult = float(getattr(self.policy, "profit_target_atr_mult", 0.0) or 0.0)
        if mult <= 0 or pos.entry_atr <= 0 or pos.entry_adj_price <= 0:
            return float(base)
        raw_pct = mult * float(pos.entry_atr) / float(pos.entry_adj_price)
        floor = float(getattr(self.policy, "profit_target_atr_floor", 0.02))
        cap = float(getattr(self.policy, "profit_target_atr_cap", 0.30))
        return float(min(max(raw_pct, floor), cap))

    def evaluate_exits(self, t: int) -> Dict[str, str]:
        close = self.features.arr("close")
        symbol_index = self.features.symbol_to_index
        state = self.strategy_market()
        market_extreme = bool(self.policy.exit_on_market_extreme and float(state.effective_cap[t]) <= 1e-12)
        exits: Dict[str, str] = {}
        for symbol, pos in self.ledger.positions.items():
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
            trigger_price = price
            if bool(getattr(self.cfg.strategy, "profit_trigger_on_high", False)):
                high_values = self.features.panel.field("high")
                candidate = high_values[t, j]
                if np.isfinite(candidate):
                    trigger_price = float(candidate)
            lock = float(getattr(self.policy, "profit_lock_pct", 0.0) or 0.0)
            target_pct = self._profit_target_pct(pos)
            if lock and target_pct is not None:
                arm = pos.entry_adj_price * (1.0 + target_pct)
                if pos.high_adj >= arm and price <= pos.high_adj * (1.0 - lock):
                    exits[symbol] = "profit_lock"
                    continue
            elif target_pct is not None and trigger_price >= pos.entry_adj_price * (1.0 + target_pct):
                exits[symbol] = "profit_target"
                continue
            if self.policy.use_stop and price <= pos.entry_adj_price * (1.0 - float(pos.stop_pct)):
                exits[symbol] = "stop_loss"
                continue
            if self.policy.use_trailing:
                arm = pos.entry_adj_price * (1.0 + self.policy.trailing_activate_mult * float(pos.stop_pct))
                if pos.high_adj >= arm and price <= pos.high_adj * (
                    1.0 - self.policy.trailing_distance_mult * float(pos.stop_pct)
                ):
                    exits[symbol] = "trailing_stop"
                    continue
            if (
                bool(getattr(self.cfg.strategy, "exit_rank", False))
                and held_days >= int(getattr(self.cfg.strategy, "rank_exit_min_hold", 3))
                and pos.below_top_half_days >= int(getattr(self.cfg.strategy, "rank_exit_confirm_days", 2))
            ):
                exits[symbol] = "rank_exit"
                continue
            stale_days = int(getattr(self.policy, "exit_below_entry_days", 0) or 0)
            if stale_days and held_days >= stale_days and price < pos.entry_adj_price:
                exits[symbol] = "stale_loser"
                continue
            if held_days >= int(self.policy.max_hold_days):
                exits[symbol] = "time_exit"
                continue
        if exits:
            self.exit_log.append(
                {"decision_index": t, "date": str(self.features.dates[t].date()), "exits": dict(exits)}
            )
        return exits
