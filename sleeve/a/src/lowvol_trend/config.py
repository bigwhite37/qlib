"""Configuration objects for the sleeve/a low-volatility trend strategy.

The design document fixes the first-version parameters; keeping them in a
typed object makes it hard to accidentally change a research assumption by
typo.  YAML files only need to contain overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, get_type_hints

import yaml

T = TypeVar("T")

DEFAULT_QLIB_HOME = "/Users/shuzhenyi/code/python/qlib"
DEFAULT_DB_PATH = "/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb"


@dataclass
class DataConfig:
    qlib_home: str = DEFAULT_QLIB_HOME
    db_path: str = DEFAULT_DB_PATH
    memory_limit: str = "2GB"
    temp_directory: str = "/tmp/qlib_lowvol_trend_duckdb"
    start_date: str = "2015-01-05"
    end_date: str = "2026-09-14"
    cache_dir: str = ""
    # Runtime/backtest start.  The design uses 2015 as an audit and warm-up
    # period, so by default trading starts on the first trading day of 2016.
    backtest_start: str = "2016-01-04"
    # Last date used for final out-of-sample reporting.  Data after this date
    # (the local DB currently has one extra day) is ignored.
    evaluation_end: str = "2026-09-14"


@dataclass
class UniverseConfig:
    min_history_days: int = 120
    recent_window: int = 20
    min_recent_bars: int = 18
    min_raw_price: float = 3.0
    # V2 base-universe floor on the cross-sectional 20-day amount rank.  The
    # illiquidity tilt is both the alpha source and the tail risk, so the floor
    # is a research parameter rather than a fixed constant.
    v2_min_amount_rank: float = 0.20
    liquidity_keep_quantile: float = 0.80
    vol_window: int = 60
    vol_keep_quantile: float = 0.70
    require_positive_volume: bool = True


@dataclass
class SignalConfig:
    ma_fast: int = 5
    ma_mid: int = 20
    ma_slow: int = 60
    rs_window: int = 60
    atr_window: int = 20
    drawdown_lookback: int = 20
    max_daily_gain: float = 0.04
    min_drawdown_atr: float = 1.0
    max_drawdown_atr: float = 3.0
    rs_top_quantile: float = 0.70
    score_w_rs: float = 0.50
    score_w_trend_quality: float = 0.25
    score_w_low_downvol: float = 0.25


@dataclass
class MarketConfig:
    ma_window: int = 60
    short_ma_window: int = 20
    breadth_window: int = 60
    short_breadth_window: int = 20
    drop_threshold: float = -0.03
    lookback_5d: int = 5
    extreme_5d_return: float = -0.06
    extreme_breadth: float = 0.20
    extreme_drop_diffusion: float = 0.35
    strong_breadth: float = 0.55
    weak_breadth: float = 0.40
    weak_recovery_gap: float = 0.05
    weak_recovery_lookback: int = 5
    upgrade_confirm_days: int = 2
    min_history_days_market: int = 60


@dataclass
class StrategyConfig:
    max_positions: int = 12
    max_weight: float = 0.08
    risk_per_trade: float = 0.004
    max_new_per_day: int = 2
    stop_atr_mult: float = 2.5
    stop_min: float = 0.04
    stop_max: float = 0.08
    trailing_activate_atr: float = 2.0
    trailing_atr_mult: float = 1.5
    max_hold_days: int = 20
    # Research switches (semantics may be changed during development).
    exit_trend_ma60: bool = True
    exit_trend_ma20: bool = True
    exit_rank: bool = True
    exit_market_extreme: bool = True
    profit_target_pct: float = 0.0
    profit_target_min_hold: int = 1
    rank_exit_min_hold: int = 3
    rank_exit_below_quantile: float = 0.50
    rank_exit_confirm_days: int = 2
    corr_window: int = 60
    corr_threshold: float = 0.80
    vol_target: float = 0.10
    # No-trade bands: existing positions are only trimmed back to the band edge
    # instead of being rescaled every day, which otherwise turns ordinary
    # volatility-estimate noise into constant selling.
    vol_rebalance_band: float = 0.85
    gross_rebalance_band: float = 0.95
    # "scale": shrink every position proportionally (V0 behaviour).
    # "close_weakest": meet a risk reduction by closing whole positions, lowest
    # model score first.  This removes the tiny-partial-trim churn that pays a
    # minimum commission on every shave.
    # When the market ladder drops to a 0% cap: True liquidates the book,
    # False only blocks new entries and lets each position run to its own exit.
    extreme_liquidates: bool = True
    risk_trim_mode: str = "scale"
    # Partial trims below this fraction of NAV are skipped (they pay the minimum
    # commission and add round-trip churn without moving the risk profile).
    min_trim_weight: float = 0.02
    vol_estimate_window: int = 60
    shrink_alpha: float = 0.3
    shrink_target: str = "const_corr"
    drawdown_tiers: Tuple[Tuple[float, float], ...] = ((0.04, 0.50), (0.06, 0.25))
    max_gross: float = 0.95
    min_position_weight: float = 0.01
    # Comparison switches used to build the fixed-position rule baseline.
    use_market_timing: bool = True
    use_vol_target: bool = True
    use_drawdown_control: bool = True
    # Correlation replacement implements the design rule "prefer the higher
    # score stock" when two candidates/holdings have >0.80 recent correlation.
    correlation_replace: bool = True
    # V2 keeps the correlation machinery available but disables it by default:
    # the model already ranks candidates and the design does not ask for a
    # correlation-based replacement rule.
    use_corr_filter: bool = True
    # Correlation-aware sizing: keep every candidate but scale its target weight
    # down when it duplicates exposure already in the book.
    corr_sizing: bool = False
    corr_sizing_strength: float = 1.0
    corr_sizing_base: float = 0.30
    corr_sizing_floor: float = 0.25
    # Head re-ranking: the account only ever buys from the very top of the
    # blended list (see the round-18 breadth experiment), so a second ranking
    # stage applied to the top head_rerank_k candidates can change which single
    # name is bought each day without touching the rest of the list.
    # head_rerank_sign = -1 prefers LOW values of the key, +1 prefers HIGH.
    head_rerank_k: int = 0
    head_rerank_key: str = "vol60"
    head_rerank_sign: float = -1.0
    # Risk-consistent profit target: exit when the price reaches
    # entry + mult * ATR(entry), clamped to a fraction of the entry price.  A
    # flat percentage target is inconsistent across a book whose names differ in
    # volatility by a factor of three - 7.5% is half a standard deviation for one
    # name and a fifth for another.  Expressing the target in ATR makes every
    # position exit after the same number of "normal days".  0 keeps the flat
    # profit_target_pct.
    profit_target_atr_mult: float = 0.0
    profit_target_atr_floor: float = 0.02
    profit_target_atr_cap: float = 0.30
    # Scale-out: when the first profit target is reached, sell this fraction of the
    # position and let the remainder run to profit_target_pct_2 (or the time cap).
    # 0 disables it and reproduces the single-target behaviour.
    scale_out_fraction: float = 0.0
    profit_target_pct_2: float = 0.15
    # Drawdown is normally measured against the all-time peak NAV.  That makes a
    # hard de-risking rule self-locking: once the account is in cash its NAV is
    # frozen, so the drawdown never improves and the rule never releases.  Setting
    # this to N measures the drawdown against the rolling N-day high instead, so a
    # cut-to-cash rule re-enters automatically after N days.  0 keeps the all-time
    # peak.
    drawdown_peak_window: int = 0
    # Risk-balanced re-weighting of the existing sleeve.  New positions are sized
    # inversely to volatility but existing ones are never re-sized, so the book
    # drifts and its risk is dominated by whichever names happen to have run.  When
    # this is on, every held name is re-weighted toward w ~ 1/vol at the current
    # gross, within the per-name caps.
    rebalance_inverse_vol: bool = False
    rebalance_strength: float = 1.0
    # Patience on the entry side.  The book is slot-constrained (11.73 of 12 slots
    # full while thousands of names pass the gate every day), so a slot that frees
    # is filled with that day's best candidate whatever its quality.  When
    # entry_min_score is set, a day whose best blended score is below it is skipped.
    entry_min_score: float = 0.0
    # Trigger the profit target on an intraday touch of the high rather than on the
    # close.  A resting limit order fills as soon as the price reaches it, so this is
    # the realistic semantic; whether it helps the account is an empirical question,
    # because the fill still happens at the next close in this engine.
    profit_trigger_on_high: bool = False


@dataclass
class ExecutionConfig:
    initial_cash: float = 100_000.0
    lot_size: int = 100
    buy_premium: float = 0.03
    participation_rate: float = 0.0005
    volume_lookback: int = 20
    friction_bps_per_side: float = 10.0
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    transfer_fee_rate: float = 0.00001
    stamp_tax_rate_before_2023_08_28: float = 0.001
    stamp_tax_rate_after_2023_08_28: float = 0.0005
    stamp_tax_change_date: str = "2023-08-28"
    unknown_st_conservative: bool = False
    conservative_limit_pct: float = 0.05
    price_tolerance: float = 0.001


@dataclass
class BacktestConfig:
    run_id_prefix: str = "v0"
    output_dir: str = ""
    segments: Tuple[Tuple[str, str, str], ...] = (
        ("dev_2016_2019", "2016-01-01", "2019-12-31"),
        ("validation_2020_2022", "2020-01-01", "2022-12-31"),
        ("oos_2023_2024", "2023-01-01", "2024-12-31"),
        ("final_oos_2025_2026", "2025-01-01", "2026-09-14"),
        ("continuous_oos_2023_2026", "2023-01-01", "2026-09-14"),
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _fill_dataclass(cls: Type[T], values: Mapping[str, Any], path: str = "") -> T:
    """Recursively build a dataclass from a mapping, rejecting unknown keys."""

    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{path or cls.__name__} must be a mapping, got {type(values)!r}")
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config keys at {path or cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        raw = values[f.name]
        hint = hints[f.name]
        if is_dataclass(hint):
            kwargs[f.name] = _fill_dataclass(hint, raw, f"{path}.{f.name}".strip("."))
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


def load_config(path: Optional[str | Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load a YAML config, recursively merged over the built-in defaults.

    Unknown keys are rejected so a typo can not silently change the strategy.
    """

    def _merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for key, value in extra.items():
            if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
                out[key] = _merge(out[key], value)
            else:
                out[key] = value
        return out

    default = _dataclass_to_mapping(Config())
    merged = default
    if path is not None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Config file {path} must contain a mapping at the top level")
        merged = _merge(merged, loaded)
    if overrides:
        merged = _merge(merged, overrides)
    cfg = _fill_dataclass(Config, merged)
    _normalise_config(cfg)
    return cfg


def _dataclass_to_mapping(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_mapping(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return tuple(_dataclass_to_mapping(v) for v in obj)
    return obj


def _normalise_config(cfg: Config) -> None:
    cfg.data.db_path = str(Path(cfg.data.db_path).expanduser())
    cfg.data.qlib_home = str(Path(cfg.data.qlib_home).expanduser())
    if not cfg.data.cache_dir:
        cfg.data.cache_dir = str(Path(__file__).resolve().parents[2] / "cache")
    if not cfg.backtest.output_dir:
        cfg.backtest.output_dir = str(Path(__file__).resolve().parents[2] / "output")
    cfg.strategy.drawdown_tiers = tuple(
        (float(level), float(cap)) for level, cap in cfg.strategy.drawdown_tiers
    )
    cfg.backtest.segments = tuple(
        (str(name), str(start), str(end)) for name, start, end in cfg.backtest.segments
    )


def asdict(cfg: Config) -> Dict[str, Any]:
    return _dataclass_to_mapping(cfg)
