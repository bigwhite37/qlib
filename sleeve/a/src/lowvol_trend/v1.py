"""V1 LightGBM enhancement trained on the V0 candidate pool.

Design requirements implemented here:

* label: buy at T+1 close, hold 10 trading days, binary target
  ``R(T+1 -> T+11) > 0.5%``.  Missing future prices remain missing (they are
  dropped from training, never converted to class 0).
* model: Qlib ``LGBModel(loss="binary")`` with a small tree configuration.
* rolling protocol: every natural quarter, train on the previous 3 years,
  validate on the immediately previous year, predict the next quarter.
* label crossing: every training label's end date must be strictly before the
  validation segment start, and every validation label must mature before the
  prediction quarter starts.
* features: Alpha158-family features (all kbar/price/rolling families except
  VWAP, which is not trustworthy in this provider) plus the market
  trend/breadth/drop-diffusion indicators.  Features are evaluated vectorised
  on the Qlib panel; see :mod:`lowvol_trend.alpha158_features`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .bootstrap import ensure_local_qlib

ensure_local_qlib()

from qlib.contrib.model.gbdt import LGBModel  # noqa: E402
from qlib.data.dataset import DatasetH  # noqa: E402
from qlib.data.dataset.handler import DataHandlerLP  # noqa: E402
from qlib.data.dataset.loader import StaticDataLoader  # noqa: E402

from .alpha158_features import Alpha158Features  # noqa: E402
from .config import Config  # noqa: E402
from .features import FeatureStore, build_features, row_pct_rank  # noqa: E402

LABEL_THRESHOLD = 0.005
LABEL_HORIZON = 10  # T+1 buy, T+11 label end => 10 trading days held


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


@dataclass
class V1Artifacts:
    predictions: pd.DataFrame  # columns: datetime, instrument, prob
    quarter_metrics: List[Dict[str, Any]]
    feature_names: List[str]
    boundary_checks: List[Dict[str, Any]]
    calibration: pd.DataFrame = None  # columns: quarter, bucket, count, mean_prob, hit_rate


def build_candidate_frame(features: FeatureStore, cfg: Config, include_alpha158: bool = True) -> pd.DataFrame:
    """Build one row per V0 entry candidate with alpha-like features and label."""

    panel = features.panel
    close = features.arr("close")
    n_dates, n_symbols = close.shape
    dates = features.dates
    symbols = features.symbols

    entry = features.arr("entry_signal")
    t_idx, j_idx = np.nonzero(entry)
    n = len(t_idx)
    if n == 0:
        return pd.DataFrame()

    close_df = pd.DataFrame(close, index=dates)
    raw_volume_df = pd.DataFrame(panel.raw_volume_shares(), index=dates)
    amount_df = pd.DataFrame(panel.amount_yuan(), index=dates)

    # ---------------- returns over multiple horizons ----------------
    feature_arrays: Dict[str, np.ndarray] = {}
    for window in (1, 2, 3, 5, 10, 20, 60):
        if window == 1:
            mat = features.arr("ret1")
        else:
            mat = close_df.pct_change(window, fill_method=None).to_numpy(dtype=np.float32)
        feature_arrays[f"ret_{window}"] = mat[t_idx, j_idx]

    # ---------------- moving-average distances ----------------
    close_np = close
    for name, ma_name in (("ma5", "ma5"), ("ma20", "ma20"), ("ma60", "ma60")):
        feature_arrays[f"close_over_{name}"] = _safe_div(close_np, features.arr(ma_name))[t_idx, j_idx] - 1.0
    feature_arrays["ma20_over_ma60"] = _safe_div(features.arr("ma20"), features.arr("ma60"))[t_idx, j_idx] - 1.0
    feature_arrays["ma60_slope5"] = _safe_div(features.arr("ma60"), features.arr("ma60_prev5"))[t_idx, j_idx] - 1.0
    feature_arrays["atr_over_close"] = _safe_div(features.arr("atr20"), close_np)[t_idx, j_idx]
    feature_arrays["drawdown_over_atr"] = _safe_div(features.arr("dist_high20"), features.arr("atr20"))[t_idx, j_idx]

    # ---------------- volatility / trend quality ----------------
    ret_df = close_df.pct_change(fill_method=None)
    vol20 = ret_df.rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)
    feature_arrays["vol_20"] = vol20[t_idx, j_idx]
    feature_arrays["vol_60"] = features.arr("vol60")[t_idx, j_idx]
    feature_arrays["downvol_20"] = features.arr("downvol20")[t_idx, j_idx]
    feature_arrays["trend_quality_60"] = features.arr("trend_quality60")[t_idx, j_idx]
    feature_arrays["dist_high_20"] = features.arr("dist_high20")[t_idx, j_idx]

    # ---------------- RSI(14) ----------------
    delta = close_df.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss + 1e-12)
    rsi = (100.0 - 100.0 / (1.0 + rs)).to_numpy(dtype=np.float32)
    feature_arrays["rsi_14"] = rsi[t_idx, j_idx]

    # ---------------- liquidity ratios ----------------
    adv20 = raw_volume_df.rolling(20, min_periods=20).mean()
    adv60 = raw_volume_df.rolling(60, min_periods=60).mean()
    feature_arrays["volume_over_adv20"] = _safe_div(raw_volume_df.to_numpy(dtype=np.float32), adv20.to_numpy(dtype=np.float32))[t_idx, j_idx]
    feature_arrays["adv20_over_adv60"] = _safe_div(adv20.to_numpy(dtype=np.float32), adv60.to_numpy(dtype=np.float32))[t_idx, j_idx]
    amount20 = amount_df.rolling(20, min_periods=20).mean()
    feature_arrays["amount_over_amount20"] = _safe_div(amount_df.to_numpy(dtype=np.float32), amount20.to_numpy(dtype=np.float32))[t_idx, j_idx]
    feature_arrays["amount20_log"] = np.log1p(np.maximum(0.0, features.arr("amount20_yuan")[t_idx, j_idx]))

    # ---------------- relative strength / V0 score context ----------------
    feature_arrays["rs60"] = features.arr("rs60")[t_idx, j_idx]
    feature_arrays["rs_rank"] = features.arr("rs_rank")[t_idx, j_idx]
    feature_arrays["v0_score"] = features.arr("score")[t_idx, j_idx]
    feature_arrays["score_rank"] = features.arr("score_rank")[t_idx, j_idx]
    feature_arrays["amount_rank"] = features.arr("amount_rank")[t_idx, j_idx]
    feature_arrays["vol_rank"] = features.arr("vol_rank")[t_idx, j_idx]
    feature_arrays["history_days"] = features.arr("valid_count")[t_idx, j_idx]
    feature_arrays["raw_price"] = features.arr("raw_close")[t_idx, j_idx]

    # ---------------- market regime features ----------------
    market = features.market
    feature_arrays["market_cap"] = market.effective_cap[t_idx]
    feature_arrays["market_breadth60"] = market.breadth_60[t_idx]
    feature_arrays["market_breadth20"] = market.breadth_20[t_idx]
    feature_arrays["market_drop_diffusion"] = market.drop_diffusion[t_idx]
    proxy = pd.Series(market.proxy, index=dates)
    feature_arrays["market_ret5"] = (proxy / proxy.shift(5) - 1.0).to_numpy(dtype=np.float32)[t_idx]
    feature_arrays["market_ret20"] = (proxy / proxy.shift(20) - 1.0).to_numpy(dtype=np.float32)[t_idx]
    feature_arrays["market_ret60"] = market.proxy_return_60[t_idx]

    # ---------------- board one-hot ----------------
    boards = panel.instrument_meta["board"].reindex(symbols).fillna("unknown").tolist()
    board_arr = np.asarray(boards, dtype=object)
    for board_name in ("sh_main", "sz_main", "chinext", "star", "bse"):
        feature_arrays[f"board_{board_name}"] = (board_arr[j_idx] == board_name).astype(np.float32)

    # ---------------- Alpha158-family features ----------------
    # Exact operator families from qlib.contrib.data.loader.Alpha158DL,
    # evaluated vectorised on the panel; VWAP features are excluded.
    if include_alpha158:
        feature_arrays.update(Alpha158Features(features).compute(t_idx, j_idx))

    feature_names = sorted(feature_arrays)

    # ---------------- label ----------------
    exec_close = np.full_like(close, np.nan)
    end_close = np.full_like(close, np.nan)
    # Rows are trading dates, columns are instruments; shift along the date axis.
    exec_close[:-1, :] = close[1:, :]
    # Design label: buy at T+1 close and hold 10 trading days, so the label
    # end is T+11 close (Ref($close,-11) / Ref($close,-1) - 1).
    end_close[: -(LABEL_HORIZON + 1), :] = close[LABEL_HORIZON + 1 :, :]
    label_end_index = t_idx + LABEL_HORIZON + 1
    label_ok = label_end_index < n_dates
    label_cont = _safe_div(end_close[t_idx, j_idx], exec_close[t_idx, j_idx])
    label_cont = label_cont - 1.0
    label = np.where(label_ok & np.isfinite(label_cont), (label_cont > LABEL_THRESHOLD).astype(np.float32), np.nan)
    label_end_date = np.where(
        label_ok,
        pd.Series(dates[np.minimum(label_end_index, n_dates - 1)]).dt.strftime("%Y-%m-%d").to_numpy(),
        None,
    )

    index = pd.MultiIndex.from_arrays(
        [dates[t_idx], np.asarray(symbols, dtype=object)[j_idx]],
        names=["datetime", "instrument"],
    )
    columns = pd.MultiIndex.from_product(
        [["feature"], feature_names],
        names=["group", "feature_name"],
    )
    frame = pd.DataFrame(
        np.column_stack([feature_arrays[name] for name in feature_names]).astype(np.float32),
        index=index,
        columns=columns,
    )
    label_series = pd.Series(label, index=index, name="LABEL0")
    frame[("label", "LABEL0")] = label_series
    frame[("meta", "label_end_date")] = label_end_date
    frame[("meta", "label_end_index")] = label_end_index.astype(np.int32)
    return frame.sort_index()


def _split_quarter_frame(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the candidate frame for one rolling task using matured labels."""

    return {}


def rolling_predict(
    frame: pd.DataFrame,
    cfg: Config,
    start_quarter: str = "2020Q1",
    end_quarter: Optional[str] = None,
    min_train_rows: int = 500,
    min_valid_rows: int = 100,
) -> V1Artifacts:
    """Train per natural quarter and return stitched out-of-sample probabilities."""

    if frame.empty:
        return V1Artifacts(pd.DataFrame(columns=["datetime", "instrument", "prob"]), [], [], [])
    frame = frame.sort_index()
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    min_date, max_date = dates.min(), dates.max()
    quarters = pd.period_range(start=start_quarter, end=max_date, freq="Q")
    if end_quarter is not None:
        quarters = quarters[quarters <= pd.Period(end_quarter, freq="Q")]

    predictions: List[pd.DataFrame] = []
    metrics: List[Dict[str, Any]] = []
    boundary_checks: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    feature_names = list(frame["feature"].columns)
    for q in quarters:
        q_start = q.start_time
        q_end = min(q.end_time, max_date)
        if q_start > max_date:
            continue
        train_start = q_start - pd.DateOffset(years=4)
        train_end = q_start - pd.DateOffset(years=1)
        valid_start = train_end
        valid_end = q_start
        dt = frame.index.get_level_values("datetime")
        label_end = pd.to_datetime(frame[("meta", "label_end_date")].astype(str), errors="coerce")
        train_mask = (dt >= train_start) & (dt < train_end) & (label_end < valid_start)
        valid_mask = (dt >= valid_start) & (dt < valid_end) & (label_end < q_start)
        test_mask = (dt >= q_start) & (dt <= q_end)
        train_df = frame.loc[train_mask]
        valid_df = frame.loc[valid_mask]
        test_df = frame.loc[test_mask]
        if len(train_df) < min_train_rows or len(valid_df) < min_valid_rows or test_df.empty:
            metrics.append(
                {
                    "quarter": str(q),
                    "skipped": True,
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "test_rows": int(len(test_df)),
                }
            )
            continue
        # Explicit label-crossing checks.
        train_label_max = pd.to_datetime(train_df[("meta", "label_end_date")].astype(str)).max()
        valid_label_max = pd.to_datetime(valid_df[("meta", "label_end_date")].astype(str)).max()
        boundary_checks.append(
            {
                "quarter": str(q),
                "train_label_end_max": str(train_label_max),
                "valid_start": str(valid_start),
                "train_ok": bool(train_label_max < valid_start),
                "valid_label_end_max": str(valid_label_max),
                "test_start": str(q_start),
                "valid_ok": bool(valid_label_max < q_start),
            }
        )
        if not (train_label_max < valid_start and valid_label_max < q_start):
            raise RuntimeError(f"Label crossing detected for quarter {q}")

        combined = pd.concat([train_df, valid_df, test_df], axis=0).sort_index()
        handler = DataHandlerLP(
            data_loader=StaticDataLoader(combined),
            infer_processors=[],
            learn_processors=[{"class": "DropnaLabel"}],
            shared_processors=[],
            process_type=DataHandlerLP.PTYPE_I,
            drop_raw=False,
        )
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (q_start, q_end),
            },
        )
        model = LGBModel(
            loss="binary",
            num_leaves=15,
            max_depth=4,
            learning_rate=0.03,
            num_boost_round=500,
            early_stopping_rounds=50,
            seed=cfg.v1_seed if hasattr(cfg, "v1_seed") else 42,
            bagging_fraction=0.8,
            feature_fraction=0.8,
            min_data_in_leaf=50,
        )
        try:
            model.fit(dataset, verbose_eval=0)
        except Exception as exc:  # pragma: no cover - model failure must be visible
            metrics.append({"quarter": str(q), "error": repr(exc)})
            continue
        # Calibration diagnostics on the matured validation segment.  Raw
        # probabilities are checked before any use of the 0.60 gate.
        valid_pred = model.predict(dataset, segment="valid")
        if isinstance(valid_pred, pd.DataFrame):
            valid_pred = valid_pred.iloc[:, 0]
        valid_pred = valid_pred.rename("prob")
        valid_label = valid_df[("label", "LABEL0")]
        valid_join = pd.concat([valid_pred, valid_label.rename("label")], axis=1).dropna()
        valid_auc = float("nan")
        valid_base_rate = float(valid_join["label"].mean()) if len(valid_join) else float("nan")
        if len(valid_join) > 20 and valid_join["label"].nunique() > 1:
            from sklearn.metrics import roc_auc_score

            valid_auc = float(roc_auc_score(valid_join["label"].astype(int), valid_join["prob"]))
            try:
                valid_join = valid_join.copy()
                valid_join["bucket"] = pd.qcut(valid_join["prob"], q=10, duplicates="drop")
                for bucket, group in valid_join.groupby("bucket", observed=True):
                    calibration_rows.append(
                        {
                            "quarter": str(q),
                            "bucket": str(bucket),
                            "count": int(len(group)),
                            "mean_prob": float(group["prob"].mean()),
                            "hit_rate": float(group["label"].mean()),
                        }
                    )
            except ValueError:
                pass

        pred = model.predict(dataset, segment="test")
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred = pred.rename("prob").reset_index()
        pred["prob"] = pred["prob"].astype(float)
        predictions.append(pred)
        y_test = test_df[("label", "LABEL0")]
        y_true = y_test.dropna()
        pred_test = pred.set_index(["datetime", "instrument"])["prob"]
        if len(y_true) > 0:
            aligned = pred_test.reindex(y_true.index)
            top_bucket = aligned[aligned >= 0.60]
            hit_top = float((y_true.reindex(top_bucket.index) > 0.5).mean()) if len(top_bucket) else float("nan")
        else:
            hit_top = float("nan")
        metrics.append(
            {
                "quarter": str(q),
                "skipped": False,
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "test_rows": int(len(test_df)),
                "test_label_rows": int(len(y_true)),
                "valid_auc": valid_auc,
                "valid_base_rate": valid_base_rate,
                "valid_prob_mean": float(valid_pred.mean()) if len(valid_pred) else float("nan"),
                "valid_prob_p90": float(valid_pred.quantile(0.9)) if len(valid_pred) else float("nan"),
                "top_bucket_n": int((pred["prob"] >= 0.60).sum()),
                "top_bucket_hit_rate": hit_top,
                "best_iteration": int(getattr(model.model, "best_iteration", 0) or 0)
                if getattr(model, "model", None) is not None
                else None,
            }
        )

    predictions_df = pd.concat(predictions, axis=0, ignore_index=True) if predictions else pd.DataFrame(columns=["datetime", "instrument", "prob"])
    return V1Artifacts(
        predictions=predictions_df,
        quarter_metrics=metrics,
        feature_names=feature_names,
        boundary_checks=boundary_checks,
        calibration=pd.DataFrame(calibration_rows),
    )


def apply_v1_signals(features: FeatureStore, artifacts: V1Artifacts, gate: float = 0.60) -> FeatureStore:
    """Blend V0 score with Rank(prob) and apply the probability gate."""

    arrays = dict(features.arrays)
    preds = artifacts.predictions
    if preds.empty:
        arrays["v1_prob"] = np.full(features.arr("score").shape, np.nan, dtype=np.float32)
        arrays["entry_signal_v1"] = arrays["entry_signal"].copy()
        arrays["entry_score_v1"] = arrays["entry_score"].copy()
        return FeatureStore(panel=features.panel, cfg=features.cfg, market=features.market, arrays=arrays, diagnostics=dict(features.diagnostics))

    preds = preds.copy()
    preds["datetime"] = pd.to_datetime(preds["datetime"])
    prob_matrix = np.full(features.arr("score").shape, np.nan, dtype=np.float32)
    date_to_index = features.date_to_index
    symbol_to_index = features.symbol_to_index
    for row in preds.itertuples(index=False):
        t = date_to_index.get(pd.Timestamp(row.datetime).normalize())
        j = symbol_to_index.get(row.instrument)
        if t is not None and j is not None:
            prob_matrix[t, j] = float(row.prob)

    base = features.arr("base_pass")
    prob_rank = row_pct_rank(prob_matrix, base & np.isfinite(prob_matrix))
    score = features.arr("score")
    entry_signal = features.arr("entry_signal")
    score_v1 = 0.5 * score + 0.5 * prob_rank
    has_model = np.isfinite(prob_matrix)
    signal_v1 = entry_signal & has_model & (prob_matrix >= gate)
    # Dates without model predictions keep the V0 signal and score (2016-2019).
    no_model_row = ~has_model.any(axis=1)
    signal_v1[no_model_row] = entry_signal[no_model_row]
    score_v1[no_model_row] = score[no_model_row]
    arrays["v1_prob"] = prob_matrix
    arrays["entry_signal_v1"] = signal_v1
    arrays["entry_score_v1"] = np.where(signal_v1, score_v1, np.nan).astype(np.float32)
    # The rest of the pipeline consumes ``entry_signal`` / ``entry_score``.
    arrays["entry_signal"] = signal_v1
    arrays["entry_score"] = arrays["entry_score_v1"]
    return FeatureStore(panel=features.panel, cfg=features.cfg, market=features.market, arrays=arrays, diagnostics=dict(features.diagnostics))
