"""Vectorised Alpha158-family features for the V1 candidate frame.

The design requires Alpha158, but computing the official expression graph via
Qlib's per-symbol DuckDB feature storage would issue hundreds of thousands of
queries.  This module evaluates the same Alpha158 operator families directly on
the already-loaded panel and only materialises the values at V0 candidate rows.

VWAP features are intentionally excluded (the provider's VWAP is not trusted
for all history).  K-bar, raw-price and all rolling families are implemented:

* KBAR: KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
* price: OPEN0, HIGH0, LOW0
* rolling: ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV,
  IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA,
  VSTD, WVMA, VSUMP, VSUMN, VSUMD

Formulas mirror ``qlib.contrib.data.loader.Alpha158DL``.  RANK/IMAX/IMIN/IMXD
are computed from candidate-row windows because pandas has no vectorised rolling
arg-max operator.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import FeatureStore

ALPHA158_WINDOWS = (5, 10, 20, 30, 60)


class Alpha158Features:
    def __init__(self, features: FeatureStore, windows: Sequence[int] = ALPHA158_WINDOWS):
        self.features = features
        self.panel = features.panel
        self.windows = tuple(int(w) for w in windows)
        self.dates = features.dates

    def compute(self, t_idx: np.ndarray, j_idx: np.ndarray) -> Dict[str, np.ndarray]:
        panel = self.panel
        out: Dict[str, np.ndarray] = {}

        def emit(name: str, values) -> None:
            arr = values.to_numpy(dtype=np.float32) if hasattr(values, "to_numpy") else np.asarray(values, dtype=np.float32)
            out[name] = arr[t_idx, j_idx].astype(np.float32)

        close = pd.DataFrame(panel.field("close"), index=self.dates)
        open_ = pd.DataFrame(panel.field("open"), index=self.dates)
        high = pd.DataFrame(panel.field("high"), index=self.dates)
        low = pd.DataFrame(panel.field("low"), index=self.dates)
        volume = pd.DataFrame(panel.field("volume"), index=self.dates)

        # ---------------- kbar ----------------
        emit("KMID", (close - open_) / open_)
        emit("KLEN", (high - low) / open_)
        emit("KMID2", (close - open_) / (high - low + 1e-12))
        greater = open_.where(open_ >= close, close)
        lesser = open_.where(open_ <= close, close)
        emit("KUP", (high - greater) / open_)
        emit("KUP2", (high - greater) / (high - low + 1e-12))
        emit("KLOW", (lesser - low) / open_)
        emit("KLOW2", (lesser - low) / (high - low + 1e-12))
        emit("KSFT", (2 * close - high - low) / open_)
        emit("KSFT2", (2 * close - high - low) / (high - low + 1e-12))

        # ---------------- price ----------------
        emit("OPEN0", open_ / close)
        emit("HIGH0", high / close)
        emit("LOW0", low / close)

        # ---------------- shared rolling inputs ----------------
        ret1 = close / close.shift(1) - 1.0
        log_volume = np.log(volume + 1.0)
        log_volume_change = np.log(volume / volume.shift(1) + 1.0)
        pos = pd.Series(np.arange(len(self.dates), dtype=np.float64), index=self.dates)
        abs_change = (close - close.shift(1)).abs()
        up_change = (close - close.shift(1)).clip(lower=0.0)
        down_change = (close.shift(1) - close).clip(lower=0.0)
        abs_volume_change = (volume - volume.shift(1)).abs()
        up_volume_change = (volume - volume.shift(1)).clip(lower=0.0)
        down_volume_change = (volume.shift(1) - volume).clip(lower=0.0)
        wvma_input = ret1.abs() * volume

        for w in self.windows:
            mp = w
            emit(f"ROC{w}", close.shift(w) / close)
            emit(f"MA{w}", close.rolling(w, min_periods=mp).mean() / close)
            emit(f"STD{w}", close.rolling(w, min_periods=mp).std() / close)

            # rolling linear regression on a deterministic time index
            mean_x = close.rolling(w, min_periods=mp).mean()
            mean_pos = pos.rolling(w, min_periods=mp).mean()
            mean_xpos = close.mul(pos, axis=0).rolling(w, min_periods=mp).mean()
            covariance = mean_xpos.sub(mean_x.mul(mean_pos, axis=0))
            var_pos = pos.rolling(w, min_periods=mp).var(ddof=0)
            var_x = close.rolling(w, min_periods=mp).var(ddof=0)
            beta = covariance.div(var_pos, axis=0)
            rsquare = covariance.pow(2).div(var_x.mul(var_pos, axis=0) + 1e-12).clip(lower=0.0, upper=1.0)
            residual = close - (mean_x + beta.mul(pos - mean_pos, axis=0))
            emit(f"BETA{w}", beta / close)
            emit(f"RSQR{w}", rsquare)
            emit(f"RESI{w}", residual / close)

            emit(f"MAX{w}", high.rolling(w, min_periods=mp).max() / close)
            emit(f"MIN{w}", low.rolling(w, min_periods=mp).min() / close)
            emit(f"QTLU{w}", close.rolling(w, min_periods=mp).quantile(0.8) / close)
            emit(f"QTLD{w}", close.rolling(w, min_periods=mp).quantile(0.2) / close)
            emit(f"RANK{w}", close.rolling(w, min_periods=mp).rank(pct=True))
            low_min = low.rolling(w, min_periods=mp).min()
            high_max = high.rolling(w, min_periods=mp).max()
            emit(f"RSV{w}", (close - low_min) / (high_max - low_min + 1e-12))

            # candidate-row windows for arg-max based Aroon-style features
            idx = t_idx[:, None] - (w - 1) + np.arange(w)[None, :]
            valid_rows = t_idx >= (w - 1)
            idx_clipped = np.clip(idx, 0, len(self.dates) - 1)
            candidate_cols = j_idx[:, None]
            high_window = panel.field("high")[idx_clipped, candidate_cols]
            low_window = panel.field("low")[idx_clipped, candidate_cols]
            argmax = np.argmax(np.where(np.isfinite(high_window), high_window, -np.inf), axis=1)
            argmin = np.argmin(np.where(np.isfinite(low_window), low_window, np.inf), axis=1)
            imax = (w - 1 - argmax) / float(w)
            imin = (w - 1 - argmin) / float(w)
            imax[~valid_rows] = np.nan
            imin[~valid_rows] = np.nan
            out[f"IMAX{w}"] = imax.astype(np.float32)
            out[f"IMIN{w}"] = imin.astype(np.float32)
            out[f"IMXD{w}"] = (imax - imin).astype(np.float32)

            emit(f"CORR{w}", close.rolling(w, min_periods=mp).corr(log_volume))
            emit(f"CORD{w}", ret1.rolling(w, min_periods=mp).corr(log_volume_change))
            emit(f"CNTP{w}", (close > close.shift(1)).rolling(w, min_periods=mp).mean())
            emit(f"CNTN{w}", (close < close.shift(1)).rolling(w, min_periods=mp).mean())
            emit(f"CNTD{w}", (close > close.shift(1)).rolling(w, min_periods=mp).mean() - (close < close.shift(1)).rolling(w, min_periods=mp).mean())
            denom = abs_change.rolling(w, min_periods=mp).sum() + 1e-12
            emit(f"SUMP{w}", up_change.rolling(w, min_periods=mp).sum() / denom)
            emit(f"SUMN{w}", down_change.rolling(w, min_periods=mp).sum() / denom)
            emit(f"SUMD{w}", (up_change.rolling(w, min_periods=mp).sum() - down_change.rolling(w, min_periods=mp).sum()) / denom)
            emit(f"VMA{w}", volume.rolling(w, min_periods=mp).mean() / (volume + 1e-12))
            emit(f"VSTD{w}", volume.rolling(w, min_periods=mp).std() / (volume + 1e-12))
            emit(
                f"WVMA{w}",
                wvma_input.rolling(w, min_periods=mp).std()
                / (wvma_input.rolling(w, min_periods=mp).mean() + 1e-12),
            )
            volume_denom = abs_volume_change.rolling(w, min_periods=mp).sum() + 1e-12
            emit(f"VSUMP{w}", up_volume_change.rolling(w, min_periods=mp).sum() / volume_denom)
            emit(f"VSUMN{w}", down_volume_change.rolling(w, min_periods=mp).sum() / volume_denom)
            emit(
                f"VSUMD{w}",
                (up_volume_change.rolling(w, min_periods=mp).sum() - down_volume_change.rolling(w, min_periods=mp).sum())
                / volume_denom,
            )

        # Keep only finite float32 values.
        for name in list(out):
            values = out[name].astype(np.float32)
            values[~np.isfinite(values)] = np.nan
            out[name] = values
        return out


def alpha158_feature_names(windows: Sequence[int] = ALPHA158_WINDOWS) -> List[str]:
    names = [
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
        "OPEN0",
        "HIGH0",
        "LOW0",
    ]
    families = [
        "ROC",
        "MA",
        "STD",
        "BETA",
        "RSQR",
        "RESI",
        "MAX",
        "MIN",
        "QTLU",
        "QTLD",
        "RANK",
        "RSV",
        "IMAX",
        "IMIN",
        "IMXD",
        "CORR",
        "CORD",
        "CNTP",
        "CNTN",
        "CNTD",
        "SUMP",
        "SUMN",
        "SUMD",
        "VMA",
        "VSTD",
        "WVMA",
        "VSUMP",
        "VSUMN",
        "VSUMD",
    ]
    for family in families:
        for w in windows:
            names.append(f"{family}{w}")
    return names
