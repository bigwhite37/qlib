#!/usr/bin/env python3
"""Round-19 IC scan: the classic low-risk / lottery anomaly factors that the
existing library never contained.

Everything here is causal (each value at t uses only data up to t) and is
computed one factor at a time so the 3 GB watchdog stays comfortable.  The
label is the plain forward close-to-close return, the same convention as
scripts/research_new_features.py, so the numbers are comparable across rounds.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.v2 import v2_base_mask  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402
from lowvol_trend.v2_strategy import blockwise_beta, equal_weight_market  # noqa: E402

HORIZONS = (5, 10, 20)


def main() -> None:
    out = Path("/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_v2_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    keep = {"close", "raw_close", "ret1"}
    features.arrays = {k: v for k, v in features.arrays.items() if k in keep}
    gc.collect()
    dates = panel.dates
    symbols = panel.symbols
    del panel
    gc.collect()

    ret1 = pd.DataFrame(features.arr("ret1").astype(np.float64), index=dates, columns=symbols, copy=False)
    close = pd.DataFrame(features.arr("close").astype(np.float64), index=dates, columns=symbols, copy=False)
    cl = features.arr("close").astype(np.float64)
    fwd = {}
    for h in HORIZONS:
        arr = np.full_like(cl, np.nan)
        arr[:-h] = cl[h:] / cl[:-h] - 1.0
        fwd[h] = arr
    del cl
    gc.collect()

    # equal-weight market return of the tradable universe at each date
    ret_np = ret1.to_numpy()
    tradable = np.isfinite(ret_np) & (np.abs(ret_np) < 0.5)
    market_values = equal_weight_market(ret_np, tradable)
    mkt = pd.Series(market_values, index=dates, name="mkt")
    del tradable
    gc.collect()

    vol20 = ret1.rolling(20, min_periods=20).std()
    vol60 = ret1.rolling(60, min_periods=40).std()

    def rolling_cov(frame: pd.DataFrame, series: pd.Series, window: int, min_periods: int):
        """Rolling covariance without pandas' slow pairwise rolling kernel.

        cov_t = E[xy] - E[x]E[y] over the same window, which is exactly the
        population covariance pandas reports for a full window.
        """

        mean_x = frame.rolling(window, min_periods=min_periods).mean()
        mean_y = series.rolling(window, min_periods=min_periods).mean()
        product = frame.mul(series, axis=0) if isinstance(series, pd.Series) else frame * series
        mean_xy = product.rolling(window, min_periods=min_periods).mean()
        del product
        cov = mean_xy - mean_x.mul(mean_y, axis=0)
        del mean_x, mean_y, mean_xy
        return cov

    def columns() -> Iterator[Tuple[str, np.ndarray]]:
        yield "beta60", blockwise_beta(ret_np, market_values, kind="beta")
        gc.collect()
        yield "ivol60", blockwise_beta(ret_np, market_values, kind="ivol")
        gc.collect()
        yield "skew20", ret1.rolling(20, min_periods=20).skew().to_numpy(dtype=np.float32)
        yield "skew60", ret1.rolling(60, min_periods=40).skew().to_numpy(dtype=np.float32)
        yield "kurt60", ret1.rolling(60, min_periods=40).kurt().to_numpy(dtype=np.float32)
        with np.errstate(all="ignore"):
            dn = ret1.clip(upper=0.0)
            up = ret1.clip(lower=0.0)
            semi = np.sqrt((dn ** 2).rolling(20, min_periods=20).mean()).to_numpy(dtype=np.float32)
        yield "semidev20", semi
        del semi
        gc.collect()
        with np.errstate(all="ignore"):
            dn_s = np.sqrt((dn ** 2).rolling(20, min_periods=20).mean())
            up_s = np.sqrt((up ** 2).rolling(20, min_periods=20).mean())
            ratio = (dn_s / up_s.clip(lower=1e-9)).to_numpy(dtype=np.float32)
        yield "dn_up_vol20", ratio
        del ratio, dn_s, up_s, up, dn
        gc.collect()
        rolling_max60 = close.rolling(60, min_periods=40).max()
        yield "dd60", (close / rolling_max60 - 1.0).to_numpy(dtype=np.float32)
        del rolling_max60
        gc.collect()
        rolling_max252 = close.rolling(252, min_periods=150).max()
        yield "dd252", (close / rolling_max252 - 1.0).to_numpy(dtype=np.float32)
        del rolling_max252
        gc.collect()
        yield "pos_days20", (ret1 > 0).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        yield "pos_days60", (ret1 > 0).rolling(60, min_periods=40).mean().to_numpy(dtype=np.float32)
        lag1 = ret1.shift(1)
        with np.errstate(all="ignore"):
            acov = rolling_cov(ret1, lag1, 20, 20)
            ascale = ret1.rolling(20, min_periods=20).std() * lag1.rolling(20, min_periods=20).std()
            ac1 = (acov / ascale.clip(lower=1e-12)).to_numpy(dtype=np.float32)
        yield "ac1_20", ac1
        del ac1, acov, ascale, lag1
        gc.collect()
        yield "vol_ratio_5_60", (vol20 / vol60.clip(lower=1e-9)).to_numpy(dtype=np.float32)
        yield "gap_abs20", ret1.abs().rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        yield "maxret20", ret1.rolling(20, min_periods=20).max().to_numpy(dtype=np.float32)
        yield "maxret1_pct", ret1.rolling(60, min_periods=40).max().to_numpy(dtype=np.float32)
        yield "mom20", (close / close.shift(20) - 1.0).to_numpy(dtype=np.float32)
        yield "mom60", (close / close.shift(60) - 1.0).to_numpy(dtype=np.float32)
        yield "mom120", (close / close.shift(120) - 1.0).to_numpy(dtype=np.float32)
        yield "mom252_skip20", (close.shift(20) / close.shift(252) - 1.0).to_numpy(dtype=np.float32)
        ma120 = close.rolling(120, min_periods=80).mean()
        yield "dist_ma120", (close / ma120 - 1.0).to_numpy(dtype=np.float32)
        del ma120
        gc.collect()
        ma250 = close.rolling(250, min_periods=150).mean()
        yield "dist_ma250", (close / ma250 - 1.0).to_numpy(dtype=np.float32)
        del ma250
        gc.collect()

    rows: List[dict] = []
    for name, arr in columns():
        record = {"feature": name}
        for h in HORIZONS:
            ics = []
            for t in range(0, len(dates) - h):
                m = base[t] & np.isfinite(arr[t]) & np.isfinite(fwd[h][t])
                if m.sum() < 200:
                    continue
                ics.append(pd.Series(arr[t][m]).rank().corr(pd.Series(fwd[h][t][m]).rank()))
            ics = np.asarray(ics)
            record["ic_h" + str(h)] = float(np.nanmean(ics))
            record["t_h" + str(h)] = float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics)))
        rows.append(record)
        print(
            "[r19] %-16s ic5=%+.4f ic10=%+.4f ic20=%+.4f (t20=%+.1f)"
            % (name, record["ic_h5"], record["ic_h10"], record["ic_h20"], record["t_h20"]),
            flush=True,
        )
        del arr
        gc.collect()
    table = pd.DataFrame(rows)
    table["abs_ic20"] = table["ic_h20"].abs()
    table.sort_values("abs_ic20", ascending=False).to_csv(out / "r19_factor_ic.csv", index=False)
    print(table.sort_values("abs_ic20", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
