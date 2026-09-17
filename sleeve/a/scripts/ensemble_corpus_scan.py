#!/usr/bin/env python3
"""Corpus-wide search for genuinely diversifying sub-books.

Every account run in output/ that recorded a nav.csv is loaded as a daily return
series.  For every pair the script records the correlation and the information
ratio of the equal-weight combination, so the question "is there any pair in this
corpus whose combination beats both members on IR" can be answered over the whole
search rather than over a handful of hand-picked configurations.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = np.sqrt(252.0)


def load(path: Path) -> "pd.Series | None":
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    col = "datetime" if "datetime" in frame.columns else frame.columns[0]
    if "nav" not in frame.columns:
        return None
    frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame = frame.dropna(subset=[col])
    series = frame.set_index(col)["nav"].astype(float)
    series = series[series.index >= "2019-01-02"]
    if len(series) < 1500:
        return None
    return series.pct_change().dropna()


def stats(returns: np.ndarray) -> tuple:
    years = len(returns) / 252.0
    total = float(np.prod(1.0 + returns))
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    cagr = total ** (1.0 / years) - 1.0
    vol = float(np.std(returns, ddof=1) * ANN)
    ir = cagr / vol if vol > 0 else float("nan")
    return cagr, vol, ir


def main() -> None:
    series = {}
    for path in sorted((ROOT / "output").glob("v2_*/nav.csv")):
        got = load(path)
        if got is not None:
            series[path.parent.name] = got
    print("runs with usable nav:", len(series))
    frame = pd.DataFrame(series).dropna()
    print("aligned matrix:", frame.shape)
    names = list(frame.columns)
    rets = frame.to_numpy(dtype=np.float64)
    own = np.array([stats(rets[:, i]) for i in range(len(names))])

    corr = frame.corr().to_numpy()
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        if corr[i, j] < 0.90:
            combo = 0.5 * (rets[:, i] + rets[:, j])
            cagr, vol, ir = stats(combo)
            rows.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "corr": corr[i, j],
                    "ir_a": own[i, 2],
                    "ir_b": own[j, 2],
                    "ir_combo": ir,
                    "cagr_combo": cagr,
                    "vol_combo": vol,
                    "gain": ir - max(own[i, 2], own[j, 2]),
                }
            )
    table = pd.DataFrame(rows)
    print()
    print("pairs with correlation < 0.90:", len(table))
    if len(table):
        print()
        print("largest IR gain over the better member:")
        print(table.nlargest(10, "gain").round(4).to_string(index=False))
        print()
        print("lowest-correlation pairs:")
        print(table.nsmallest(10, "corr")[["a", "b", "corr", "ir_a", "ir_b", "ir_combo"]].round(4).to_string(index=False))

    print()
    print("correlation distribution over all pairs:")
    flat = corr[np.triu_indices(len(names), 1)]
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        print("  p%-4.0f %.3f" % (q * 100, float(np.quantile(flat, q))))


if __name__ == "__main__":
    main()
