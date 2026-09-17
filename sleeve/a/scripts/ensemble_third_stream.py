#!/usr/bin/env python3
"""Where does the model/composite blend's information ratio actually come from?

The one mechanism that ever tripled IR in this project is the 50/50 blend of the
rolling model and the causal rule composite.  This script measures the individual
members, their correlation, and then searches the whole corpus for a third return
stream that is uncorrelated with BOTH and lifts the three-way blend further.
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
    if frame.empty or "nav" not in frame.columns:
        return None
    col = "datetime" if "datetime" in frame.columns else frame.columns[0]
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
    return cagr, vol, (cagr / vol if vol > 0 else float("nan"))


def main() -> None:
    keys = ["v2_model1", "v2_rule0", "v2_g4", "v2_v_c2_42", "v2_z1_42"]
    base = {}
    for key in keys:
        got = load(ROOT / "output" / key / "nav.csv")
        if got is not None:
            base[key] = got
    frame0 = pd.DataFrame(base).dropna()
    print("== the two ranker components and their blend (single seed 42) ==")
    for key in frame0.columns:
        cagr, vol, ir = stats(frame0[key].to_numpy())
        print("  %-12s CAGR %6.2f%%  vol %6.2f%%  IR %.3f" % (key, cagr * 100, vol * 100, ir))
    print()
    print("  corr(model1, rule0) = %.3f" % frame0["v2_model1"].corr(frame0["v2_rule0"]))
    blend = 0.5 * (frame0["v2_model1"].to_numpy() + frame0["v2_rule0"].to_numpy())
    cagr, vol, ir = stats(blend)
    print("  naive 50/50 of the two raw return streams: CAGR %.2f%% vol %.2f%% IR %.3f" % (cagr * 100, vol * 100, ir))
    print("  (the account blend is a rank blend, not a return blend, which is why it is stronger)")

    print()
    print("== scanning the corpus for a third stream uncorrelated with both ==")
    series = {}
    for path in sorted((ROOT / "output").glob("v2_*/nav.csv")):
        got = load(path)
        if got is not None:
            series[path.parent.name] = got
    frame = pd.DataFrame(series).dropna()
    keep = [c for c in frame.columns if c not in ("v2_model1", "v2_rule0")]
    ref_a = frame["v2_model1"].to_numpy()
    ref_b = frame["v2_rule0"].to_numpy()
    rows = []
    base_two = 0.5 * (ref_a + ref_b)
    c0, v0, ir0 = stats(base_two)
    for name in keep:
        cand = frame[name].to_numpy()
        ca = np.corrcoef(cand, ref_a)[0, 1]
        cb = np.corrcoef(cand, ref_b)[0, 1]
        three = (ref_a + ref_b + cand) / 3.0
        cagr, vol, ir = stats(three)
        rows.append(
            {
                "candidate": name,
                "corr_model": ca,
                "corr_rule": cb,
                "ir_alone": stats(cand)[2],
                "ir_3way": ir,
                "gain_vs_2way": ir - ir0,
            }
        )
    table = pd.DataFrame(rows)
    print("  two-way base IR = %.3f" % ir0)
    print()
    print("  best three-way by IR:")
    print(table.nlargest(12, "ir_3way").round(3).to_string(index=False))
    print()
    print("  lowest mean correlation with both components:")
    table["mean_corr"] = 0.5 * (table.corr_model.abs() + table.corr_rule.abs())
    print(table.nsmallest(12, "mean_corr").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
