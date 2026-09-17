#!/usr/bin/env python3
"""Corrected ranker ablation: do the two arms diversify, or does the rank blend
add something a return blend cannot?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = np.sqrt(252.0)


def series(name: str) -> pd.Series:
    frame = pd.read_csv(ROOT / "output" / name / "nav.csv")
    col = "datetime" if "datetime" in frame.columns else frame.columns[0]
    frame[col] = pd.to_datetime(frame[col])
    s = frame.set_index(col)["nav"].astype(float)
    return s[s.index >= "2019-01-02"].pct_change().dropna()


def stats(r: np.ndarray) -> tuple:
    years = len(r) / 252.0
    cagr = float(np.prod(1.0 + r)) ** (1.0 / years) - 1.0
    vol = float(np.std(r, ddof=1) * ANN)
    return cagr, vol, cagr / vol


def main() -> None:
    names = {
        "composite_only": "v2_ab_composite",
        "model_only": "v2_ab_model",
        "blend50": "v2_ab_blend50",
    }
    frame = pd.DataFrame({k: series(v) for k, v in names.items()}).dropna()
    for key in frame.columns:
        cagr, vol, ir = stats(frame[key].to_numpy())
        print("%-16s CAGR %6.2f%%  vol %6.2f%%  IR %.3f" % (key, cagr * 100, vol * 100, ir))
    print()
    print("return correlation composite vs model: %.3f" % frame["composite_only"].corr(frame["model_only"]))
    print("return correlation blend vs composite: %.3f" % frame["blend50"].corr(frame["composite_only"]))
    print("return correlation blend vs model    : %.3f" % frame["blend50"].corr(frame["model_only"]))
    print()
    two = 0.5 * (frame["composite_only"].to_numpy() + frame["model_only"].to_numpy())
    cagr, vol, ir = stats(two)
    print("50/50 RETURN blend of the two arms : CAGR %.2f%%  vol %.2f%%  IR %.3f" % (cagr * 100, vol * 100, ir))
    cagr, vol, ir = stats(frame["blend50"].to_numpy())
    print("RANK blend account (the headline)  : CAGR %.2f%%  vol %.2f%%  IR %.3f" % (cagr * 100, vol * 100, ir))
    print()
    # how much of the rank blend's edge survives if we simply scale the best arm?
    best = "composite_only"
    cagr_b, vol_b, ir_b = stats(frame[best].to_numpy())
    target_vol = vol
    scale = target_vol / vol_b
    print(
        "scaling the best single arm to the blend's volatility: CAGR %.2f%% (IR %.3f)",
        cagr_b * scale * 100,
        ir_b,
    )


if __name__ == "__main__":
    main()
