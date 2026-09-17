#!/usr/bin/env python3
"""Per-year return, drawdown and win rate for a set of V2 runs."""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/shuzhenyi/code/python/qlib/sleeve/a/output")
RUNS = [
    ("v2_g4", "g4 (old headline: design ladder, tp 3.5%, hold 45)"),
    ("v2_y4", "y4 (old compliance point)"),
    ("v2_v_c2_42", "c2 (RECOMMENDED: tp 7.5%, hold 120, vt 0.055)"),
    ("v2_v_c1_42", "c1 (tp 9%, hold 90: max return at 10% vol)"),
    ("v2_g_r120_42", "r120 (return end: tp 3.5%, hold 120, vt 0.14)"),
]

def yearly(folder: str) -> pd.DataFrame:
    run = ROOT / folder
    nav = pd.read_csv(run / "nav.csv", index_col=0)["nav"]
    nav.index = pd.to_datetime(nav.index)
    rt = pd.read_csv(run / "round_trips.csv")
    rt["exit_date"] = pd.to_datetime(rt["exit_date"])
    rows = []
    prev = float(nav.iloc[0])
    for y in sorted(set(nav.index.year)):
        sub = nav[nav.index.year == y]
        end = float(sub.iloc[-1])
        dd = (sub / sub.cummax() - 1.0).min()
        year_rt = rt[rt["exit_date"].dt.year == y]
        rows.append(
            {
                "year": y,
                "return": end / prev - 1.0,
                "max_dd": float(dd),
                "trades": int(len(year_rt)),
                "win": float((year_rt["profit"] > 0).mean()) if len(year_rt) else float("nan"),
            }
        )
        prev = end
    frame = pd.DataFrame(rows).set_index("year")
    frame.loc["total", "return"] = float(nav.iloc[-1]) / float(nav.iloc[0]) - 1.0
    frame.loc["total", "max_dd"] = float((nav / nav.cummax() - 1.0).min())
    frame.loc["total", "trades"] = int(len(rt))
    frame.loc["total", "win"] = float((rt["profit"] > 0).mean())
    return frame

for folder, label in RUNS:
    frame = yearly(folder)
    print("=== " + label + " (" + folder + ") ===")
    out = frame.copy()
    out["return"] = (out["return"] * 100).round(2)
    out["max_dd"] = (out["max_dd"] * 100).round(2)
    out["win"] = (out["win"] * 100).round(2)
    print(out.to_string())
    print()
