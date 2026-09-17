#!/usr/bin/env python3
"""Per-seed acceptance matrix for the two frontier configurations.

Reads the recorded acceptance checks of every seed of the compliant configuration
(c2) and the return-end configuration (z1) and prints, for each of the seven
contract checks, how many seeds pass - plus the seed-by-seed CAGR / vol / drawdown
/ win rate and the information ratio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ["42", "7", "2024", "11", "23", "101", "777", "31337", "5"]
CHECKS = [
    ("cagr_ge_25pct", "CAGR >= 25%"),
    ("annual_vol_le_10pct", "vol <= 10%"),
    ("max_drawdown_ge_minus10pct", "max DD >= -10%"),
    ("round_trip_win_rate_ge_70pct", "win >= 70%"),
    ("all_complete_quarters_positive", "all quarters positive"),
    ("rolling_252_effective_ge_60pct", "252d freq >= 60%"),
    ("rolling_63_effective_ge_40pct", "63d freq >= 40%"),
]

PANELS = {
    "c2 compliant (vt 0.055)": lambda s: "v2_v_c2_%s" % s,
    "z1 return end (vt 0.14)": lambda s: "v2_z1_%s" % s,
}


def load(name: str):
    path = ROOT / "output" / name / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    nav = data.get("nav") or {}
    rt = data.get("round_trip") or {}
    checks = {c["name"]: bool(c["passed"]) for c in (data.get("acceptance") or {}).get("checks", [])}
    if not nav or not rt:
        return None
    return {
        "cagr": nav.get("cagr"),
        "vol": nav.get("annual_vol"),
        "mdd": nav.get("max_drawdown"),
        "win": rt.get("win_rate"),
        "trades": rt.get("n"),
        "checks": checks,
    }


def main() -> None:
    for label, template in PANELS.items():
        rows = {}
        for seed in SEEDS:
            got = load(template(seed))
            if got:
                rows[seed] = got
        if not rows:
            print("== %s: no runs ==" % label)
            continue
        frame = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "checks"} for k, v in rows.items()}).T
        frame["IR"] = frame.cagr / frame.vol
        print("== %s (%d seeds) ==" % (label, len(frame)))
        print(frame.round(4).to_string())
        print()
        print("%-24s %s" % ("check", "seeds passing"))
        for key, text in CHECKS:
            passed = sum(1 for v in rows.values() if v["checks"].get(key))
            print("%-24s %d / %d" % (text, passed, len(rows)))
        counts = [sum(1 for v in rows.values() if v["checks"].get(k)) for k, _ in CHECKS]
        per_seed = [sum(1 for k, _ in CHECKS if v["checks"].get(k)) for v in rows.values()]
        print("total checks passed: min %d max %d" % (min(per_seed), max(per_seed)))
        print()


if __name__ == "__main__":
    main()
