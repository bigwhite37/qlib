#!/usr/bin/env python3
"""Print the swept axes of every V2 run so the search coverage is auditable."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    "predictions", "rank_blend", "composite", "composite_weights", "profit_target",
    "hold", "use_stop", "vol_target", "use_vol_target", "max_positions", "max_new",
    "max_weight", "risk_per_trade", "extreme_exit", "extreme_liquidates",
    "drawdown_tiers", "cap_scale", "corr_sizing", "min_amount_rank", "use_market_timing",
]


def main() -> None:
    rows = []
    for summary in sorted((ROOT / "output").glob("v2_*/summary.json")):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        run = data.get("run", {})
        nav = data.get("nav") or {}
        rt = data.get("round_trip") or {}
        rec = {"run": summary.parent.name}
        for k in KEYS:
            v = run.get(k)
            if k == "predictions" and v:
                v = Path(v).stem.replace("v2m_", "").replace("_predictions", "")
            rec[k] = v
        rec["cagr"] = round(nav.get("cagr", float("nan")), 4)
        rec["vol"] = round(nav.get("annual_vol", float("nan")), 4)
        rec["mdd"] = round(nav.get("max_drawdown", float("nan")), 4)
        rec["win"] = round(rt.get("win_rate", float("nan")), 4)
        rec["n"] = rt.get("n")
        rows.append(rec)
    header = ["run", "cagr", "vol", "mdd", "win", "n"] + [k for k in KEYS if k != "predictions"]
    print("\t".join(header))
    for r in sorted(rows, key=lambda x: -(x["cagr"] or -9)):
        print("\t".join(str(r.get(k, "")) for k in header))


if __name__ == "__main__":
    main()
