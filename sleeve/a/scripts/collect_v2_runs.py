#!/usr/bin/env python3
"""Collect every V2 account run into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")


def main() -> None:
    rows = []
    for summary in sorted((ROOT / "output").glob("v2_*/summary.json")):
        run_dir = summary.parent
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        nav = data.get("nav")
        rt = data.get("round_trip")
        freq = data.get("frequency")
        if not nav or not rt or not freq:
            continue
        fills_path = run_dir / "fills.csv"
        fee_pct = traded_x = mean_notional = float("nan")
        if fills_path.exists():
            fills = pd.read_csv(fills_path)
            years = max(nav.get("days", 0) / 252.0, 1e-9)
            initial = nav.get("initial_nav", 1.0)
            fee_pct = float(fills["total_fee"].sum() / initial / years)
            traded_x = float(fills["notional"].sum() / initial / years)
            mean_notional = float(fills["notional"].mean())
        checks = {c["name"]: c for c in data.get("acceptance", {}).get("checks", [])}
        pass_flags = {
            "p_cagr": bool(checks.get("cagr_ge_25pct", {}).get("passed")),
            "p_vol": bool(checks.get("annual_vol_le_10pct", {}).get("passed")),
            "p_mdd": bool(checks.get("max_drawdown_ge_minus10pct", {}).get("passed")),
            "p_win": bool(checks.get("round_trip_win_rate_ge_70pct", {}).get("passed")),
            "p_quarters": bool(checks.get("all_complete_quarters_positive", {}).get("passed")),
            "p_f252": bool(checks.get("rolling_252_effective_ge_60pct", {}).get("passed")),
            "p_f63": bool(checks.get("rolling_63_effective_ge_40pct", {}).get("passed")),
        }
        pass_flags["n_pass"] = int(sum(pass_flags.values()))
        rows.append(
            {
                "run": run_dir.name,
                "cagr": nav["cagr"],
                "vol": nav["annual_vol"],
                "mdd": nav["max_drawdown"],
                "ir": nav["cagr"] / nav["annual_vol"] if nav["annual_vol"] else float("nan"),
                "win": rt["win_rate"],
                "trades": rt["n"],
                "avg_trade": rt["avg_return"],
                "hold": rt["avg_hold_days"],
                "f63": freq["rolling_63_min"],
                "f252": freq["rolling_252_min"],
                "neg_q": checks.get("all_complete_quarters_positive", {}).get("value"),
                **pass_flags,
                "fee_pct_per_year": fee_pct,
                "traded_x_nav": traded_x,
                "mean_fill_notional": mean_notional,
            }
        )
    if not rows:
        raise SystemExit("no runs found")
    table = pd.DataFrame(rows).sort_values(["n_pass", "cagr"], ascending=False)
    out = OUT / "account_runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.round(4).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
