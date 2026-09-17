#!/usr/bin/env python3
"""Run the frozen V0 plan through the design's execution pressure matrix.

Scenarios:
* baseline friction 10bp / 100k
* friction 20bp and 30bp
* conservative unknown-ST limit handling (5% on main boards)
* account sizes 300k and 1M (capacity / minimum-commission effects)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qrun_v0.py"

SCENARIOS = [
    ("v0_base_100k", []),
    ("v0_friction_20bp", ["--friction-bps", "20"]),
    ("v0_friction_30bp", ["--friction-bps", "30"]),
    ("v0_conservative_st", ["--conservative-st"]),
    ("v0_cash_300k", ["--initial-cash", "300000"]),
    ("v0_cash_1000k", ["--initial-cash", "1000000"]),
    ("v0_signal_lag1", ["--signal-lag", "1"]),
]


def _summarize(out_root: Path, rows) -> None:
    lines = [
        "# sleeve/a V0 执行压力矩阵",
        "",
        "| scenario | CAGR | ann_vol | MDD | win_rate | rounds | f63_min | f252_min | accept | orders | fills |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {cagr:.2%} | {annual_vol:.2%} | {max_drawdown:.2%} | {win_rate:.2%} | "
            "{round_trips} | {freq63_min:.3f} | {freq252_min:.3f} | {acceptance} | {orders} | {fills} |".format(**row)
        )
    report = "\n".join(lines) + "\n"
    (out_root / "pressure_matrix.md").write_text(report, encoding="utf-8")
    (out_root / "pressure_matrix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)


def _load_rows(out_root: Path):
    rows = []
    for tag, _ in SCENARIOS:
        path = out_root / tag / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scenario": tag,
                "cagr": summary["nav"]["cagr"],
                "annual_vol": summary["nav"]["annual_vol"],
                "max_drawdown": summary["nav"]["max_drawdown"],
                "win_rate": summary["round_trip"]["win_rate"],
                "round_trips": summary["round_trip"]["n"],
                "freq63_min": summary["frequency"]["rolling_63_min"],
                "freq252_min": summary["frequency"]["rolling_252_min"],
                "acceptance": summary["acceptance"]["passed"],
                "orders": summary["counts"]["orders"],
                "fills": summary["counts"]["fills"],
            }
        )
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated scenario names")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out_root = ROOT / "output" / "pressure"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _summarize(out_root, _load_rows(out_root))
        return
    wanted = {name for name in args.only.split(",") if name}
    rows = []
    for tag, extra in SCENARIOS:
        if wanted and tag not in wanted:
            continue
        out_dir = out_root / tag
        cmd = [
            sys.executable,
            str(RUNNER),
            "--signal",
            "v0",
            "--start",
            "2016-01-04",
            "--end",
            "2026-09-14",
            "--output",
            str(out_dir),
            "--experiment",
            "sleeve_a_pressure",
            "--recorder",
            tag,
        ] + extra
        print("[pressure]", tag, flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "scenario": tag,
                "cagr": summary["nav"]["cagr"],
                "annual_vol": summary["nav"]["annual_vol"],
                "max_drawdown": summary["nav"]["max_drawdown"],
                "win_rate": summary["round_trip"]["win_rate"],
                "round_trips": summary["round_trip"]["n"],
                "freq63_min": summary["frequency"]["rolling_63_min"],
                "freq252_min": summary["frequency"]["rolling_252_min"],
                "acceptance": summary["acceptance"]["passed"],
                "orders": summary["counts"]["orders"],
                "fills": summary["counts"]["fills"],
            }
        )
    _summarize(out_root, rows)


if __name__ == "__main__":
    main()
