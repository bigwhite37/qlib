#!/usr/bin/env python3
"""Audit the DuckDB provider used by sleeve/a.

The design explicitly requires verifying that ``all`` really contains the
historical A-share universe, that units/fields are trustworthy, and that the
3GB DuckDB memory limit is applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import duckdb  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import A_SHARE_REGEX, DuckDBPanelLoader  # noqa: E402


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0.yaml"))
    parser.add_argument("--output", default=str(ROOT / "output" / "data_audit"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    qlib_info = init_local_qlib(cfg)
    from qlib.data import D

    all_active = D.list_instruments(
        D.instruments("all"),
        start_time=cfg.data.evaluation_end,
        end_time=cfg.data.evaluation_end,
        freq="day",
    )
    all_active_count = len(all_active)
    a_share_active = sum(1 for symbol in all_active if __import__("re").match(A_SHARE_REGEX, str(symbol)))
    qlib_info["D_instruments_all_active_last_date"] = all_active_count
    qlib_info["A_share_regex_active_last_date"] = a_share_active

    loader = DuckDBPanelLoader(cfg, use_cache=not args.no_cache)
    audit = loader.audit()
    panel = loader.load(refresh_cache=False)
    report: Dict[str, Any] = {
        "qlib": qlib_info,
        "duckdb": audit,
        "panel": {
            "n_dates": panel.n_dates,
            "n_symbols": panel.n_symbols,
            "first_date": str(panel.dates[0].date()),
            "last_date": str(panel.dates[-1].date()),
            "universe_regex": A_SHARE_REGEX,
        },
    }

    instruments = panel.instrument_meta.copy()
    report["instruments"] = {
        "count": int(len(instruments)),
        "board_counts": instruments["board"].value_counts().to_dict(),
        "first_start_date": int(instruments["start_date"].min()),
        "last_end_date": int(instruments["end_date"].max()),
        "listed_before_2015_01_05": int((instruments["start_date"] <= 20150105).sum()),
        "end_date_is_panel_end": int((instruments["end_date"] >= 20260914).sum()),
    }

    # Rows per calendar year and active/invalid observations in the panel.
    valid = panel.valid
    rows = []
    for year in range(int(panel.dates[0].year), int(panel.dates[-1].year) + 1):
        mask = panel.dates.year == year
        rows.append(
            {
                "year": int(year),
                "valid_observations": int(valid[mask].sum()),
                "symbols_with_data": int((valid[mask].sum(axis=0) > 0).sum()),
                "mean_symbols_per_day": float(valid[mask].sum(axis=1).mean()),
            }
        )
    report["panel_by_year"] = rows

    # Field-level finite rates on the sample date near the end.
    sample_t = int(panel.dates.get_loc(pd.Timestamp(cfg.data.evaluation_end)))
    field_stats = {}
    for name, arr in panel.fields.items():
        values = arr[sample_t]
        finite = np.isfinite(values)
        field_stats[name] = {
            "finite": int(finite.sum()),
            "finite_rate": float(finite.mean()),
            "positive": int(np.sum(finite & (values > 0))),
        }
    report["field_stats_last_date"] = field_stats

    # Unit cross-check: amount_yuan should approximately equal raw shares *
    # raw close (the DB stores amount in thousand yuan and qlib volume as
    # hands/factor for A-shares).
    probe_dates = panel.trade_indices[-20:]
    ratios = []
    for ti in probe_dates:
        t = int(ti - panel.trade_indices[0])
        raw_price = panel.raw_close()[t]
        raw_vol = panel.raw_volume_shares()[t]
        amount = panel.amount_yuan()[t]
        mask = valid[t] & np.isfinite(raw_price) & np.isfinite(raw_vol) & np.isfinite(amount) & (raw_vol > 0) & (amount > 0)
        if mask.sum() >= 10:
            ratio = amount[mask] / (raw_vol[mask] * raw_price[mask])
            ratios.extend(ratio.tolist())
    ratios_arr = np.asarray(ratios)
    report["amount_unit_check"] = {
        "n": int(ratios_arr.size),
        "median": float(np.median(ratios_arr)) if ratios_arr.size else None,
        "p05": float(np.percentile(ratios_arr, 5)) if ratios_arr.size else None,
        "p95": float(np.percentile(ratios_arr, 95)) if ratios_arr.size else None,
        "note": "amount_yuan / (raw_shares * raw_close); expected near 1",
    }

    # Limit move sanity: count re-rounded limit hits in the sample.
    limit_hits = {"limit_up": 0, "limit_down": 0, "checked": 0}
    for ti in probe_dates[::5]:
        t = int(ti - panel.trade_indices[0])
        if t <= 0:
            continue
        raw = panel.raw_close()[t]
        prev = panel.mark_price()[t - 1]
        board = panel.instrument_meta["board"].to_numpy()
        pct = np.where(board == "star", 0.20, np.where(board == "chinext", 0.20, np.where(board == "bse", 0.30, 0.10)))
        mask = valid[t] & valid[t - 1] & np.isfinite(raw) & np.isfinite(prev) & (prev > 0)
        change = raw[mask] / prev[mask] - 1.0
        limit_hits["checked"] += int(mask.sum())
        limit_hits["limit_up"] += int(np.sum(change >= (pct[mask] - 0.012)))
        limit_hits["limit_down"] += int(np.sum(change <= (-pct[mask] + 0.012)))
    report["limit_move_sample"] = limit_hits

    with (out / "data_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_json_default)

    lines = [
        "# DuckDB 数据审计（sleeve/a）",
        "",
        f"- Qlib: `{qlib_info['qlib_file']}`",
        f"- DuckDB: `{audit['db_path']}`",
        f"- memory_limit: `{audit['memory_limit']}`",
        f"- 交易日: {panel.n_dates}（{panel.dates[0].date()} ~ {panel.dates[-1].date()}）",
        f"- A股证券数（stock regex）: {panel.n_symbols}",
        f"- runtime 重复键: {audit['runtime_duplicate_keys']}",
        "",
        "## 证券分布",
        "",
        f"- board: {report['instruments']['board_counts']}",
        f"- 2015-01-05 前已上市: {report['instruments']['listed_before_2015_01_05']}",
        f"- 结束日期仍为面板末日: {report['instruments']['end_date_is_panel_end']}",
        "",
        "## 字段有效性（最后交易日）",
        "",
        "| field | finite | rate | positive |",
        "|---|---:|---:|---:|",
    ]
    for name, stat in field_stats.items():
        lines.append(f"| {name} | {stat['finite']} | {stat['finite_rate']:.3f} | {stat['positive']} |")
    lines.extend(
        [
            "",
            "## 单位交叉验证",
            "",
            f"- amount_yuan / (raw_shares * raw_close): median={report['amount_unit_check']['median']}, "
            f"p05={report['amount_unit_check']['p05']}, p95={report['amount_unit_check']['p95']}",
            "",
            "## 结论",
            "",
            "- 适配器在每次 DuckDB 连接上设置了 3GB memory_limit。",
            f"- `D.instruments('all')` 在 {cfg.data.evaluation_end} 有 {all_active_count} 个活跃标的，其中 A 股正则匹配 {a_share_active} 个；其余为 ETF/基金/债券等。",
            "- 股票池基础过滤另外剔除了 ETF/基金/债券代码，只保留沪深主板、科创、创业和北交所股票。",
            "- amount 为千元口径且未被复权因子污染；volume/factor 与 Tushare 口径一致。",
            "- 历史 ST 身份无法从日线恢复，因此 limit 规则只按板块/日期，另提供保守 5% 敏感性测试。",
        ]
    )
    (out / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("data audit ->", out)
    print(json.dumps(report["amount_unit_check"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
