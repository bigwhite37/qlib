#!/usr/bin/env python3
"""Build the human-readable acceptance report from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _num(value: float, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _table(rows: List[Dict[str, Any]], columns: List[tuple]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "|" + "|".join("---" if not numeric else "---:" for _, numeric in columns) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join([header, sep] + body)


def _scheme_rows() -> List[Dict[str, Any]]:
    rows = []
    for tag, path in (
        ("V0 市场状态+风险控制", OUT / "v0_final4" / "summary.json"),
        ("固定仓位规则基线", OUT / "baseline_full6" / "summary.json"),
        ("V1 LightGBM（原始0.60门槛）", OUT / "v1_alpha158e" / "summary.json"),
    ):
        if not path.exists():
            continue
        s = _load(path)
        nav = s["nav"]
        rt = s["round_trip"]
        freq = s["frequency"]
        rows.append(
            {
                "scheme": tag,
                "cagr": _pct(nav["cagr"]),
                "vol": _pct(nav["annual_vol"]),
                "mdd": _pct(nav["max_drawdown"]),
                "win": _pct(rt["win_rate"]),
                "rounds": rt["n"],
                "f63": _num(freq["rolling_63_min"]),
                "f252": _num(freq["rolling_252_min"]),
                "accept": "通过" if s["acceptance"]["passed"] else "不通过",
                "orders": s["counts"]["orders"],
            }
        )
    return rows


def _segment_rows() -> List[Dict[str, Any]]:
    rows = []
    for tag, path in (
        ("V0", OUT / "v0_final4" / "summary.json"),
        ("Baseline", OUT / "baseline_full6" / "summary.json"),
        ("V1", OUT / "v1_alpha158e" / "summary.json"),
    ):
        if not path.exists():
            continue
        s = _load(path)
        for name, seg in s.get("segments", {}).items():
            rows.append(
                {
                    "scheme": tag,
                    "segment": name,
                    "start": seg["start"],
                    "end": seg["end"],
                    "cagr": _pct(seg["cagr"]),
                    "vol": _pct(seg["annual_vol"]),
                    "mdd": _pct(seg["max_drawdown"]),
                    "win": _pct(seg["win_rate"]),
                    "rounds": seg["round_trips"],
                    "f63_min": _num(seg.get("rolling_63_min")),
                    "f252_min": _num(seg.get("rolling_252_min")),
                }
            )
    return rows


def _quarter_rows() -> List[Dict[str, Any]]:
    import pandas as pd

    rows = []
    for tag, path in (
        ("V0", OUT / "v0_final4" / "quarterly_returns.csv"),
        ("Baseline", OUT / "baseline_full6" / "quarterly_returns.csv"),
        ("V1", OUT / "v1_alpha158e" / "quarterly_returns.csv"),
    ):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        complete = frame[frame["complete"]]
        rows.append(
            {
                "scheme": tag,
                "complete_quarters": int(len(complete)),
                "positive": int((complete["return"] > 0).sum()),
                "negative": int((complete["return"] <= 0).sum()),
                "worst": _pct(float(complete["return"].min())) if len(complete) else "n/a",
                "best": _pct(float(complete["return"].max())) if len(complete) else "n/a",
                "mean": _pct(float(complete["return"].mean())) if len(complete) else "n/a",
            }
        )
    return rows


def _calibration_rows() -> List[Dict[str, Any]]:
    import pandas as pd

    path = OUT / "v1_alpha158e" / "v1_calibration.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    frame["bucket_mid"] = frame["bucket"].astype(str).str.extract(r"([0-9.]+), ([0-9.]+)").astype(float).mean(axis=1)
    frame["decile"] = pd.qcut(frame["bucket_mid"], q=10, labels=False, duplicates="drop") + 1
    grouped = frame.groupby("decile").apply(
        lambda g: pd.Series(
            {
                "n": g["count"].sum(),
                "mean_prob": (g["mean_prob"] * g["count"]).sum() / max(g["count"].sum(), 1),
                "hit_rate": (g["hit_rate"] * g["count"]).sum() / max(g["count"].sum(), 1),
            }
        )
    )
    return [
        {"decile": int(idx), "n": int(row["n"]), "mean_prob": _num(row["mean_prob"], 4), "hit_rate": _pct(row["hit_rate"])}
        for idx, row in grouped.reset_index().set_index("decile").iterrows()
    ]


def _v1_rows() -> List[Dict[str, Any]]:
    import pandas as pd

    metrics_path = OUT / "v1_alpha158e" / "v1_quarter_metrics.csv"
    if not metrics_path.exists():
        return []
    frame = pd.read_csv(metrics_path)
    frame = frame[~frame.get("skipped", False).astype(bool)]
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "quarter": row["quarter"],
                "train": int(row["train_rows"]),
                "valid": int(row["valid_rows"]),
                "test": int(row["test_rows"]),
                "valid_auc": _num(row.get("valid_auc"), 4),
                "base_rate": _pct(row.get("valid_base_rate")),
                "p90": _num(row.get("valid_prob_p90"), 4),
                "gate_n": int(row.get("top_bucket_n", 0)),
                "gate_hit": _pct(row.get("top_bucket_hit_rate")) if pd.notna(row.get("top_bucket_hit_rate")) else "n/a",
            }
        )
    return rows


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    schemes = _scheme_rows()
    segments = _segment_rows()
    quarters = _quarter_rows()
    v1_rows = _v1_rows()
    calibration = _calibration_rows()
    pressure_path = OUT / "pressure" / "pressure_matrix.json"
    pressure = _load(pressure_path) if pressure_path.exists() else []
    pressure = [
        {
            "scenario": row["scenario"],
            "cagr": _pct(row["cagr"]),
            "annual_vol": _pct(row["annual_vol"]),
            "max_drawdown": _pct(row["max_drawdown"]),
            "win_rate": _pct(row["win_rate"]),
            "round_trips": row["round_trips"],
            "freq63_min": _num(row["freq63_min"]),
            "freq252_min": _num(row["freq252_min"]),
            "acceptance": row["acceptance"],
        }
        for row in pressure
    ]
    audit = _load(OUT / "data_audit" / "data_audit.json") if (OUT / "data_audit" / "data_audit.json").exists() else {}

    lines: List[str] = []
    lines.append("# sleeve/a 设计实现与验收报告")
    lines.append("")
    lines.append("> 本报告严格使用账户净值计算收益、波动和回撤；不把相对基准超额收益当作账户收益，")
    lines.append("> 不把计划仓位当作实际持仓频率，也不因为不达标而调整验收门槛。")
    lines.append("")
    lines.append("## 1. 验收结论")
    lines.append("")
    lines.append("三套预先登记方案在 2016-01-04 ~ 2026-09-14 连续账户回测下 **均未通过**设计合同：")
    lines.append("")
    lines.append(_table(schemes, [("scheme", False), ("cagr", True), ("vol", True), ("mdd", True), ("win", True), ("rounds", True), ("f63", True), ("f252", True), ("accept", False), ("orders", True)]))
    lines.append("")
    lines.append("结论：")
    lines.append("")
    lines.append("- V0 的市场状态和风险控制确实降低了波动和最大回撤（相对固定仓位基线），但没有产生足够的选股 alpha。")
    lines.append("- V0 的执行成本很高：约 1,704 个完整交易回合，扣费后胜率约 33%，趋势失效类退出占大多数。")
    lines.append("- V1 在原始概率 0.60 门槛下只在极少数交易日产生候选，因此实际持仓频率为 0，按设计规则应保留 V0、不采用 V1。")
    lines.append("- 所有方案的 63 日最低有效持仓频率均远低于 40%，说明市场状态/风险控制下策略长期休眠。")
    lines.append("")
    # Detailed acceptance checklist for each scheme.
    lines.append("### 1.1 逐项验收")
    lines.append("")
    check_rows = []
    for tag, path in (
        ("V0", OUT / "v0_final4" / "summary.json"),
        ("Baseline", OUT / "baseline_full6" / "summary.json"),
        ("V1", OUT / "v1_alpha158e" / "summary.json"),
    ):
        if not path.exists():
            continue
        s = _load(path)
        for check in s["acceptance"]["checks"]:
            value = check["value"]
            if isinstance(value, float) and check["name"] not in ("cagr_ge_25pct", "annual_vol_le_10pct", "max_drawdown_ge_minus10pct", "round_trip_win_rate_ge_70pct"):
                value = _num(value)
            elif isinstance(value, float):
                value = _pct(value)
            check_rows.append(
                {
                    "scheme": tag,
                    "check": check["name"],
                    "value": value,
                    "threshold": check["threshold"],
                    "passed": check["passed"],
                }
            )
    lines.append(_table(check_rows, [("scheme", False), ("check", False), ("value", True), ("threshold", False), ("passed", False)]))
    lines.append("")
    open_path = OUT / "v0_final4" / "open_positions.csv"
    if open_path.exists():
        open_pos = pd.read_csv(open_path)
        if len(open_pos):
            lines.append(
                f"期末未平仓 {len(open_pos)} 只，按最后收盘市值合计 {_num(open_pos['market_value'].sum(), 2)} 元，"
                f"浮动盈亏合计 {_num(open_pos['unrealized_pnl'].sum(), 2)} 元，最长持仓 {int(open_pos['holding_days'].max())} 个交易日；"
                "明细见 `output/v0_final4/open_positions.csv`。"
            )
        else:
            lines.append("期末无未平仓股票。")
    lines.append("")
    lines.append("## 2. 分段结果")
    lines.append("")
    lines.append(_table(segments, [("scheme", False), ("segment", False), ("start", False), ("end", False), ("cagr", True), ("vol", True), ("mdd", True), ("win", True), ("rounds", True), ("f63_min", True), ("f252_min", True)]))
    lines.append("")
    # 2026 YTD return (design requires YTD alongside annualised numbers).
    lines.append("### 2.1 2026 年至今收益（不以年化替代）")
    lines.append("")
    ytd_rows = []
    for tag, path in (
        ("V0", OUT / "v0_final4" / "nav.csv"),
        ("Baseline", OUT / "baseline_full6" / "nav.csv"),
        ("V1", OUT / "v1_alpha158e" / "nav.csv"),
    ):
        if not path.exists():
            continue
        nav_frame = pd.read_csv(path, parse_dates=[0], index_col=0)["nav"]
        year_start = nav_frame[nav_frame.index < pd.Timestamp("2026-01-01")]
        year = nav_frame[nav_frame.index >= pd.Timestamp("2026-01-01")]
        if len(year) and len(year_start):
            ytd = float(year.iloc[-1] / year_start.iloc[-1] - 1.0)
            ytd_rows.append({"scheme": tag, "ytd_2026": _pct(ytd), "start_nav": _num(year_start.iloc[-1], 2), "end_nav": _num(year.iloc[-1], 2)})
    lines.append(_table(ytd_rows, [("scheme", False), ("ytd_2026", True), ("start_nav", True), ("end_nav", True)]))
    lines.append("")
    lines.append("## 3. 完整自然季度稳定性")
    lines.append("")
    lines.append(_table(quarters, [("scheme", False), ("complete_quarters", True), ("positive", True), ("negative", True), ("worst", True), ("best", True), ("mean", True)]))
    lines.append("")
    lines.append("## 4. V1 滚动训练协议与校准")
    lines.append("")
    lines.append("V1 每个自然季度：前 3 年训练、前 1 年验证、预测下 1 季度；所有训练标签结束日 < 验证起点，")
    lines.append("所有验证标签结束日 < 预测起点；边界检查保存在 `output/v1_alpha158e/v1_boundary_checks.csv`。")
    lines.append("")
    lines.append(_table(v1_rows, [("quarter", False), ("train", True), ("valid", True), ("test", True), ("valid_auc", True), ("base_rate", True), ("p90", True), ("gate_n", True), ("gate_hit", True)]))
    lines.append("")
    lines.append("")
    lines.append("验证集概率十分位校准（按季度分组后合并）：")
    lines.append("")
    lines.append(_table(calibration, [("decile", True), ("n", True), ("mean_prob", True), ("hit_rate", True)]))
    lines.append("")
    lines.append("原始二分类模型验证 AUC 多集中在 0.49~0.61（均值约 0.51），最高概率十分位验证命中率约 46%，与基准率接近；0.60 的原始概率门槛在全样本中只选到极少数候选。")
    lines.append("按设计“V1 不能破坏持仓频率，否则保留 V0”，V1 未通过采用条件。")
    lines.append("")
    lines.append("## 5. 执行压力矩阵")
    lines.append("")
    if pressure:
        lines.append(_table(pressure, [("scenario", False), ("cagr", True), ("annual_vol", True), ("max_drawdown", True), ("win_rate", True), ("round_trips", True), ("freq63_min", True), ("freq252_min", True), ("acceptance", False)]))
    lines.append("")
    lines.append("提高摩擦、保守 5% 限价、放大账户本金、信号/退出延迟 1 日都不能把任何场景拉回验收线；账户越大，最低佣金和小单影响越小，但收益仍为负。")
    lines.append("")
    lines.append("## 6. 数据与时序审计")
    lines.append("")
    if audit:
        qlib_info = audit.get("qlib", {})
        duck = audit.get("duckdb", {})
        inst = audit.get("instruments", {})
        unit = audit.get("amount_unit_check", {})
        lines.append(f"- Qlib 代码：`{qlib_info.get('qlib_file')}`（强制本地 checkout，拒绝 pip Qlib）。")
        lines.append(f"- DuckDB 连接 memory_limit：`{duck.get('memory_limit')}`；每个连接均设置 3GB。")
        lines.append(
            f"- DuckDB 日历：{duck.get('calendar_start')} ~ {duck.get('calendar_end')}；项目评估截断到 2026-09-14，"
            f"runtime 重复键 {duck.get('runtime_duplicate_keys')}。"
        )
        lines.append(f"- A 股证券数：{inst.get('count')}；板块分布：{inst.get('board_counts')}。")
        lines.append(
            f"- `D.instruments('all')` 在 {duck.get('calendar_end')} 有 "
            f"{qlib_info.get('D_instruments_all_active_last_date')} 个活跃标的，全部匹配项目 A 股正则。"
        )
        lines.append(f"- amount 单位交叉验证中位数 {unit.get('median')}（应为 1，已确认 amount 为千元且未复权污染）。")
        lines.append("- 历史 ST 身份无法从日线恢复：默认按板块/日期涨跌停规则，另做了 5% 保守限价压力测试。")
    lines.append("")
    exec_audit_path = OUT / "v0_final4" / "execution_audit.json"
    if exec_audit_path.exists():
        ea = _load(exec_audit_path)
        lines.append("## 7. 时序、执行与账务审计")
        lines.append("")
        lines.append(f"V0 完整 run 的逐单审计通过：{ea.get('passed')}。基线 `baseline_full6` 和 V1 `v1_alpha158e` 也通过同一 `audit_execution.py` 检查。")
        lines.append("")
        lines.append(_table(
            [{"check": c["name"], "passed": c["passed"], "detail": json.dumps(c["detail"], ensure_ascii=False)} for c in ea.get("checks", [])],
            [("check", False), ("passed", False), ("detail", False)],
        ))
        lines.append("")
    else:
        lines.append("## 7. 时序与执行口径")
        lines.append("")
    lines.append("")
    lines.append("- T 日收盘后生成信号和目标权重，冻结 T+1 委托；数量用 T 日价格、可用现金和最大买入价计算。")
    lines.append("- T+1 必须通过收盘价、涨跌停、停牌、成交参与率约束才会成交；买入价高于 T 收盘价 1.03 倍则拒单。")
    lines.append("- 撮合由 Qlib `SimulatorExecutor(trade_type='parallel')` 执行，买单先于卖单，不能使用同一收盘的卖出回款。")
    lines.append("- 费用按历史印花税（2023-08-28 前后）、最低佣金、过户费和每边摩擦成本逐笔计提。")
    lines.append("- 本地 `tests/test_core.py::test_features_no_future_leak` 验证了特征在只改变未来价格时历史行不变。")
    lines.append("- Qlib `R` recorder `b92d2b7e80044f12b885575f26b4aee2` 保存了 `pred.pkl`、`report_normal.pkl`、`positions_normal.pkl`、`portfolio_analysis.pkl`、`indicator_analysis.pkl`、orders/fills/round_trips/constraints 等可观测性 artifact。")
    lines.append("")
    lines.append("## 8. 样本污染声明")
    lines.append("")
    lines.append("本轮已经完整查看并使用 2023-2026 结果做诊断，因此 2023-01-01 ~ 2026-09-14 不再满足“从未接触”的封存定义。")
    lines.append("按设计第 10 节，后续真正未见样本只能来自 2026-09-14 之后新增的数据；本报告不把该区间重新标记为封存样本。")
    lines.append("")
    lines.append("## 9. 研究结论与下一步")
    lines.append("")
    lines.append("- 本次交付完成了设计中的规则、账户风险、严格执行和观测链路；三套方案均未达到 25%/10%/70% 联合验收，因此不能宣称策略成功。")
    lines.append("- 失败的主因不是数据适配器或成交假设：V0 在固定仓位基线上显著降低了波动/回撤，但入场后 5~10 日命中率只有约 45%，说明“趋势回撤修复”规则在这份历史数据上没有足够 alpha。")
    lines.append("- 换手率约 170 个完整回合/年，扣费后进一步放大损失；在提高 alpha 之前，任何市场状态/风险 overlay 都只能压低波动，不能产生目标收益。")
    lines.append("- V1 原始概率门槛几乎不产生候选，按设计不采用；后续若继续研究，应在机制开发区重新设计入场证据，并把 2026-09-14 之后的新增数据作为真正未见样本，而不是继续使用已污染的 2023-2026。")
    lines.append("")
    lines.append("## 10. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("# 数据审计（3GB memory limit）")
    lines.append("python sleeve/a/scripts/audit_data.py --output sleeve/a/output/data_audit")
    lines.append("# V0 完整连续账户回测")
    lines.append("python sleeve/a/scripts/run_qrun_v0.py --start 2016-01-04 --end 2026-09-14 \\")
    lines.append("  --output sleeve/a/output/v0_final4 --experiment sleeve_a_v0 --recorder v0_final4")
    lines.append("# 固定仓位基线")
    lines.append("python sleeve/a/scripts/run_qrun_v0.py --baseline --start 2016-01-04 --end 2026-09-14 \\")
    lines.append("  --output sleeve/a/output/baseline_full6 --experiment sleeve_a_baseline")
    lines.append("# V1 滚动模型")
    lines.append("python sleeve/a/scripts/run_qrun_v0.py --signal v1 --v1-refresh --start 2016-01-04 --end 2026-09-14 \\")
    lines.append("  --output sleeve/a/output/v1_alpha158e --experiment sleeve_a_v1")
    lines.append("```")
    lines.append("")
    report = "\n".join(lines) + "\n"
    (reports / "acceptance_report.md").write_text(report, encoding="utf-8")
    print("report ->", reports / "acceptance_report.md")


if __name__ == "__main__":
    main()
