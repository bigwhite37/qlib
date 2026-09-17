# sleeve/a 实现状态与证据索引

本文件记录“设计是否已经落地”和“验收结果如何”，不替代 `reports/acceptance_report.md`。

## 1. 设计条目落地情况

| 设计条目 | 实现位置 | 证据 |
|---|---|---|
| 前一日决策、下一日收盘尝试成交 | `qlib_backtest.py:LowVolTrendStrategy/FrozenOrderGenerator` | `output/v0_final4/execution_audit.json` 中 `orders_execute_next_step`、`buy_quantity_price_is_decision_only` |
| 买入最高价 T 收盘×1.03、数量按最大价/现金/费用/交易单位冻结 | `FrozenOrderGenerator`、`LowVolExchange.check_order` | 审计 `buy_fill_le_frozen_max_price` |
| 不预支同收盘卖出资金 | `SimulatorExecutor(trade_type='parallel')`、订单生成只用 `current.get_cash()` | 审计 `buys_do_not_pre_spend_same_close_sells` |
| 股票池过滤（历史/连续性/原始价/流动性/波动/数据质量） | `features.py:build_features` | `output/v0_final4/daily_decisions.csv`、`data_audit.md` |
| 市场代理、B20/B60、Q 与四状态 | `market.py` | `output/v0_final4/market_daily.csv` |
| 降仓立即、加仓两日确认、弱势修复门槛 | `market.py`、`portfolio.py` | `market_daily.csv` 的 `raw_state/effective_state/entry_allowed` |
| 四组入场条件、规则评分 | `features.py` | `entry_signal` 列见 `signals.csv` |
| 最多 12 只、单股 8%、风险 0.4%、日新增 2 只、逆波动、相关性保留高分 | `portfolio.py` | `output/v0_final4/daily_decisions.csv`、`orders.csv` |
| 止损/趋势/跟踪/时间/排名退出 | `portfolio.py:evaluate_exits` | `output/v0_final4/round_trips.csv` 的 `exit_reason` |
| 目标波动 10% + ShrinkCovEstimator | `portfolio.py:estimate_annual_vol` | 决策表 `pre_vol/scaled_vol` |
| 4%/6% 回撤降仓 | `portfolio.py:drawdown_cap` | `daily_decisions.csv` 的 `drawdown_cap` |
| A 股板块/日期涨跌停、停牌、封板保守不成交 | `qlib_backtest.py:LowVolExchange` | 审计各项；`execution.py:board_limit_pct` 单测 |
| 成交量参与率上限（计划 + 成交日） | `FrozenOrderGenerator`、`LowVolExchange._calc_trade_info_by_order` | 审计 `planned_qty_le_20d_volume_participation`、`fill_qty_le_execution_volume_participation` |
| 历史印花税、最低佣金、过户费、摩擦成本 | `execution.py:compute_fees`、`LowVolExchange._calc_trade_info_by_order` | 单测 `test_fee_schedule...`，成交表 `total_fee` |
| 账户级 NAV/波动/回撤/胜率/滚动 63/252 频率/季度稳定性 | `metrics.py` | `output/v0_final4/summary.json`、`reports/acceptance_report.md` |
| 每日决策、委托成交、完整回合、账户约束表 | `run_qrun_v0.py`、`ledger.py`、`metrics.py` | `output/v0_final4/{daily_decisions,orders,fills,round_trips,constraints_daily}.csv` |
| V0 / 固定仓位基线 / V1 同执行条件比较 | runner 三种模式 | `output/{v0_final4,baseline_full6,v1_alpha158e}` |
| Qlib 可观测性 | `qlib.workflow.R` + `risk_analysis`/`indicator_analysis` | V0 recorder `b92d2b7e80044f12b885575f26b4aee2` 含 pred/report/positions/portfolio_analysis/indicator/orders/fills/round_trips/constraints |
| V1 季度滚动 3年训练/1年验证/1季度预测 | `v1.py:rolling_predict` | `output/v1_alpha158e/v1_quarter_metrics.csv`、`v1_boundary_checks.csv` |
| Alpha158 特征族（去掉 VWAP） | `alpha158_features.py` | 157 个 Alpha158 家族特征，候选池向量化计算 |
| V1 标签 T+1 买入、T+11 结束、缺失不填 0 | `v1.py:build_candidate_frame` | 标签代码与 `v1_boundary_checks.csv` |
| V1 取消标签标准化、只 DropnaLabel | `v1.py:rolling_predict` | `learn_processors=[DropnaLabel]` |
| V1 概率校准检查 | `v1.py`、`output/v1_alpha158e/v1_calibration.csv` | 十分位校准表见验收报告 |
| 成本 10/20/30bp、保守 ST、100k/300k/1M 压力 | `scripts/run_pressure_tests.py` | `output/pressure/pressure_matrix.{md,json}` |
| 信号/退出延迟压力 | `--signal-lag 1`、pressure `v0_signal_lag1` | 已覆盖；订单 T+2 执行延迟仍未实现，报告中列为下一轮补测项 |
| 双段封存/污染声明 | `reports/acceptance_report.md` 第 8 节 | — |

## 2. 关键证据

```text
sleeve/a/output/data_audit/data_audit.{json,md}       DuckDB schema/覆盖/单位/内存
sleeve/a/output/v0_final4/execution_audit.json         逐单时序与账务审计（当前 True）
sleeve/a/output/v0_final4/summary.json                  V0 账户验收
sleeve/a/output/baseline_full6/summary.json            固定仓位基线
sleeve/a/output/v1_alpha158e/summary.json                  V1 原始 0.60 门槛
sleeve/a/output/v1_alpha158e/v1_quarter_metrics.csv        滚动季度训练/验证/测试与 AUC
sleeve/a/output/v1_alpha158e/v1_boundary_checks.csv        标签穿越边界检查
sleeve/a/output/pressure/pressure_matrix.md            执行压力矩阵
sleeve/a/reports/acceptance_report.md                  最终中文验收报告
```

## 3. 当前验收结论

| 方案 | CAGR | 年化波动 | MDD | 完整回合胜率 | 252/63 最低有效持仓 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| V0 | -4.16% | 4.40% | -40.35% | 32.04% | 0.282 / 0.000 | 未通过 |
| 固定仓位基线 | -7.95% | 8.20% | -61.79% | 33.61% | 0.730 / 0.016 | 未通过 |
| V1（Alpha158 + 原始 0.60） | -1.41% | 2.58% | -18.09% | 34.83% | 0.000 / 0.000 | 未通过，按设计保留 V0 |

设计合同要求 25% 年化、波动 ≤10%、MDD ≤10%、回合胜率 ≥70%、每个完整季度为正、
252 日窗口 ≥60% 有效持仓、63 日窗口 ≥40% 有效持仓同时成立。当前只满足波动目标，
区间分段中有个别阶段为正，但联合验收明确失败。

## 4. 结论口径说明

- 账户 NAV 来自 Qlib `report_normal.pkl` 的 `account` 列；CAGR 用实际自然年数计算。
- 完整回合胜率来自原始股数账本（`round_trips.csv`），包含买卖双边费用，不把减仓拆成多个回合。
- 有效持仓频率来自 Qlib `positions_normal` 每日真实持仓，不是计划仓位。
- 2023-2026 已被查看，按设计不再算未见样本；本仓库不据此宣称样本外成功。
