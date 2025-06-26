# AI股票选择器 (AI Stock Selector)

基于Qlib + AkShare的智能选股系统，集成机器学习、投资组合优化和风险控制的完整量化投资解决方案。

## 特性

- 🤖 **智能选股**: 基于LightGBM的机器学习模型，整合26个技术指标进行股票预测
- 📊 **数据整合**: 结合Qlib历史数据和AkShare实时数据，支持中国A股市场
- 🎯 **投资组合优化**: 使用CVXPY进行约束优化，自动权重分配
- 🛡️ **风险控制**: 内置止损机制、回撤控制和ST股票过滤
- 📈 **完整回测**: 提供详细的绩效分析和风险指标
- 🔄 **灵活调仓**: 支持日度/周度/月度调仓策略
- 💰 **成本考虑**: 分级手续费计算，贴近真实交易成本
- ⚡ **实时交易模拟**: 本地实时交易模拟，支持实时数据和智能交易时段检测

## 系统要求

- Python 3.8+
- Qlib数据环境
- 网络连接(用于AkShare数据获取)

## 安装

1. 克隆或下载项目文件
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

主要依赖包：
- `pyqlib`: 量化投资框架
- `akshare`: 中国金融数据接口  
- `lightgbm`: 机器学习模型
- `cvxpy`: 凸优化求解器
- `pandas`, `numpy`: 数据处理

## 快速开始

### 1. 数据准备
```bash
# 下载并初始化Qlib数据
python ai_stock_selector.py --prepare_data
```

### 2. 模型训练
```bash
# 训练LightGBM选股模型
python ai_stock_selector.py --train
```

### 3. 生成选股信号
```bash
# 基于最新数据生成预测信号
python ai_stock_selector.py --predict
```

### 4. 选股投资
```bash
# 选出前10只股票用于投资
python ai_stock_selector.py --select_stocks --topk 10
```

### 5. 回测验证
```bash
# 运行完整回测评估策略效果
python ai_stock_selector.py --backtest
```

## 使用方法

### 命令行参数

```bash
python ai_stock_selector.py [选项]

主要操作:
  --prepare_data     下载并准备训练数据
  --train           训练选股模型
  --predict         生成预测信号
  --select_stocks   选股并生成投资组合
  --backtest        运行回测评估
  --live_trade      启动本地实时交易模拟

选股参数:
  --topk N          选择前N只股票 (默认: 10)
  --rebalance_freq  调仓频率: D(日), W(周), M(月) (默认: M)
  --price_min       最低价格限制 (默认: 0)
  --price_max       最高价格限制 (默认: 1e9)
  --weight_method   权重分配方法: equal, score, risk_opt (默认: equal)

实时交易参数:
  --capital         初始资金 (默认: 100000)
  --interval        轮询间隔(秒) (默认: 60)
  --trade_hours     交易时段 (默认: "09:30-11:30,13:00-15:00")

其他选项:
  --start_date      开始日期 (格式: YYYY-MM-DD)
  --end_date        结束日期 (格式: YYYY-MM-DD)
  --universe        股票池: csi300, csi500, all (默认: all)
```

### 使用示例

```bash
# 基础选股 - 选择前20只股票，月度调仓
python ai_stock_selector.py --select_stocks --topk 20 --rebalance_freq M

# 高频策略 - 选择前5只股票，周度调仓
python ai_stock_selector.py --select_stocks --topk 5 --rebalance_freq W

# 价格过滤选股 - 选择价格在10-50元区间的前8只股票
python ai_stock_selector.py --select_stocks --topk 8 --price_min 10 --price_max 50

# 自定义时间范围回测
python ai_stock_selector.py --backtest --start_date 2023-01-01 --end_date 2024-12-31

# 仅在沪深300范围内选股
python ai_stock_selector.py --select_stocks --topk 15 --universe csi300

# 组合策略 - CSI300股票池中价格20-100元的前10只股票
python ai_stock_selector.py --select_stocks --topk 10 --universe csi300 --price_min 20 --price_max 100

# 实时交易模拟 - 10万资金，选择5只股票
python ai_stock_selector.py --live_trade --capital 100000 --topk 5

# 高频实时交易 - 30秒轮询，风险优化权重
python ai_stock_selector.py --live_trade --capital 200000 --topk 8 --interval 30 --weight_method risk_opt

# 仅上午交易时段
python ai_stock_selector.py --live_trade --capital 100000 --topk 5 --trade_hours "09:30-11:30"
```

## 🚀 实时交易模拟 (--live_trade)

### 功能概述

`--live_trade` 提供完整的本地实时交易模拟功能，结合AI模型进行智能选股和自动交易执行。**这是一个纯模拟系统，不会进行真实交易**，主要用于策略验证和模型测试。

### 🎯 目标股票来源详解

**重要说明**: 实时交易的目标股票来源于经过训练的AI机器学习模型，具体流程如下：

#### 1. **AI模型预测**
```python
# 系统首先调用已训练的模型生成预测信号
predictions = self.predict()  # 调用AI模型对全市场股票评分
```

#### 2. **股票筛选管道**
```python
# 获取所有股票的预测评分
stock_universe = predictions['ensemble']  # 使用集成模型结果

# 应用多重筛选条件
filtered_stocks = self.select_stocks(
    pred_scores=stock_universe,     # AI预测评分
    topk=self.topk,                # 选择前N只股票
    date=current_date,             # 当前交易日
    price_min=self.price_min,      # 价格下限过滤
    price_max=self.price_max,      # 价格上限过滤
    weight_method=self.weight_method  # 权重分配方法
)
```

#### 3. **模型依赖性**
⚠️ **必须先训练模型**: 实时交易功能依赖于预先训练好的机器学习模型，使用前必须完成以下步骤：

```bash
# 步骤1: 准备历史数据
python ai_stock_selector.py --prepare_data

# 步骤2: 训练AI选股模型 (必须执行)
python ai_stock_selector.py --train

# 步骤3: 生成预测信号 (可选，实时交易会自动调用)
python ai_stock_selector.py --predict

# 步骤4: 启动实时交易模拟
python ai_stock_selector.py --live_trade --capital 100000 --topk 5
```

#### 4. **股票评分机制**
AI模型基于26个技术指标对**全市场所有股票**进行评分：

- **输入**: 所有A股的实时技术指标数据
- **处理**: LightGBM模型计算每只股票的预期收益评分
- **输出**: 按评分排序，选择前topk只股票作为目标

### 完整使用流程

#### 第一次使用(完整流程)
```bash
# 1. 数据准备 (首次必须执行)
python ai_stock_selector.py --prepare_data
echo "✅ 历史数据下载完成"

# 2. 模型训练 (必须执行)
python ai_stock_selector.py --train
echo "✅ AI选股模型训练完成"

# 3. 启动实时交易模拟
python ai_stock_selector.py --live_trade --capital 100000 --topk 5
echo "🚀 实时交易模拟已启动"
```

#### 日常使用(模型已训练)
```bash
# 如果模型已存在，可直接启动实时交易
python ai_stock_selector.py --live_trade --capital 100000 --topk 5
```

### 核心特性

#### 🤖 **智能选股系统**
- **AI驱动**: 基于LightGBM集成学习模型
- **全市场扫描**: 自动评估所有A股股票
- **动态排序**: 实时更新股票评分排名
- **多因子分析**: 整合26个技术指标进行决策

#### ⏰ **智能交易时段检测**
```python
# 自动识别A股交易时间
trading_hours = "09:30-11:30,13:00-15:00"  # 默认交易时段
is_trading_time()  # 自动判断当前是否为交易时间
is_trading_day()   # 自动排除周末和法定节假日
```

#### 💰 **分级手续费结构**
- **小额交易** (< ¥20,000): 固定收取 ¥5.00 手续费
- **大额交易** (≥ ¥20,000): 按交易金额的 0.025% 收取手续费
- **举例说明**:
  - ¥10,000 交易 → ¥5.00 手续费
  - ¥50,000 交易 → ¥12.50 手续费 (50000 × 0.00025)

#### 🛡️ **严格风险控制**
- **权重限制**: 单只股票持仓不超过20%
- **整手交易**: 严格按100股整手进行交易
- **现金管理**: 实时跟踪现金流和持仓价值
- **止损机制**: 自动风险控制和回撤管理

### 实时数据源

#### 📊 **数据获取渠道**
- **历史数据**: Qlib标准化金融数据
- **实时行情**: AkShare API获取最新股价
- **基础信息**: 股票名称、交易状态等

#### 🔄 **数据更新机制**
```python
# 每个轮询周期的数据获取流程
current_quotes = self.get_realtime_quotes(target_stocks)
# 包含：当前价格、成交量、涨跌幅等实时数据
```

### 输出文件详解

实时交易过程中会在 `output/` 目录生成以下文件：

#### 📝 **交易日志 (trade_log.csv)**
```csv
date,time,code,price,qty,action,cash,position_value
2024-06-19,09:30:15,SH600000,20.50,1000,BUY,20005.00,20500.00
2024-06-19,09:31:30,SZ000001,15.80,500,SELL,7895.00,15800.00
```
记录每笔交易的详细信息，包括时间、股票代码、价格、数量、买卖方向、手续费等。

#### 📈 **投资组合净值 (portfolio_nav.csv)**
```csv
date,time,nav,drawdown,turnover,cash,position_value
2024-06-19,09:30:00,100000.00,0.0000,0.0000,100000.00,0.00
2024-06-19,09:35:00,100250.00,0.0000,0.5800,45250.00,55000.00
```
实时记录投资组合净值变化、回撤情况、换手率等关键指标。

### 交互式可视化

#### 📊 **实时监控面板**
系统会自动生成6个子图的交互式Plotly可视化面板：

1. **净值曲线**: 实时组合价值变化
2. **持仓分布**: 当前股票持仓饼图
3. **回撤分析**: 最大回撤和回撤恢复
4. **交易活动**: 买卖交易时间线
5. **现金流**: 现金与持仓价值对比
6. **换手率**: 组合调整频率

```bash
# 可视化文件会自动生成
output/live_trade_dashboard_YYYYMMDD_HHMMSS.html
```

### 高级配置

#### 🎛️ **自定义参数**
```bash
# 完整配置示例
python ai_stock_selector.py --live_trade \
    --capital 200000 \              # 初始资金20万
    --topk 8 \                     # 选择8只股票
    --interval 30 \                # 30秒轮询
    --weight_method risk_opt \     # 风险优化权重
    --trade_hours "09:30-11:30" \  # 仅上午交易
    --price_min 10 \               # 最低价格10元
    --price_max 100                # 最高价格100元
```

#### 🔧 **权重分配方法**
- **equal**: 等权重分配 (默认)
- **score**: 基于AI评分的权重分配
- **risk_opt**: 风险优化权重分配 (推荐)

### 安全提醒

> ⚠️ **重要声明**: 
> 1. 这是**纯模拟交易系统**，不会执行真实交易
> 2. 仅用于**策略验证和模型测试**
> 3. 实际投资前请充分评估风险
> 4. 模型预测**不构成投资建议**

### 故障排除

#### 常见问题解决

**Q: "模型文件不存在" 错误**
```bash
# 解决方案：先训练模型
python ai_stock_selector.py --train
```

**Q: "无法获取实时数据" 错误**
```bash
# 检查网络连接和AkShare服务状态
# 重试或稍后再试
```

**Q: "选股结果为空" 问题**
```bash
# 可能原因：价格过滤条件过严或市场休市
# 调整 --price_min 和 --price_max 参数
```

## 输出文件

系统会在 `output/` 目录下生成以下文件：

### 选股结果
- `latest_stock_picks.csv`: 最新选股结果
```csv
rank,stock_code,stock_name,pred_score,weight,date
1,SZ000063,中兴通讯,0.116,0.200,2024-12-31
2,SZ300442,润泽科技,0.110,0.200,2024-12-31
...
```

### 预测信号
- `signal_lgb.csv`: 所有股票的预测分数
```csv
date,instrument,prediction
2024-01-02,SH600000,0.029
2024-01-02,SH600009,0.006
...
```

### 回测结果
- `backtest_metrics_lgb.json`: 详细回测指标
```json
{
  "ic_mean": 0.036,
  "ic_ir": 0.233,
  "annual_return": 0.274,
  "max_drawdown": -0.076,
  "sharpe_ratio": 1.45,
  "win_rate": 0.548
}
```

### 实时交易文件
- `trade_log.csv`: 实时交易记录
```csv
date,time,code,price,qty,action,cash,position_value
2024-06-19,09:30:15,SH600000,20.50,1000,BUY,20005.00,20500.00
2024-06-19,09:31:30,SZ000001,15.80,500,SELL,7895.00,15800.00
```

- `portfolio_nav.csv`: 投资组合净值跟踪
```csv
date,time,nav,drawdown,turnover,cash,position_value
2024-06-19,09:30:00,100000.00,0.0000,0.0000,100000.00,0.00
2024-06-19,09:35:00,100250.00,0.0000,0.5800,45250.00,55000.00
```

- `live_trade_dashboard_YYYYMMDD_HHMMSS.html`: 交互式可视化面板

## 系统架构

### 核心模块

#### 1. AIStockSelector类
主要功能类，包含以下核心方法：

- `prepare_qlib_data()`: 数据下载和预处理
- `create_dataset()`: 特征工程和数据集构建
- `train_model()`: 模型训练和验证
- `predict()`: 生成预测信号
- `select_stocks()`: 选股和组合构建
- `backtest()`: 回测和性能评估
- `live_trade()`: 实时交易模拟执行

#### 2. 特征工程
系统使用26个技术指标，分为三大类：

**价格动量类** (8个):
- ROC: 变动率指标
- RSI: 相对强弱指数  
- WILLR: 威廉指标
- Aroon: 阿隆指标
- PSY: 心理线指标
- BIAS: 乖离率
- BOP: 均势指标
- CCI: 商品通道指数

**估值质量类** (9个):
- PE: 市盈率
- PB: 市净率
- PS: 市销率
- PCF: 市现率
- ROE: 净资产收益率
- ROA: 资产收益率
- ROIC: 投入资本回报率
- 盈利增长率和营收增长率

**成交量波动类** (9个):
- VSTD: 成交量标准差
- VMA: 成交量移动平均
- ATR: 真实波动率
- BETA: 贝塔系数
- CMRA: 累积多期收益波动
- 各种收益率和相关性指标

#### 3. 风险控制
- **ST股票过滤**: 自动排除ST、*ST等风险股票
- **单只股票权重限制**: 单只股票权重不超过15%
- **止损机制**: 单日亏损超过1.5%时强制平仓
- **最大回撤控制**: 回撤超过6%时减仓
- **流动性要求**: 过滤成交量不足的股票

#### 4. 投资组合优化
使用CVXPY求解器进行约束优化：

```python
# 目标函数: 最大化预期收益
objective = cp.Maximize(pred_scores.values @ weights)

# 约束条件
constraints = [
    cp.sum(weights) == 1,           # 权重和为1
    weights >= 0,                   # 多头策略
    weights <= max_weight,          # 单股权重限制
]
```

### 数据流程

#### 历史回测流程
1. **数据获取**: Qlib下载历史数据 + AkShare获取股票名称
2. **特征构建**: 计算26个技术指标特征
3. **模型训练**: 使用LightGBM训练预测模型
4. **信号生成**: 对所有股票生成预测分数
5. **组合优化**: 基于预测分数构建最优投资组合
6. **风险控制**: 应用各种风险管理规则
7. **回测验证**: 计算策略历史表现

#### 实时交易流程
1. **模型加载**: 加载预训练的AI选股模型
2. **实时预测**: 对全市场股票进行实时评分
3. **智能选股**: 基于AI评分选择topk只股票
4. **权重分配**: 使用指定方法分配投资权重
5. **交易执行**: 自动执行买卖交易指令
6. **风险监控**: 实时监控持仓和风险指标
7. **记录追踪**: 保存交易记录和净值变化
8. **可视化**: 生成实时交易监控面板

## 性能指标

系统通过以下指标评估策略效果：

### 预测能力
- **IC (Information Coefficient)**: 预测准确性
- **IC_IR**: IC信息比率，衡量预测稳定性
- **RankIC**: 排序预测能力

### 投资回报
- **年化收益率**: 策略年化回报
- **夏普比率**: 风险调整后收益
- **最大回撤**: 最大资产损失幅度
- **胜率**: 盈利交易占比

### 交易特征
- **平均换手率**: 组合调整频率
- **交易次数**: 总交易笔数
- **手续费成本**: 交易成本占比

## 配置说明

可在脚本开头的 `CONFIG` 字典中调整系统参数：

```python
CONFIG = {
    'data_path': '~/.qlib/qlib_data/cn_data',  # Qlib数据路径
    'model_path': './models',                   # 模型保存路径
    'output_path': './output',                  # 输出文件路径
    'start_date': '2014-01-01',                # 数据开始日期
    'end_date': '2025-06-04',                  # 数据结束日期
    'train_end': '2023-12-31',                 # 训练数据结束日期
    'universe': 'all',                         # 股票池范围
    'random_seed': 42,                         # 随机种子
    'n_jobs': 8                                # 并行任务数
}
```

## 测试

运行单元测试验证系统功能：

```bash
cd tests/
python test_ai_stock_selector.py    # 基础功能测试
python test_live_trade.py          # 实时交易测试
```

测试覆盖：
- 选股器初始化
- 股票选择功能
- 投资组合优化
- 调仓日期计算
- 交易费用计算
- 回测指标验证
- 实时交易模拟
- 交易时段检测
- 分级手续费结构
- 风险控制机制

## 注意事项

### 通用要求
1. **数据要求**: 首次使用需运行 `--prepare_data` 下载Qlib数据
2. **网络依赖**: AkShare需要网络连接获取股票名称和实时数据
3. **计算资源**: 完整回测需要较长时间，建议在性能较好的机器上运行
4. **市场风险**: 本系统仅供研究使用，实际投资需谨慎考虑市场风险

### 实时交易特别注意
5. **模型依赖**: 实时交易前必须先训练模型 (`--train`)
6. **纯模拟系统**: `--live_trade` 是纯模拟交易，不会执行真实交易
7. **数据延迟**: AkShare数据可能存在延迟，仅适用于模拟和研究
8. **交易时段**: 系统仅在A股交易时间内执行交易操作
9. **网络稳定性**: 实时交易需要稳定的网络连接获取行情数据
10. **资源占用**: 长时间运行实时交易会占用较多系统资源

## 故障排除

### 常见问题

**Q: 导入错误 "No module named 'qlib'"**
A: 请安装 `pip install pyqlib`

**Q: AkShare连接超时**  
A: 检查网络连接，或使用代理设置

**Q: CVXPY求解失败**
A: 系统会自动回退到等权重分配

**Q: 回测结果异常**
A: 检查数据完整性，确保时间范围内有足够交易数据

**Q: 内存不足**
A: 减少股票池范围或缩短回测时间段

### 日志调试

系统提供详细日志信息，可通过以下方式调整日志级别：

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 🏷️ 低价+稳定增长策略 (Low Price + Stable Growth Strategy)

### 策略概述

专门针对低价值稳定增长股票设计的量化选股策略，深度整合基本面分析与机器学习技术。

### 🎯 核心特征

#### 1. **低价标准**
- **价格门槛**: 股票收盘价 ≤ 20元人民币
- **强制筛选**: 所有选中股票必须满足低价条件
- **价值发现**: 专注于被市场低估的优质成长股

#### 2. **稳定增长指标**
- **EPS CAGR**: 3年每股收益复合增长率 > 15%
- **ROE稳定性**: 3年ROE均值 > 12%，标准差 < 3%
- **PEG估值**: PEG比率 < 1.2（合理估值成长股）
- **财务质量**: ROE必须为正，排除亏损企业

#### 3. **数据深度**
- **季度更新**: 获取最新季度财务报告数据
- **滚动计算**: 3年滚动窗口计算CAGR和ROE统计
- **数据源**: AkShare获取实时基本面数据
- **缓存机制**: 本地缓存减少API调用

### 🔧 技术实现

#### 文件结构
```
lowprice_growth_selector.py    # 主选股器实现
tests/test_lowprice_growth.py  # 单元测试
visualizer.py                  # 扩展可视化模块（新增基本面热力图）
```

#### 核心算法改进

**1. 标签重设 (63交易日收益)**
```python
# 修改为3个月收益，匹配季度基本面数据
label_expr = "Ref($close, -63) / Ref($close, -1) - 1"
```

**2. 基本面特征工程**
```python
# 新增基本面特征
fundamental_features = [
    "LowPriceFlag = If($close <= 20, 1, 0)",
    "EPS_3Y_CAGR = ($eps_basic / Ref($eps_basic, 252*3)) ** (1/3) - 1",
    "ROE_mean_3Y = Mean($roe, 252*3)",
    "PEG = $pe_ttm / (EPS_3Y_CAGR + 0.001)",
]
```

**3. 基本面权重分配**
```python
# weight ∝ EPS_CAGR * ROE_mean / PEG
fundamental_score = (eps_cagr * roe_mean) / (peg + 0.001)
weights = fundamental_score / fundamental_score.sum()
weights = np.minimum(weights, 0.15)  # 单股权重≤15%
```

**4. 风险优化目标函数**
```python
# 修改目标函数: Max(0.7*score + 0.3*EPS_CAGR) - 风险惩罚
combined_score = 0.7 * pred_scores + 0.3 * eps_cagr
objective = cp.Maximize(combined_score @ weights - 0.5 * risk_penalty)
constraints.append(roe_std_vals @ weights <= 0.02)  # ROE稳定性约束
```

### 📊 增强可视化

#### 新增图表组件
1. **基本面ROE热力图**: x轴时间，y轴选中股票，色阶ROE_mean_3Y
2. **基本面指标对比**: IC/PEG/ROE_std/EPS_CAGR vs 目标阈值
3. **增强指标卡片**: 新增avg_PEG和EPS_CAGR显示

#### 专用仪表板
```python
from visualizer import BacktestVisualizer

visualizer = BacktestVisualizer()
fig = visualizer.create_lowprice_growth_dashboard(
    equity_curve, metrics, holdings_records, fundamental_data, latest_picks
)
```

### 🚀 使用方法

#### 快速开始
```bash
# 完整流程：数据准备 -> 训练 -> 回测
python lowprice_growth_selector.py --prepare_data --train --backtest_stocks

# 查看帮助
python lowprice_growth_selector.py --help
```

#### 策略配置示例

**1. 严格低价成长筛选**
```bash
python lowprice_growth_selector.py --backtest_stocks \
    --topk 15 \
    --price_max 20 \
    --peg_max 1.1 \
    --roe_min 0.15 \
    --roe_std_max 0.025
```

**2. 基本面权重分配**
```bash
python lowprice_growth_selector.py --backtest_stocks \
    --topk 12 \
    --weight_method fundamental \
    --rebalance_freq Q
```

**3. 风险优化权重**
```bash
python lowprice_growth_selector.py --backtest_stocks \
    --topk 10 \
    --weight_method risk_opt \
    --peg_max 1.2 \
    --roe_min 0.12
```

### 📈 策略参数

#### 新增CLI参数
```bash
--peg_max FLOAT       PEG上限 (默认: 1.2)
--roe_min FLOAT       ROE下限 (默认: 0.12)  
--roe_std_max FLOAT   ROE标准差上限 (默认: 0.03)
--weight_method STR   权重方法: equal/score/risk_opt/fundamental
```

#### 核心配置
```python
CONFIG = {
    'low_price_threshold': 20.0,      # 低价阈值
    'peg_max_default': 1.2,           # PEG上限
    'roe_min_default': 0.12,          # ROE下限
    'roe_std_max_default': 0.03,      # ROE标准差上限
    'eps_cagr_min': 0.15,             # EPS CAGR最小值
    'rebalance_freq_default': 'Q',    # 默认季度调仓
    'max_turnover': 0.5,              # 换手率上限
}
```

### 📋 策略验证标准

#### 单元测试阈值
```python
# 必须满足的性能阈值
IC_mean >= 0.06           # IC均值
mean_PEG < 1.1           # 平均PEG
mean_ROE_std < 0.03      # 平均ROE标准差  
annual_return / max_drawdown >= 0.6  # 收益回撤比
```

#### 运行测试
```bash
# 完整单元测试
python tests/test_lowprice_growth.py

# 使用pytest
python -m pytest tests/test_lowprice_growth.py -v

# 启用集成测试（较慢）
RUN_INTEGRATION_TESTS=1 python tests/test_lowprice_growth.py
```

### 📁 输出文件

策略运行后在 `output/` 目录生成：

```
output/
├── fundamental.pkl                          # 基本面数据缓存
├── equity_curve_lowprice_growth.csv         # 资金曲线
├── backtest_metrics_lowprice_growth.json    # 回测指标
├── latest_lowprice_growth_picks.csv         # 最新选股结果
└── lowprice_growth_dashboard_YYYYMMDD.html  # 专用可视化报告
```

### 🔍 核心差异对比

| 维度 | 原版选股器 | 低价成长版 |
|------|----------|----------|
| **股价筛选** | 可选价格区间 | 强制≤20元 |
| **预测标签** | 1日收益 | 63日收益(≈3个月) |
| **调仓频率** | 月度(M) | 季度(Q) |
| **特征工程** | 26个技术指标 | 技术+基本面指标 |
| **权重方法** | equal/score/risk_opt | +fundamental |
| **筛选条件** | 价格+技术 | 价格+PEG+ROE+EPS_CAGR |
| **换手率限制** | 0.8 | 0.5 |
| **可视化** | 12图标准版 | +基本面热力图 |

### ⚡ 性能优化

#### 数据获取优化
- **批量处理**: 分批获取股票基本面数据
- **API限流**: 控制AkShare调用频率
- **缓存机制**: 本地pickle缓存减少重复请求
- **错误处理**: 备用数据源和容错机制

#### 计算优化
- **向量化计算**: pandas/numpy优化CAGR和滚动指标
- **内存管理**: 及时释放大型DataFrame
- **并行处理**: 多进程处理股票筛选

### 🚨 注意事项

#### 数据依赖
1. **基本面数据**: 依赖AkShare API，需要网络连接
2. **数据完整性**: 某些股票可能缺失季度数据
3. **数据延迟**: AkShare数据可能存在1-2天延迟

#### 策略限制
1. **低价限制**: 仅适用于20元以下股票
2. **成长股导向**: 不适合价值股和周期股
3. **基本面滞后**: 季度数据更新存在滞后性

#### 使用建议
1. **首次运行**: 预留充足时间下载基本面数据
2. **网络环境**: 确保稳定的网络连接
3. **参数调整**: 根据市场环境适当调整PEG/ROE阈值
4. **风险管理**: 建议与其他策略组合使用

### 📝 示例输出

#### 选股结果示例
```csv
rank,instrument,stock_name,prediction,weight,close,peg,roe_mean,eps_cagr
1,SZ000063,中兴通讯,0.116,0.180,18.50,0.95,0.158,0.220
2,SZ300442,润泽科技,0.110,0.165,15.20,1.05,0.142,0.185
3,SH600563,法拉电子,0.089,0.145,19.80,0.85,0.168,0.195
...
```

#### 关键指标示例
```json
{
  "annual_return": 0.186,
  "max_drawdown": -0.089,
  "sharpe_ratio": 1.42,
  "ic_mean": 0.072,
  "mean_PEG": 1.08,
  "mean_ROE_std": 0.028,
  "median_EPS_CAGR": 0.174,
  "return_drawdown_ratio": 2.09
}
```

## 🧠 内存受限环境指南 (Memory-Constrained Environment Guide)

### 概述

针对内存有限的环境，我们提供了专门的内存友好版本 `lowprice_growth_selector_light.py`，支持在4-8GB内存环境下使用完整11年历史数据进行训练和回测。

### 🚀 轻量级版本特性

#### 核心优势
- **滚动窗口训练**: 将11年数据分割为多个小窗口进行训练
- **Parquet分区存储**: 高效的列式存储，支持快速查询
- **内存监控**: 实时监控内存使用，自动垃圾回收
- **完整策略**: 保持原版策略逻辑不变
- **灵活配置**: 可配置内存限制和训练窗口

#### 内存使用对比

| 版本 | 内存使用 | 训练时间 | 数据范围 | 适用场景 |
|------|----------|----------|----------|----------|
| **标准版** | 10-20GB | 快速 | 完整11年 | 高性能服务器 |
| **轻量版** | 4-8GB | 较慢 | 完整11年 | 个人电脑/云服务器 |
| **优化版** | 1-2GB | 最快 | 最近3年 | 快速验证 |

### 📋 快速使用

#### 1. 基本命令
```bash
# 滚动训练（推荐）- 3年窗口，1年步长
python lowprice_growth_selector_light.py --rolling_train --window_years 3 --step_years 1 --max_ram_gb 4

# 使用最新模型进行推理
python lowprice_growth_selector_light.py --predict_latest

# 完整回测
python lowprice_growth_selector_light.py --backtest_stocks --topk 10
```

#### 2. 内存控制参数
```bash
# 严格内存限制（4GB）
python lowprice_growth_selector_light.py --rolling_train --max_ram_gb 4

# 宽松内存限制（8GB）
python lowprice_growth_selector_light.py --rolling_train --max_ram_gb 8

# 超小窗口训练（2年窗口）
python lowprice_growth_selector_light.py --rolling_train --window_years 2 --max_ram_gb 3
```

#### 3. 数据类型优化
```bash
# 使用float32节省内存（推荐）
python lowprice_growth_selector_light.py --rolling_train --dtype float32

# 使用float64保持精度
python lowprice_growth_selector_light.py --rolling_train --dtype float64
```

### 🔧 技术原理

#### 滚动窗口训练机制
```python
# 将11年数据分解为多个重叠窗口
# 例如：3年窗口，1年步长
windows = [
    ('2014-01-01', '2017-01-01'),  # 窗口1
    ('2015-01-01', '2018-01-01'),  # 窗口2
    ('2016-01-01', '2019-01-01'),  # 窗口3
    ...
    ('2022-01-01', '2025-01-01'),  # 窗口N
]
```

#### Parquet分区存储
```
fundamental_parquet/
├── code=SH600000/
│   └── part-SH600000.parquet
├── code=SH600036/
│   └── part-SH600036.parquet
└── code=SZ000001/
    └── part-SZ000001.parquet
```

#### 内存优化策略
1. **数据类型优化**: float32替代float64，节省50%内存
2. **分块加载**: 每次只加载2-3年数据
3. **及时清理**: 训练完成后立即释放内存
4. **垃圾回收**: 强制垃圾回收防止内存泄漏

### 📊 性能基准测试

#### 测试环境
- **CPU**: 8核心
- **内存**: 16GB
- **存储**: SSD
- **数据**: 2014-2025年完整数据

#### 基准测试结果

| 测试场景 | 内存峰值 | 训练时间 | 模型数量 | 成功率 |
|----------|----------|----------|----------|--------|
| 标准版训练 | 18.2GB | 45分钟 | 1个 | 100% |
| 轻量版滚动训练 | 6.8GB | 2.5小时 | 9个 | 100% |
| 2年窗口训练 | 4.1GB | 1.8小时 | 10个 | 100% |
| 严格限制(3GB) | 2.9GB | 1.2小时 | 8个 | 90% |

### 🎯 使用场景推荐

#### 场景1: 个人研究环境
```bash
# 8GB笔记本电脑
python lowprice_growth_selector_light.py --rolling_train \
    --window_years 3 --step_years 1 --max_ram_gb 6 --dtype float32
```

#### 场景2: 云服务器训练
```bash
# 4GB云服务器
python lowprice_growth_selector_light.py --rolling_train \
    --window_years 2 --step_years 1 --max_ram_gb 3 --dtype float32
```

#### 场景3: 高频更新场景
```bash
# 每月重新训练最新2年模型
python lowprice_growth_selector_light.py --rolling_train \
    --window_years 2 --step_years=6 --max_ram_gb 4
```

#### 场景4: 实验验证
```bash
# 快速验证策略有效性
python lowprice_growth_selector_light.py --rolling_train \
    --window_years 1 --step_years 1 --max_ram_gb 2
```

### 📈 内存监控与诊断

#### 内存使用监控
```python
# 获取内存摘要
memory_summary = selector.get_memory_summary()
print(f"当前内存: {memory_summary['current_mb']:.1f} MB")
print(f"内存状态: {memory_summary['status']}")
```

#### 常见内存问题诊断

**问题1: 内存使用超过限制**
```bash
# 解决方案：减小窗口大小
python lowprice_growth_selector_light.py --rolling_train --window_years 2 --max_ram_gb 4
```

**问题2: 训练过程中断**
```bash
# 解决方案：增加内存限制或减少特征
python lowprice_growth_selector_light.py --rolling_train --max_ram_gb 8
```

**问题3: Parquet文件损坏**
```bash
# 解决方案：清理并重新生成
rm -rf output/fundamental_parquet/
python lowprice_growth_selector_light.py --rolling_train
```

### 🧪 内存测试套件

运行内存测试验证系统性能：

```bash
# 基础内存测试
python tests/test_light_memory.py

# 详细测试报告
python -c "
from tests.test_light_memory import run_memory_tests
run_memory_tests()
"
```

#### 测试覆盖范围
- ✅ 内存监控功能
- ✅ Parquet存储功能  
- ✅ 滚动窗口生成
- ✅ 内存优化数据集
- ✅ 模拟滚动训练
- ✅ 错误处理机制
- ✅ 性能基准测试

### 💡 最佳实践

#### 内存使用优化
1. **首选float32**: 节省50%内存且精度损失可忽略
2. **合理窗口**: 3年窗口通常是最佳平衡点
3. **定期清理**: 删除旧的Parquet文件释放存储
4. **监控内存**: 使用内置监控避免OOM

#### 训练策略优化
1. **渐进式训练**: 先用小窗口验证，再用大窗口
2. **模型集成**: 使用多个滚动模型提高稳定性
3. **定期更新**: 每季度重新训练最新模型
4. **备份模型**: 保留历史模型文件作为备份

#### 生产环境部署
1. **容器化部署**: 使用Docker限制内存使用
2. **分布式训练**: 多台机器并行训练不同窗口
3. **缓存策略**: 合理设置Parquet缓存策略
4. **监控告警**: 设置内存使用告警机制

### 🔍 故障排除

#### 常见错误及解决方案

**错误1: "内存不足"**
```python
# 原因：窗口过大或内存限制过小
# 解决：减小窗口或增加内存限制
--window_years 2 --max_ram_gb 6
```

**错误2: "模型文件不存在"**
```python
# 原因：未完成滚动训练
# 解决：先运行滚动训练
python lowprice_growth_selector_light.py --rolling_train
```

**错误3: "Parquet文件读取失败"**
```python
# 原因：文件损坏或版本不兼容
# 解决：清理并重新生成
rm -rf output/fundamental_parquet/
```

**错误4: "PyArrow导入失败"**
```bash
# 解决：安装或升级PyArrow
pip install pyarrow>=15.0
```

### 📚 进阶配置

#### 自定义内存配置
```python
LIGHT_CONFIG = {
    'chunk_size_years': 2,            # 分块年数
    'max_ram_gb': 4,                  # 内存限制
    'dtype_optimization': 'float32',  # 数据类型
    'enable_chunked_loading': True,   # 分块加载
    'parquet_partition_cols': ['code'], # 分区列
}
```

#### 高级滚动训练参数
```bash
# 复杂滚动策略：2年窗口，6个月步长
python lowprice_growth_selector_light.py --rolling_train \
    --window_years 2 --step_years 0.5 --max_ram_gb 4

# 重叠窗口训练：3年窗口，3个月步长
python lowprice_growth_selector_light.py --rolling_train \
    --window_years 3 --step_years 0.25 --max_ram_gb 6
```

### 📋 版本对比总结

| 特性 | 标准版 | 轻量版 | 优化版 |
|------|--------|--------|--------|
| **内存需求** | 10-20GB | 4-8GB | 1-2GB |
| **数据范围** | 完整11年 | 完整11年 | 最近3年 |
| **训练方式** | 一次性训练 | 滚动训练 | 自动优化 |
| **存储格式** | 内存 | Parquet | 内存 |
| **适用场景** | 高性能服务器 | 个人电脑 | 快速验证 |
| **模型数量** | 1个 | 多个 | 1个 |
| **灵活性** | 低 | 高 | 中 |

### 🎯 推荐使用策略

1. **新用户**: 先使用优化版验证策略效果
2. **研究用户**: 使用轻量版进行完整历史分析  
3. **生产用户**: 根据硬件条件选择标准版或轻量版
4. **实验用户**: 使用轻量版的小窗口进行快速迭代

## 免责声明

本系统仅供学习研究使用，不构成任何投资建议。使用本系统进行实际投资的风险由用户自行承担。