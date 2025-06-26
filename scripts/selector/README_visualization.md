# AI选股器Plotly可视化功能

## 🎯 功能概述

为`ai_stock_selector.py`开发的全面可视化分析系统，提供专业的金融图表和交互式分析报告。

## ✨ 主要特性

### 📊 综合仪表板 (Comprehensive Dashboard)
- **12个子图表布局**：3行4列的专业仪表板设计
- **美观配色方案**：专业金融风格，蓝绿主色调
- **响应式布局**：自适应不同屏幕尺寸
- **交互式控件**：时间范围选择器、缩放、平移

### 🎨 可视化图表类型

1. **资金曲线图** - 策略净值走势，支持时间范围选择
2. **收益分布直方图** - 日收益率分布及正态拟合
3. **IC时序分析** - 信息系数时间序列
4. **最大回撤分析** - 回撤幅度可视化
5. **月度收益热力图** - 年月收益矩阵热图
6. **滚动夏普比率** - 动态风险调整收益
7. **换手率分析** - 交易频率统计
8. **风险指标柱状图** - 多维度风险度量
9. **最新选股表格** - AI推荐股票列表
10. **预测评分分布** - 模型预测分数直方图
11. **累计收益对比** - 策略vs基准表现
12. **关键指标总览** - 核心绩效指标展示

### 🔧 详细分析报告
- **风险归因分析** - 市场、行业、个股风险分解
- **收益贡献分解** - 选股、择时、成本收益构成
- **预测精度分析** - 模型预测准确性评估
- **交易成本分析** - 换手率与成本效益关系

## 🚀 使用方法

### 1. 基本使用
```bash
# 运行完整回测并生成可视化
python ai_stock_selector.py --backtest

# 仅生成可视化报告（基于已有数据）
python ai_stock_selector.py --visualize

# 运行可视化演示
python demo_visualization.py
```

### 2. 程序集成
```python
from visualizer import BacktestVisualizer

# 创建可视化器
visualizer = BacktestVisualizer()

# 生成综合仪表板
fig = visualizer.create_comprehensive_dashboard(
    equity_curve=equity_data,
    metrics=backtest_metrics,
    predictions=model_predictions,
    latest_picks=stock_recommendations
)

# 保存为HTML
html_path = visualizer.save_dashboard_as_html(fig, "report.html")
```

### 3. 命令行选项
```bash
python ai_stock_selector.py [OPTIONS]

选项说明:
  --visualize          创建可视化分析报告
  --backtest          运行回测(包含可视化)
  --topk INT          选股数量(默认10)
  --price_min FLOAT   最低价格限制
  --price_max FLOAT   最高价格限制
```

## 📁 文件结构

```
scripts/selector/
├── ai_stock_selector.py      # 主程序(已集成可视化)
├── visualizer.py            # 可视化核心模块
├── demo_visualization.py    # 演示脚本
├── README_visualization.md  # 本文档
└── output/                  # 输出文件夹
    ├── *_dashboard_*.html   # 综合仪表板
    ├── *_analysis_*.html    # 详细分析报告
    ├── backtest_metrics_*.json  # 回测指标
    ├── equity_curve_*.csv   # 资金曲线数据
    └── latest_stock_picks.csv   # 最新选股
```

## 🎨 配色方案与主题

### 专业金融配色
- **主色调**: 蓝色系 (#1f77b4) - 稳定、专业
- **辅助色**: 橙色 (#ff7f0e) - 活跃、警示
- **成功色**: 绿色 (#2ca02c) - 盈利、正向
- **警告色**: 红色 (#d62728) - 风险、负向
- **中性色**: 灰色系 - 背景、辅助信息

### 美化元素
- **渐变背景**: 现代科技感
- **阴影效果**: 层次感和立体感
- **圆角设计**: 友好的视觉体验
- **响应式字体**: 不同设备适配

## 🎯 交互功能

### 图表交互
- **悬停提示**: 详细数据显示
- **缩放平移**: 鼠标滚轮、拖拽操作
- **图例控制**: 点击显示/隐藏数据系列
- **双击重置**: 恢复默认视图

### 时间控制
- **范围选择器**: 1M、3M、6M、1Y、全部
- **滑动条**: 连续时间范围调整
- **日期选择**: 精确时间定位

### 数据导出
- **高质量PNG**: 2倍分辨率图片导出
- **PDF导出**: 专业报告格式
- **数据表格**: CSV格式数据导出

## 📊 核心指标解释

### 收益指标
- **年化收益率**: 策略年化回报率
- **夏普比率**: 风险调整后收益
- **最大回撤**: 最大亏损幅度
- **胜率**: 盈利交易占比

### 风险指标
- **年化波动率**: 收益率标准差
- **下行偏差**: 负收益波动率
- **VaR(95%)**: 95%置信水平风险价值
- **换手率**: 交易频率指标

### 预测指标
- **IC均值**: 信息系数平均值
- **IC_IR**: 信息比率
- **预测精度**: 方向预测准确率

## 🛠️ 技术实现

### 核心技术栈
- **Plotly**: 交互式图表库
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **HTML/CSS/JavaScript**: 前端展示

### 关键特性
- **子图布局**: 3×4网格设计
- **数据绑定**: 动态数据更新
- **主题系统**: 一致的视觉风格
- **错误处理**: 数据缺失容错

## 📋 使用示例

### 示例1: 基础回测可视化
```python
# 运行AI选股器回测
selector = AIStockSelector()
selector.prepare_data()
selector.train()
results = selector.backtest()  # 自动生成可视化
```

### 示例2: 自定义可视化
```python
# 创建自定义可视化
visualizer = BacktestVisualizer()

# 设置自定义配色
visualizer.colors['primary'] = '#2E86AB'
visualizer.colors['success'] = '#A23B72'

# 生成报告
fig = visualizer.create_comprehensive_dashboard(data, metrics)
```

### 示例3: 批量报告生成
```python
# 批量处理多个模型
models = ['lgb', 'xgb', 'ensemble']
for model in models:
    selector.create_visualization_only(model_name=model)
```

## 🔧 依赖安装

```bash
# 必需依赖
pip install plotly pandas numpy

# 可选依赖(用于更多功能)
pip install kaleido  # 静态图片导出
pip install psutil   # 系统资源监控
```

## 🚨 注意事项

### 数据要求
- 回测数据必须包含时间序列
- 预测数据格式需要规范化
- 股票代码需要标准化格式

### 性能考虑
- 大数据集可能影响渲染速度
- 建议使用Chrome浏览器查看
- 移动端查看效果可能受限

### 兼容性
- 支持主流现代浏览器
- 需要JavaScript支持
- 建议1920×1080以上分辨率

## 📈 示例输出

生成的HTML报告包含：

1. **顶部标题区域** - 项目名称、生成时间
2. **使用说明面板** - 交互操作指导
3. **12个图表区域** - 核心分析内容
4. **底部信息栏** - 版权、技术信息

## 🎯 最佳实践

### 数据准备
- 确保数据完整性和一致性
- 预处理异常值和缺失值
- 标准化时间格式

### 可视化设计
- 选择合适的图表类型
- 保持配色一致性
- 突出关键信息

### 用户体验
- 提供清晰的操作指导
- 优化加载速度
- 确保跨设备兼容性

---

*本可视化系统为AI选股器提供了专业级的分析展示能力，结合了现代Web技术和金融分析需求，为用户提供直观、交互式的投资分析体验。*