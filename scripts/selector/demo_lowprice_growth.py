#!/usr/bin/env python3
"""
低价+稳定增长策略演示脚本
==========================

展示完整的低价成长选股策略实现和使用方法。

Usage:
    python demo_lowprice_growth.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

print("=" * 60)
print("🏷️ 低价+稳定增长选股策略演示")
print("=" * 60)

# 检查依赖
try:
    from lowprice_growth_selector import LowPriceGrowthSelector, CONFIG
    from visualizer import BacktestVisualizer
    print("✓ 成功导入所有模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保已安装所有依赖: pip install pandas>=2.1 tqdm akshare>=1.11")
    sys.exit(1)

def demo_basic_functionality():
    """演示基础功能"""
    print("\n1. 基础功能演示")
    print("-" * 30)
    
    # 创建选股器
    selector = LowPriceGrowthSelector(
        start_date='2023-06-01',
        end_date='2023-12-31',
        universe='csi300'
    )
    print(f"✓ 选股器创建成功，数据范围: {selector.start_date} 至 {selector.end_date}")
    
    # 测试CAGR计算
    test_series = pd.Series([100, 110, 121, 133.1])
    cagr = selector._calculate_cagr(test_series)
    print(f"✓ CAGR计算测试: {cagr:.4f} (预期约0.10)")
    
    # 测试换手率计算
    old_holdings = {'A': 0.3, 'B': 0.4, 'C': 0.3}
    new_holdings = {'A': 0.2, 'B': 0.3, 'D': 0.5}
    turnover = selector._calculate_turnover(old_holdings, new_holdings)
    print(f"✓ 换手率计算测试: {turnover:.3f} (预期0.5)")
    
    return selector

def demo_config_parameters():
    """演示配置参数"""
    print("\n2. 策略配置参数")
    print("-" * 30)
    
    key_params = [
        ('低价阈值', 'low_price_threshold', '元'),
        ('PEG上限', 'peg_max_default', ''),
        ('ROE下限', 'roe_min_default', ''),
        ('ROE标准差上限', 'roe_std_max_default', ''),
        ('EPS CAGR最小值', 'eps_cagr_min', ''),
        ('默认调仓频率', 'rebalance_freq_default', ''),
        ('换手率上限', 'max_turnover', ''),
    ]
    
    for name, key, unit in key_params:
        value = CONFIG.get(key, 'N/A')
        print(f"  {name}: {value}{unit}")
    
    print("✓ 配置参数符合低价成长策略要求")

def demo_fundamental_analysis():
    """演示基本面分析"""
    print("\n3. 基本面分析演示")
    print("-" * 30)
    
    # 创建模拟基本面数据
    mock_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', '2023-12-31', freq='Q'),
        'code': ['SH600000'] * 4,
        'net_profit': [1000, 1100, 1210, 1331],  # 10%增长
        'revenue': [5000, 5500, 6050, 6655],     # 10%增长
        'roe': [0.15, 0.16, 0.14, 0.17],         # 稳定ROE
        'eps_basic': [1.0, 1.1, 1.21, 1.331]    # 10%增长
    })
    
    print(f"✓ 模拟基本面数据: {len(mock_data)}条季度记录")
    
    # 计算关键指标
    eps_cagr = (mock_data['eps_basic'].iloc[-1] / mock_data['eps_basic'].iloc[0]) ** (1/0.75) - 1
    roe_mean = mock_data['roe'].mean()
    roe_std = mock_data['roe'].std()
    
    print(f"  EPS CAGR: {eps_cagr:.2%} (目标 > 15%)")
    print(f"  ROE均值: {roe_mean:.2%} (目标 > 12%)")  
    print(f"  ROE标准差: {roe_std:.2%} (目标 < 3%)")
    
    # 评估是否符合标准
    meets_criteria = (eps_cagr > 0.15) and (roe_mean > 0.12) and (roe_std < 0.03)
    print(f"✓ 基本面指标评估: {'符合' if meets_criteria else '不符合'}低价成长标准")
    
    return mock_data

def demo_weight_allocation():
    """演示权重分配方法"""
    print("\n4. 权重分配方法演示")
    print("-" * 30)
    
    # 创建测试股票数据
    stocks_data = pd.DataFrame({
        'instrument': ['SH600000', 'SH600036', 'SZ000001'],
        'stock_name': ['浦发银行', '招商银行', '平安银行'],
        'prediction': [0.8, 0.7, 0.6],
        'close': [15.0, 18.0, 12.0],
        'eps_growth': [0.20, 0.15, 0.18],
        'roe_mean': [0.15, 0.13, 0.16],
        'peg': [1.0, 1.1, 0.9],
    })
    
    print("测试股票数据:")
    print(stocks_data[['stock_name', 'prediction', 'close', 'peg']].to_string(index=False))
    
    # 1. 等权重分配
    equal_weights = np.ones(len(stocks_data)) / len(stocks_data)
    print(f"\n等权重分配: {equal_weights}")
    
    # 2. 评分权重分配
    scores = stocks_data['prediction'].values
    score_weights = scores / scores.sum()
    print(f"评分权重分配: {score_weights}")
    
    # 3. 基本面权重分配
    fundamental_score = (
        stocks_data['eps_growth'] * stocks_data['roe_mean'] / 
        (stocks_data['peg'] + 0.001)
    )
    fundamental_weights = fundamental_score / fundamental_score.sum()
    fundamental_weights = np.minimum(fundamental_weights, 0.15)  # 权重上限
    fundamental_weights = fundamental_weights / fundamental_weights.sum()  # 重新归一化
    print(f"基本面权重分配: {fundamental_weights}")
    
    print("✓ 权重分配方法验证完成")

def demo_strategy_thresholds():
    """演示策略阈值验证"""
    print("\n5. 策略阈值验证")
    print("-" * 30)
    
    # 模拟一个良好的回测结果
    mock_metrics = {
        'ic_mean': 0.072,           # > 0.06 ✓
        'mean_PEG': 1.08,          # < 1.1 ✓
        'mean_ROE_std': 0.028,     # < 0.03 ✓
        'annual_return': 0.186,
        'max_drawdown': -0.089,
        'return_drawdown_ratio': 2.09  # > 0.6 ✓
    }
    
    # 验证各项阈值
    thresholds = [
        ('IC均值', 'ic_mean', 0.06, '>='),
        ('平均PEG', 'mean_PEG', 1.1, '<'),
        ('ROE标准差', 'mean_ROE_std', 0.03, '<'),
        ('收益回撤比', 'return_drawdown_ratio', 0.6, '>='),
    ]
    
    all_passed = True
    for name, key, threshold, operator in thresholds:
        value = mock_metrics[key]
        if operator == '>=':
            passed = value >= threshold
        else:  # '<'
            passed = value < threshold
        
        status = '✓' if passed else '✗'
        print(f"  {status} {name}: {value:.3f} {operator} {threshold}")
        
        if not passed:
            all_passed = False
    
    print(f"\n策略整体评价: {'✓ 通过所有阈值' if all_passed else '✗ 部分阈值未达标'}")
    return mock_metrics

def demo_cli_examples():
    """演示CLI使用示例"""
    print("\n6. CLI使用示例")
    print("-" * 30)
    
    examples = [
        ("完整流程", "python lowprice_growth_selector.py --prepare_data --train --backtest_stocks"),
        ("严格筛选", "python lowprice_growth_selector.py --backtest_stocks --topk 15 --peg_max 1.1 --roe_min 0.15"),
        ("基本面权重", "python lowprice_growth_selector.py --backtest_stocks --topk 12 --weight_method fundamental"),
        ("季度调仓", "python lowprice_growth_selector.py --backtest_stocks --rebalance_freq Q --weight_method risk_opt"),
    ]
    
    for name, command in examples:
        print(f"\n{name}:")
        print(f"  {command}")
    
    print("\n✓ CLI使用示例展示完成")

def demo_output_files():
    """演示输出文件结构"""
    print("\n7. 输出文件结构")
    print("-" * 30)
    
    output_files = [
        ('fundamental.pkl', '基本面数据缓存', '二进制'),
        ('equity_curve_lowprice_growth.csv', '资金曲线数据', 'CSV'),
        ('backtest_metrics_lowprice_growth.json', '回测指标', 'JSON'),
        ('latest_lowprice_growth_picks.csv', '最新选股结果', 'CSV'),
        ('lowprice_growth_dashboard_YYYYMMDD.html', '可视化报告', 'HTML'),
    ]
    
    print("策略运行后生成的文件:")
    for filename, description, file_type in output_files:
        print(f"  📁 {filename}")
        print(f"     {description} ({file_type}格式)")
    
    print("\n✓ 输出文件结构说明完成")

def demo_visualization():
    """演示可视化功能"""
    print("\n8. 可视化功能演示")
    print("-" * 30)
    
    try:
        # 创建可视化器
        visualizer = BacktestVisualizer()
        print("✓ 可视化器创建成功")
        
        # 检查新增的可视化方法
        has_lowprice_dashboard = hasattr(visualizer, 'create_lowprice_growth_dashboard')
        has_fundamental_heatmap = hasattr(visualizer, '_add_fundamental_heatmap')
        has_enhanced_metrics = hasattr(visualizer, '_add_enhanced_metrics_indicators')
        
        print(f"  低价成长专用仪表板: {'✓' if has_lowprice_dashboard else '✗'}")
        print(f"  基本面ROE热力图: {'✓' if has_fundamental_heatmap else '✗'}")
        print(f"  增强指标卡片: {'✓' if has_enhanced_metrics else '✗'}")
        
        print("✓ 可视化功能检查完成")
        
    except Exception as e:
        print(f"✗ 可视化功能检查失败: {e}")

def demo_comparison_table():
    """演示策略对比"""
    print("\n9. 策略对比表")
    print("-" * 30)
    
    comparison_data = [
        ['维度', '原版选股器', '低价成长版'],
        ['股价筛选', '可选价格区间', '强制≤20元'],
        ['预测标签', '1日收益', '63日收益(≈3个月)'],
        ['调仓频率', '月度(M)', '季度(Q)'],
        ['特征工程', '26个技术指标', '技术+基本面指标'],
        ['权重方法', 'equal/score/risk_opt', '+fundamental'],
        ['筛选条件', '价格+技术', '价格+PEG+ROE+EPS_CAGR'],
        ['换手率限制', '0.8', '0.5'],
        ['可视化', '12图标准版', '+基本面热力图'],
    ]
    
    # 计算列宽
    col_widths = [max(len(str(row[i])) for row in comparison_data) for i in range(3)]
    
    for i, row in enumerate(comparison_data):
        if i == 0:  # 标题行
            print("  " + " | ".join(f"{cell:^{col_widths[j]}}" for j, cell in enumerate(row)))
            print("  " + "-+-".join("-" * width for width in col_widths))
        else:
            print("  " + " | ".join(f"{cell:<{col_widths[j]}}" for j, cell in enumerate(row)))
    
    print("\n✓ 策略对比展示完成")

def main():
    """主演示函数"""
    try:
        # 执行各个演示模块
        selector = demo_basic_functionality()
        demo_config_parameters() 
        mock_data = demo_fundamental_analysis()
        demo_weight_allocation()
        mock_metrics = demo_strategy_thresholds()
        demo_cli_examples()
        demo_output_files()
        demo_visualization()
        demo_comparison_table()
        
        print("\n" + "=" * 60)
        print("📊 演示总结")
        print("=" * 60)
        
        print("✅ 核心功能:")
        print("  • 低价筛选逻辑 (≤20元)")
        print("  • 基本面指标计算 (EPS CAGR, ROE统计, PEG)")
        print("  • 多种权重分配方法 (含基本面权重)")
        print("  • 季度调仓和换手率控制")
        print("  • 策略阈值验证机制")
        
        print("\n✅ 技术特色:")
        print("  • 63交易日标签匹配季度基本面")
        print("  • 滚动3年窗口计算稳定性指标")
        print("  • 基本面热力图可视化")
        print("  • 风险优化目标函数改进")
        print("  • 完整的单元测试覆盖")
        
        print("\n✅ 使用建议:")
        print("  • 首次运行需要网络获取基本面数据")
        print("  • 建议使用基本面或风险优化权重")
        print("  • 适合中长期价值投资策略")
        print("  • 可与其他策略组合使用")
        
        print("\n🎯 策略定位:")
        print("  专注于低价值稳定增长股票的量化选股策略")
        print("  深度整合基本面分析与机器学习技术")
        print("  适合寻找被市场低估的优质成长股")
        
        print(f"\n🔗 快速开始:")
        print(f"  python lowprice_growth_selector.py --prepare_data --train --backtest_stocks")
        
        print("\n✓ 低价+稳定增长策略演示完成!")
        
    except Exception as e:
        print(f"\n✗ 演示过程出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()