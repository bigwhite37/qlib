#!/usr/bin/env python3
"""
AI选股器可视化演示脚本
====================

运行此脚本可以生成完整的可视化演示报告，
展示AI选股器的所有可视化功能。

Usage:
    python demo_visualization.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from visualizer import BacktestVisualizer
    print("✓ 可视化模块导入成功")
except ImportError as e:
    print(f"❌ 可视化模块导入失败: {e}")
    sys.exit(1)

def create_demo_data():
    """创建演示数据"""
    print("🔄 正在生成演示数据...")
    
    # 创建时间序列
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 6, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 创建模拟收益率数据
    np.random.seed(42)
    n_days = len(dates)
    
    # 基础趋势 + 噪声 + 季节性
    trend = np.linspace(0, 0.15, n_days) / 252  # 年化15%收益的趋势
    seasonality = 0.002 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # 年度季节性
    noise = np.random.normal(0, 0.015, n_days)  # 日波动
    
    daily_returns = trend + seasonality + noise
    
    # 添加一些极端事件
    crash_dates = [252, 500, 800]  # 模拟三次大跌
    for crash_date in crash_dates:
        if crash_date < len(daily_returns):
            daily_returns[crash_date:crash_date+5] = np.random.normal(-0.04, 0.01, 5)
    
    # 计算累计净值
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # 创建资金曲线数据框
    equity_curve = pd.DataFrame({
        'date': dates,
        'returns': daily_returns,
        'equity': cumulative_returns
    })
    
    # 创建回测指标
    total_return = cumulative_returns[-1] - 1
    annual_return = total_return / ((end_date - start_date).days / 365.25)
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 计算最大回撤
    cumulative_series = pd.Series(cumulative_returns)
    peak = cumulative_series.expanding().max()
    drawdown = (cumulative_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # 其他指标
    win_rate = (daily_returns > 0).mean()
    
    metrics = {
        'ic_mean': np.random.normal(0.06, 0.01),
        'ic_ir': np.random.normal(1.5, 0.2),
        'ic_std': np.random.normal(0.12, 0.02),
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_turnover': np.random.normal(0.75, 0.1),
        'trading_days': len(daily_returns),
        'downside_deviation': volatility * 0.8,  # 下行偏差通常小于总波动率
        'var_95': np.percentile(daily_returns, 5)  # 95% VaR
    }
    
    # 创建预测数据
    predictions = {
        'lgb': pd.Series(np.random.normal(0.02, 0.05, 300), name='lgb_predictions'),
        'xgb': pd.Series(np.random.normal(0.015, 0.04, 300), name='xgb_predictions'),
        'ensemble': pd.Series(np.random.normal(0.025, 0.03, 300), name='ensemble_predictions')
    }
    
    # 创建最新选股数据
    stock_codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ', 
                   '002415.SZ', '300015.SZ', '000725.SZ', '002049.SZ', '600519.SH']
    stock_names = ['平安银行', '万科A', '浦发银行', '招商银行', '五粮液',
                   '海康威视', '爱尔眼科', '京东方A', '紫光国微', '贵州茅台']
    
    latest_picks = pd.DataFrame({
        'rank': range(1, 11),
        'stock_code': stock_codes,
        'stock_name': stock_names,
        'pred_score': np.random.uniform(0.65, 0.95, 10),
        'weight': np.random.dirichlet(np.ones(10)),
        'current_price': np.random.uniform(10, 200, 10),
        'change_pct': np.random.normal(0, 3, 10),
        'change_amount': np.random.normal(0, 2, 10),
        'volume': np.random.uniform(1000000, 50000000, 10),
        'turnover': np.random.uniform(10000000, 1000000000, 10),
        'date': datetime.now().strftime('%Y-%m-%d')
    })
    
    print("✅ 演示数据生成完成")
    return equity_curve, metrics, predictions, latest_picks

def create_demo_visualization():
    """创建演示可视化"""
    print("\n🎨 开始创建可视化演示...")
    
    # 生成演示数据
    equity_curve, metrics, predictions, latest_picks = create_demo_data()
    
    # 创建可视化器
    visualizer = BacktestVisualizer()
    
    print("📊 正在生成综合仪表板...")
    
    # 创建综合仪表板
    dashboard_fig = visualizer.create_comprehensive_dashboard(
        equity_curve=equity_curve,
        metrics=metrics,
        predictions=predictions,
        latest_picks=latest_picks,
        title="AI选股器回测分析演示报告"
    )
    
    # 保存综合仪表板
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dashboard_path = visualizer.save_dashboard_as_html(
        dashboard_fig, 
        f"demo_comprehensive_dashboard_{timestamp}.html"
    )
    
    print("📈 正在生成详细分析报告...")
    
    # 创建详细分析报告
    detailed_fig = visualizer.create_detailed_analysis(
        equity_curve=equity_curve,
        metrics=metrics,
        predictions=predictions
    )
    
    # 保存详细分析报告
    detailed_path = visualizer.save_dashboard_as_html(
        detailed_fig,
        f"demo_detailed_analysis_{timestamp}.html"
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("🎉 AI选股器可视化演示报告生成完成!")
    print("="*80)
    print(f"📊 综合仪表板: {dashboard_path}")
    print(f"📈 详细分析报告: {detailed_path}")
    print("\n💡 使用说明:")
    print("1. 用浏览器打开HTML文件查看交互式图表")
    print("2. 使用时间范围选择器查看不同时期数据")
    print("3. 悬停查看详细信息，双击重置缩放")
    print("4. 点击工具栏相机图标可导出高质量图片")
    print("5. 点击图例可显示/隐藏数据系列")
    
    # 展示关键指标
    print(f"\n📋 演示数据关键指标:")
    print(f"   年化收益率: {metrics['annual_return']:.2%}")
    print(f"   夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"   最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"   胜率: {metrics['win_rate']:.2%}")
    print(f"   IC均值: {metrics['ic_mean']:.4f}")
    print("="*80)
    
    return dashboard_path, detailed_path

def main():
    """主函数"""
    print("🚀 AI选股器可视化演示程序")
    print("="*50)
    
    try:
        # 检查输出目录
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        # 创建演示可视化
        dashboard_path, detailed_path = create_demo_visualization()
        
        # 尝试自动打开浏览器（仅在支持的系统上）
        try:
            import webbrowser
            print(f"\n🌐 正在浏览器中打开: {dashboard_path}")
            webbrowser.open(f"file://{Path(dashboard_path).absolute()}")
        except Exception:
            print("📝 请手动在浏览器中打开HTML文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)