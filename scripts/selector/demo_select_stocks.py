#!/usr/bin/env python3
"""
演示 --select_stocks 功能的股价获取
"""
import akshare as ak
from datetime import datetime, timedelta
import pandas as pd
import sys

def demo_select_stocks_with_prices():
    """演示选股功能及股价获取"""
    print("\n" + "="*120)
    print("🎯 AI选股功能演示 - 获取最新股价")
    print("="*120)
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 模拟选出的股票（来自AI模型预测）
    selected_stocks = [
        {'rank': 1, 'stock_code': 'SH600941', 'stock_name': '中国移动', 'pred_score': 0.1007, 'weight': 0.33},
        {'rank': 2, 'stock_code': 'SH600660', 'stock_name': '福耀玻璃', 'pred_score': 0.0916, 'weight': 0.33},
        {'rank': 3, 'stock_code': 'SH600115', 'stock_name': '中国东航', 'pred_score': 0.0871, 'weight': 0.34}
    ]

    print("正在获取最新股价信息...")

    # 获取股价数据
    for stock in selected_stocks:
        try:
            stock_code = stock['stock_code']
            # 转换股票代码格式
            code = stock_code[2:] if stock_code.startswith(('SH', 'SZ')) else stock_code

            print(f"  获取 {stock_code} 股价数据...")

            # 获取历史数据（最近5天）
            hist_data = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d'),
                adjust="qfq"
            )

            if not hist_data.empty:
                latest_row = hist_data.iloc[-1]
                prev_close = hist_data.iloc[-2]['收盘'] if len(hist_data) > 1 else latest_row['收盘']
                current_price = float(latest_row['收盘'])
                change_amount = current_price - prev_close
                change_pct = (change_amount / prev_close * 100) if prev_close > 0 else 0

                # 更新股票信息
                stock.update({
                    'current_price': current_price,
                    'change_pct': change_pct,
                    'change_amount': change_amount,
                    'volume': int(latest_row['成交量']) if '成交量' in latest_row else 0,
                    'turnover': float(latest_row['成交额']) if '成交额' in latest_row else 0,
                    'date': latest_row['日期']
                })
            else:
                # 无数据时设置默认值
                stock.update({
                    'current_price': 0,
                    'change_pct': 0,
                    'change_amount': 0,
                    'volume': 0,
                    'turnover': 0,
                    'date': datetime.now().strftime('%Y-%m-%d')
                })

        except Exception as e:
            print(f"    获取 {stock_code} 失败: {e}")
            stock.update({
                'current_price': 0,
                'change_pct': 0,
                'change_amount': 0,
                'volume': 0,
                'turnover': 0,
                'date': datetime.now().strftime('%Y-%m-%d')
            })

    # 保存为CSV
    df = pd.DataFrame(selected_stocks)
    output_file = 'demo_latest_stock_picks.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 打印结果表格
    print("\n" + "-"*120)
    print(f"{'排名':<4} {'股票代码':<10} {'股票名称':<10} {'当前价格':<8} {'涨跌幅':<8} {'涨跌额':<8} {'成交量(万)':<10} {'预测评分':<8} {'权重':<6}")
    print("-"*120)

    total_volume = 0
    valid_prices = []

    for stock in selected_stocks:
        volume_wan = stock['volume'] / 10000 if stock['volume'] > 0 else 0
        total_volume += volume_wan

        if stock['current_price'] > 0:
            valid_prices.append(stock['current_price'])

        # 格式化涨跌幅和涨跌额
        if stock['change_pct'] > 0:
            change_pct_str = f"+{stock['change_pct']:.2f}%"
            change_amount_str = f"+{stock['change_amount']:.2f}"
        elif stock['change_pct'] < 0:
            change_pct_str = f"{stock['change_pct']:.2f}%"
            change_amount_str = f"{stock['change_amount']:.2f}"
        else:
            change_pct_str = "0.00%"
            change_amount_str = "0.00"

        print(f"{stock['rank']:<4} {stock['stock_code']:<10} {stock['stock_name']:<10} "
              f"{stock['current_price']:<8.2f} {change_pct_str:<8} {change_amount_str:<8} "
              f"{volume_wan:<10.0f} {stock['pred_score']:<8.4f} {stock['weight']:<6.1%}")

    print("-"*120)

    # 计算统计信息
    avg_score = sum(s['pred_score'] for s in selected_stocks) / len(selected_stocks)
    avg_price = sum(valid_prices) / len(valid_prices) if valid_prices else 0

    print(f"平均预测评分: {avg_score:.4f} | 平均股价: {avg_price:.2f}元 | 总成交量: {total_volume:.0f}万股")
    print(f"选股文件已保存: {output_file}")
    print("="*120)
    print()

    return selected_stocks

if __name__ == "__main__":
    try:
        results = demo_select_stocks_with_prices()
        print("✅ 股价获取功能演示完成！")
        print(f"成功获取 {len([s for s in results if s['current_price'] > 0])} 只股票的实时价格")
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        sys.exit(1)
