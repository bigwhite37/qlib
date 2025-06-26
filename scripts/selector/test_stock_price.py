#!/usr/bin/env python3
"""
测试股价获取功能
"""
import akshare as ak
from datetime import datetime, timedelta
import pandas as pd

def test_stock_price_fetching():
    """测试股价获取功能"""
    stock_codes = ['600941', '600660', '600115']
    results = []

    for stock_code_with_exchange in ['SH600941', 'SH600660', 'SH600115']:
        try:
            # 转换股票代码格式：SH600000 -> 600000
            if stock_code_with_exchange.startswith('SH'):
                code = stock_code_with_exchange[2:]
            elif stock_code_with_exchange.startswith('SZ'):
                code = stock_code_with_exchange[2:]
            else:
                code = stock_code_with_exchange

            print(f'获取股票 {stock_code_with_exchange} ({code}) 的数据...')

            # 获取历史数据
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

                result = {
                    'stock_code': stock_code_with_exchange,
                    'current_price': current_price,
                    'change_pct': change_pct,
                    'change_amount': change_amount,
                    'volume': int(latest_row['成交量']) if '成交量' in latest_row else 0,
                    'turnover': float(latest_row['成交额']) if '成交额' in latest_row else 0,
                    'date': latest_row['日期']
                }

                results.append(result)

                print(f'  ✓ {stock_code_with_exchange}: ')
                print(f'    价格: {current_price:.2f}元')
                print(f'    涨跌幅: {change_pct:+.2f}%')
                print(f'    涨跌额: {change_amount:+.2f}元')
                print(f'    成交量: {latest_row["成交量"]:,}股')
                print(f'    日期: {latest_row["日期"]}')
            else:
                print(f'  ✗ {stock_code_with_exchange}: 无数据')

        except Exception as e:
            print(f'  ✗ {stock_code_with_exchange}: 错误 - {e}')

        print()

    # 创建DataFrame并保存
    if results:
        df = pd.DataFrame(results)
        df.to_csv('test_stock_prices.csv', index=False, encoding='utf-8-sig')
        print(f'✓ 测试结果已保存到 test_stock_prices.csv')

        # 打印汇总表格
        print("\n" + "="*80)
        print("📈 股价获取测试结果")
        print("="*80)
        print(f"{'股票代码':<12} {'当前价格':<8} {'涨跌幅':<8} {'涨跌额':<8} {'成交量(万)':<10}")
        print("-"*80)

        for result in results:
            volume_wan = result['volume'] / 10000 if result['volume'] > 0 else 0
            change_pct_str = f"{result['change_pct']:+.2f}%"
            change_amount_str = f"{result['change_amount']:+.2f}"

            print(f"{result['stock_code']:<12} {result['current_price']:<8.2f} "
                  f"{change_pct_str:<8} {change_amount_str:<8} {volume_wan:<10.0f}")

        print("="*80)
    else:
        print('✗ 没有成功获取任何股价数据')

if __name__ == "__main__":
    test_stock_price_fetching()
