#!/usr/bin/env python3
"""
Qlib股价查询工具
=================

查询本地qlib数据中的个股股价信息

Usage:
    python query_qlib_stock_price.py SH600000
    python query_qlib_stock_price.py SH600000 --date 2024-01-15
    python query_qlib_stock_price.py SH600941 --range 2023-01-01 2023-12-31
    python query_qlib_stock_price.py --check-calendar  # 检查交易日历
"""

import os
import sys
import argparse
import qlib
import pandas as pd
from qlib.data import D
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QlibStockQuery:
    """Qlib股价查询类"""

    def __init__(self, data_path=None):
        """初始化qlib"""
        if data_path is None:
            data_path = os.path.expanduser("~/.qlib/qlib_data/cn_data")

        try:
            qlib.init(provider_uri=data_path, region="cn")
            logger.info(f"Qlib初始化成功，数据路径: {data_path}")
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            sys.exit(1)

    def check_trading_calendar(self, start_date="2025-06-01", end_date="2025-06-10"):
        """检查交易日历"""
        try:
            print(f"\n📅 检查交易日历 ({start_date} 到 {end_date})")
            print("="*60)

            # 获取完整交易日历
            calendar = D.calendar()

            # 转换为日期范围
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            # 生成日期范围内的所有日期
            all_dates = pd.date_range(start, end, freq='D')

            print(f"{'日期':<12} {'星期':<8} {'是否交易日':<10} {'说明'}")
            print("-"*60)

            for date in all_dates:
                weekday = date.strftime('%A')
                weekday_cn = {
                    'Monday': '周一', 'Tuesday': '周二', 'Wednesday': '周三',
                    'Thursday': '周四', 'Friday': '周五', 'Saturday': '周六', 'Sunday': '周日'
                }[weekday]

                is_trading = date in calendar
                status = "✓ 交易日" if is_trading else "✗ 非交易日"

                # 判断非交易日原因
                reason = ""
                if not is_trading:
                    if weekday in ['Saturday', 'Sunday']:
                        reason = "(周末)"
                    else:
                        reason = "(节假日)"

                print(f"{date.strftime('%Y-%m-%d'):<12} {weekday_cn:<8} {status:<10} {reason}")

            print("-"*60)
            print(f"最新交易日: {calendar[-1].strftime('%Y-%m-%d')}")
            print(f"数据覆盖范围: {calendar[0].strftime('%Y-%m-%d')} 到 {calendar[-1].strftime('%Y-%m-%d')}")
            print(f"总交易日数: {len(calendar)}")

        except Exception as e:
            logger.error(f"检查交易日历失败: {e}")

    def get_latest_trading_date(self, stock_code=None):
        """获取最新交易日期"""
        try:
            # 获取交易日历
            calendar = D.calendar()
            if len(calendar) > 0:
                latest_date = calendar[-1]

                if stock_code:
                    # 验证该股票在最新交易日是否有数据
                    data = D.features([stock_code], ['$close'], start_time=latest_date, end_time=latest_date)
                    if not data.empty and not data.isna().all().all():
                        return latest_date

                    # 如果最新交易日没有数据，向前查找
                    for i in range(1, min(10, len(calendar))):
                        check_date = calendar[-(i+1)]
                        data = D.features([stock_code], ['$close'], start_time=check_date, end_time=check_date)
                        if not data.empty and not data.isna().all().all():
                            return check_date

                return latest_date

            return None
        except Exception as e:
            logger.error(f"获取最新交易日期失败: {e}")
            return None

    def is_trading_day(self, date):
        """检查指定日期是否为交易日"""
        try:
            calendar = D.calendar()
            check_date = pd.to_datetime(date)
            return check_date in calendar
        except Exception as e:
            logger.error(f"检查交易日失败: {e}")
            return False

    def find_nearest_trading_day(self, date, direction='backward'):
        """查找最近的交易日"""
        try:
            calendar = D.calendar()
            target_date = pd.to_datetime(date)

            if target_date in calendar:
                return target_date

            # 向前或向后查找
            if direction == 'backward':
                # 向前查找（更早的日期）
                for i in range(1, 30):  # 最多查找30天
                    check_date = target_date - pd.Timedelta(days=i)
                    if check_date in calendar:
                        return check_date
            else:
                # 向后查找（更晚的日期）
                for i in range(1, 30):
                    check_date = target_date + pd.Timedelta(days=i)
                    if check_date in calendar:
                        return check_date

            return None
        except Exception as e:
            logger.error(f"查找最近交易日失败: {e}")
            return None

    def query_stock_price(self, stock_code, date=None, fields=None):
        """查询单只股票的价格信息"""
        if fields is None:
            fields = ['$open', '$high', '$low', '$close', '$volume', '$amount']

        try:
            # 标准化股票代码
            stock_code = self._normalize_stock_code(stock_code)

            if date is None:
                # 获取最新交易日期
                date = self.get_latest_trading_date(stock_code)
                if date is None:
                    logger.error("无法获取最新交易日期")
                    return None
                logger.info(f"使用最新交易日期: {date}")
            else:
                # 检查指定日期是否为交易日
                target_date = pd.to_datetime(date)
                if not self.is_trading_day(target_date):
                    logger.warning(f"指定日期 {target_date.strftime('%Y-%m-%d')} 不是交易日")

                    # 查找最近的交易日
                    nearest_date = self.find_nearest_trading_day(target_date, 'backward')
                    if nearest_date:
                        logger.info(f"使用最近的交易日: {nearest_date.strftime('%Y-%m-%d')}")
                        date = nearest_date
                    else:
                        logger.error("找不到有效的交易日")
                        return None

            # 查询数据
            data = D.features([stock_code], fields, start_time=date, end_time=date)

            if data.empty:
                logger.warning(f"股票 {stock_code} 在日期 {date} 没有数据")
                return None

            # 提取数据 - 处理MultiIndex情况
            try:
                if isinstance(data.index, pd.MultiIndex):
                    # MultiIndex情况：(date, instrument)
                    stock_data = data.loc[(date, stock_code)]
                else:
                    # 单层索引情况
                    stock_data = data.iloc[0]
            except (KeyError, IndexError):
                # 尝试其他方式提取数据
                try:
                    stock_data = data.iloc[0]
                except IndexError:
                    logger.warning(f"无法提取股票 {stock_code} 的数据")
                    return None

            # 确保stock_data是Series或array-like
            if hasattr(stock_data, 'values'):
                values = stock_data.values
            elif hasattr(stock_data, '__getitem__'):
                values = stock_data
            else:
                values = [stock_data]

            # 格式化结果
            result = {
                'stock_code': stock_code,
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'open': float(values[0]) if len(values) > 0 and not pd.isna(values[0]) else None,
                'high': float(values[1]) if len(values) > 1 and not pd.isna(values[1]) else None,
                'low': float(values[2]) if len(values) > 2 and not pd.isna(values[2]) else None,
                'close': float(values[3]) if len(values) > 3 and not pd.isna(values[3]) else None,
                'volume': int(values[4]) if len(values) > 4 and not pd.isna(values[4]) else None,
                'amount': float(values[5]) if len(values) > 5 and not pd.isna(values[5]) else None,
            }

            return result

        except Exception as e:
            logger.error(f"查询股票 {stock_code} 价格失败: {e}")
            return None

    def _normalize_stock_code(self, stock_code):
        """标准化股票代码格式"""
        stock_code = stock_code.upper().strip()

        # 如果只有6位数字，自动添加交易所前缀
        if stock_code.isdigit() and len(stock_code) == 6:
            if stock_code.startswith('6'):
                stock_code = f"SH{stock_code}"
            elif stock_code.startswith(('0', '3')):
                stock_code = f"SZ{stock_code}"

        return stock_code

    def print_stock_info(self, result):
        """格式化打印股票信息"""
        if not result:
            print("❌ 未找到股票数据")
            return

        print("\n" + "="*60)
        print(f"📈 股票信息查询结果")
        print("="*60)
        print(f"股票代码: {result['stock_code']}")
        print(f"交易日期: {result['date']}")
        print("-"*60)

        if result['open'] is not None:
            print(f"开盘价:   {result['open']:.2f}")
        if result['high'] is not None:
            print(f"最高价:   {result['high']:.2f}")
        if result['low'] is not None:
            print(f"最低价:   {result['low']:.2f}")
        if result['close'] is not None:
            print(f"收盘价:   {result['close']:.2f}")

        if result['volume'] is not None:
            print(f"成交量:   {result['volume']:,} 股")
        if result['amount'] is not None:
            print(f"成交额:   {result['amount']:,.2f}")

        # 计算涨跌幅（如果有开盘和收盘价）
        if result['open'] and result['close'] and result['open'] > 0:
            change_pct = (result['close'] - result['open']) / result['open'] * 100
            change_amount = result['close'] - result['open']
            print(f"日涨跌:   {change_amount:+.2f} ({change_pct:+.2f}%)")

        # 计算振幅
        if result['high'] and result['low'] and result['low'] > 0:
            amplitude = (result['high'] - result['low']) / result['low'] * 100
            print(f"振幅:     {amplitude:.2f}%")

        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Qlib股价查询工具')
    parser.add_argument('stock_code', nargs='?', help='股票代码 (如: SH600000, 600000)')
    parser.add_argument('--date', '-d', type=str, help='查询日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--check-calendar', '-c', action='store_true', help='检查交易日历')
    parser.add_argument('--data-path', type=str, help='qlib数据路径')
    parser.add_argument('--export', '-e', type=str, help='导出到CSV文件')

    args = parser.parse_args()

    # 创建查询器
    query = QlibStockQuery(args.data_path)

    try:
        if args.check_calendar:
            # 检查交易日历
            query.check_trading_calendar()
            return

        if not args.stock_code:
            print("❌ 请指定股票代码或使用 --check-calendar 查看交易日历")
            parser.print_help()
            return

        # 查询单日数据
        target_date = None
        if args.date:
            try:
                target_date = pd.to_datetime(args.date)
                print(f"🔍 查询股票 {args.stock_code} 在 {args.date} 的价格...")
            except Exception as e:
                print(f"❌ 日期格式错误: {e}")
                return
        else:
            print(f"🔍 查询股票 {args.stock_code} 的最新价格...")

        result = query.query_stock_price(args.stock_code, target_date)
        query.print_stock_info(result)

        # 导出CSV
        if args.export and result:
            df = pd.DataFrame([result])
            df.to_csv(args.export, index=False, encoding='utf-8-sig')
            print(f"✅ 数据已导出到: {args.export}")

    except KeyboardInterrupt:
        print("\n👋 查询已取消")
    except Exception as e:
        logger.error(f"查询失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()