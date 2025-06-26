#!/usr/bin/env python3
"""
实时交易模拟单元测试
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from ai_stock_selector import AIStockSelector


class TestLiveTrade(unittest.TestCase):
    """实时交易模拟测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model_path': os.path.join(self.temp_dir, 'models'),
            'output_path': os.path.join(self.temp_dir, 'output'),
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'train_end': '2024-06-30',
            'valid_end': '2024-09-30',
        }
        
        # 创建必要目录
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(self.config['output_path'], exist_ok=True)
        
        self.selector = AIStockSelector(self.config)
        self.selector.capital = 100000
        self.selector.topk = 5
        self.selector.interval = 1  # 1秒间隔用于测试
        
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trading_time_detection(self):
        """测试交易时间判断"""
        # 测试工作日交易时间
        weekday_morning = datetime(2024, 6, 19, 10, 30)  # 周三上午10:30
        self.assertTrue(self.selector.is_trading_time(weekday_morning))
        
        weekday_afternoon = datetime(2024, 6, 19, 14, 30)  # 周三下午14:30
        self.assertTrue(self.selector.is_trading_time(weekday_afternoon))
        
        # 测试非交易时间
        weekday_lunch = datetime(2024, 6, 19, 12, 30)  # 午休时间
        self.assertFalse(self.selector.is_trading_time(weekday_lunch))
        
        weekend = datetime(2024, 6, 22, 10, 30)  # 周六
        self.assertFalse(self.selector.is_trading_time(weekend))
        
        night = datetime(2024, 6, 19, 18, 0)  # 晚上
        self.assertFalse(self.selector.is_trading_time(night))
    
    def test_trading_day_detection(self):
        """测试交易日判断"""
        # 工作日
        weekday = datetime(2024, 6, 19)  # 周三
        self.assertTrue(self.selector.is_trading_day(weekday))
        
        # 周末
        saturday = datetime(2024, 6, 22)  # 周六
        self.assertFalse(self.selector.is_trading_day(saturday))
        
        sunday = datetime(2024, 6, 23)  # 周日
        self.assertFalse(self.selector.is_trading_day(sunday))
        
        # 节假日
        new_year = datetime(2024, 1, 1)  # 元旦
        self.assertFalse(self.selector.is_trading_day(new_year))
    
    def create_mock_quotes(self, stock_codes):
        """创建模拟行情数据"""
        quotes = {}
        base_price = 20.0
        
        for i, stock_code in enumerate(stock_codes):
            price = base_price + i * 5  # 不同股票不同价格
            quotes[stock_code] = {
                'current_price': price,
                'volume': 1000000,
                'amount': price * 1000000,
                'high': price * 1.05,
                'low': price * 0.95,
                'open': price * 1.02,
                'prev_close': price * 0.98,
                'change_pct': 2.0,
                'timestamp': datetime.now()
            }
        
        return quotes
    
    def create_mock_predictions(self):
        """创建模拟预测数据"""
        # 模拟预测数据结构
        stock_codes = ['SH600000', 'SH600001', 'SZ000001', 'SZ000002', 'SH600002']
        scores = [0.15, 0.12, 0.10, 0.08, 0.05]
        
        # 创建MultiIndex
        dates = [datetime.now().date()]
        index = pd.MultiIndex.from_product([dates, stock_codes], names=['date', 'instrument'])
        
        # 创建预测Series
        pred_data = []
        for score in scores:
            pred_data.append(score)
        
        predictions = pd.Series(pred_data, index=index)
        return {'ensemble': predictions}
    
    @patch('ai_stock_selector.AIStockSelector.get_realtime_quotes')
    @patch('ai_stock_selector.AIStockSelector.predict')
    @patch('ai_stock_selector.AIStockSelector.select_stocks')
    @patch('ai_stock_selector.AIStockSelector._load_models')
    @patch('ai_stock_selector.AIStockSelector._init_qlib')
    def test_live_trade_simulation(self, mock_init_qlib, mock_load_models, 
                                 mock_select_stocks, mock_predict, mock_get_quotes):
        """测试实时交易模拟（伪行情桩函数）"""
        
        # 设置模拟数据
        stock_codes = ['SH600000', 'SH600001', 'SZ000001', 'SZ000002', 'SH600002']
        
        # 模拟3轮交易数据
        mock_predictions_data = self.create_mock_predictions()
        mock_predict.return_value = mock_predictions_data
        
        mock_quotes_data = self.create_mock_quotes(stock_codes)
        mock_get_quotes.return_value = mock_quotes_data
        
        # 模拟选股结果
        mock_weights = pd.Series([0.3, 0.3, 0.4], index=['SH600000', 'SH600001', 'SZ000001'])
        mock_select_stocks.return_value = mock_weights
        
        # 模拟交易时间检查（总是返回True以便测试）
        with patch.object(self.selector, 'is_trading_day', return_value=True), \
             patch.object(self.selector, 'is_trading_time', return_value=True):
            
            # 创建一个计数器来限制循环次数
            call_count = [0]  # 使用列表以便在闭包中修改
            
            def side_effect_predict():
                call_count[0] += 1
                if call_count[0] > 3:  # 限制为3轮
                    raise KeyboardInterrupt("测试完成")
                return mock_predictions_data
            
            mock_predict.side_effect = side_effect_predict
            
            try:
                # 运行实时交易（会被KeyboardInterrupt中断）
                self.selector.live_trade()
            except KeyboardInterrupt:
                pass  # 预期的中断
        
        # 验证文件是否创建
        output_path = Path(self.config['output_path'])
        trade_log_file = output_path / 'trade_log.csv'
        portfolio_nav_file = output_path / 'portfolio_nav.csv'
        
        self.assertTrue(trade_log_file.exists(), "交易日志文件应该被创建")
        self.assertTrue(portfolio_nav_file.exists(), "投资组合净值文件应该被创建")
        
        # 验证交易日志内容
        if trade_log_file.exists():
            trade_log = pd.read_csv(trade_log_file)
            self.assertGreater(len(trade_log), 0, "交易日志应该有记录")
            
            # 验证列名
            expected_columns = ['date', 'time', 'code', 'price', 'qty', 'action', 'cash', 'position_value']
            for col in expected_columns:
                self.assertIn(col, trade_log.columns, f"交易日志应该包含列: {col}")
        
        # 验证投资组合净值
        if portfolio_nav_file.exists():
            portfolio_nav = pd.read_csv(portfolio_nav_file)
            self.assertGreater(len(portfolio_nav), 0, "投资组合净值应该有记录")
            
            # 验证列名
            expected_columns = ['date', 'time', 'nav', 'drawdown', 'turnover', 'cash', 'position_value']
            for col in expected_columns:
                self.assertIn(col, portfolio_nav.columns, f"投资组合净值应该包含列: {col}")
            
            # 验证净值非递减（考虑到可能的交易成本，改为检查净值合理性）
            navs = portfolio_nav['nav'].values
            self.assertTrue(all(nav > 0 for nav in navs), "净值应该为正数")
            self.assertTrue(abs(navs[0] - self.selector.capital) < 1000, "初始净值应该接近初始资金")
    
    def test_execute_trades(self):
        """测试交易执行"""
        # 创建模拟数据
        stock_codes = ['SH600000', 'SH600001']
        target_weights = pd.Series([0.5, 0.5], index=stock_codes)
        current_quotes = self.create_mock_quotes(stock_codes)
        
        portfolio = {}
        cash = 100000
        total_value = 100000
        
        # 创建临时交易日志文件
        trade_log_file = Path(self.config['output_path']) / 'test_trade_log.csv'
        with open(trade_log_file, 'w') as f:
            f.write('date,time,code,price,qty,action,cash,position_value\n')
        
        current_time = datetime.now()
        
        # 执行交易
        turnover, updated_cash = self.selector._execute_trades(
            target_weights, current_quotes, portfolio, cash, total_value,
            trade_log_file, current_time
        )
        
        # 验证结果
        self.assertGreater(turnover, 0, "应该产生换手率")
        self.assertGreater(len(portfolio), 0, "应该有持仓")
        
        # 验证权重限制（20%）
        for stock_code in portfolio:
            if stock_code in current_quotes:
                position_value = portfolio[stock_code] * current_quotes[stock_code]['current_price']
                weight = position_value / total_value
                self.assertLessEqual(weight, 0.21, f"股票 {stock_code} 权重不应超过20%（允许1%误差）")
    
    def test_visualize_live_trade(self):
        """测试实时交易可视化"""
        # 创建模拟交易数据
        output_path = Path(self.config['output_path'])
        
        # 创建模拟交易日志
        trade_log_data = {
            'date': ['2024-06-19', '2024-06-19', '2024-06-19'],
            'time': ['09:30:00', '09:31:00', '09:32:00'],
            'code': ['SH600000', 'SH600001', 'SH600000'],
            'price': [20.0, 25.0, 20.1],
            'qty': [1000, 800, -500],
            'action': ['BUY', 'BUY', 'SELL'],
            'cash': [80000, 60000, 70000],
            'position_value': [20000, 40000, 30000]
        }
        trade_log_df = pd.DataFrame(trade_log_data)
        trade_log_df.to_csv(output_path / 'trade_log.csv', index=False)
        
        # 创建模拟投资组合净值
        portfolio_nav_data = {
            'date': ['2024-06-19', '2024-06-19', '2024-06-19'],
            'time': ['09:30:00', '09:31:00', '09:32:00'],
            'nav': [100000, 100500, 100200],
            'drawdown': [0.0, 0.0, -0.003],
            'turnover': [0.0, 0.4, 0.2],
            'cash': [80000, 60000, 70000],
            'position_value': [20000, 40000, 30000]
        }
        portfolio_nav_df = pd.DataFrame(portfolio_nav_data)
        portfolio_nav_df.to_csv(output_path / 'portfolio_nav.csv', index=False)
        
        # 测试可视化函数
        try:
            result_path = self.selector.visualize_live_trade()
            if result_path:  # 如果Plotly可用
                self.assertTrue(os.path.exists(result_path), "可视化HTML文件应该被创建")
        except ImportError:
            # Plotly未安装时跳过测试
            pass


class TestTradingConstraints(unittest.TestCase):
    """交易约束测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model_path': os.path.join(self.temp_dir, 'models'),
            'output_path': os.path.join(self.temp_dir, 'output'),
        }
        self.selector = AIStockSelector(self.config)
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_weight_constraints(self):
        """测试权重约束（单股≤20%）"""
        # 创建目标权重（超过20%的权重）
        target_weights = pd.Series([0.5, 0.3, 0.2], index=['SH600000', 'SH600001', 'SZ000001'])
        
        # 模拟当前行情
        current_quotes = {
            'SH600000': {'current_price': 20.0},
            'SH600001': {'current_price': 25.0},
            'SZ000001': {'current_price': 30.0}
        }
        
        portfolio = {}
        cash = 100000
        total_value = 100000
        
        # 创建临时文件
        trade_log_file = Path(self.config['output_path']) / 'test_trade_log.csv'
        os.makedirs(self.config['output_path'], exist_ok=True)
        with open(trade_log_file, 'w') as f:
            f.write('date,time,code,price,qty,action,cash,position_value\n')
        
        current_time = datetime.now()
        
        # 执行交易
        turnover, updated_cash = self.selector._execute_trades(
            target_weights, current_quotes, portfolio, cash, total_value,
            trade_log_file, current_time
        )
        
        # 验证权重约束
        for stock_code in portfolio:
            if stock_code in current_quotes:
                position_value = portfolio[stock_code] * current_quotes[stock_code]['current_price']
                actual_weight = position_value / total_value
                self.assertLessEqual(actual_weight, 0.21, 
                                   f"股票 {stock_code} 实际权重 {actual_weight:.2%} 超过20%限制")
    
    def test_transaction_fees(self):
        """测试交易费用计算（新分级费率结构）"""
        os.makedirs(self.config['output_path'], exist_ok=True)
        current_time = datetime.now()
        
        # 测试小额交易（<20000元）- 固定5元手续费
        stock_code = 'SH600000'
        quantity = 500  # 500股 × 20元 = 10000元 < 20000元
        quote = {'current_price': 20.0}
        portfolio = {}
        trade_log_file = Path(self.config['output_path']) / 'test_small_trade.csv'
        with open(trade_log_file, 'w') as f:
            f.write('date,time,code,price,qty,action,cash,position_value\n')
        
        # 测试买入费用计算
        commission = self.selector._calculate_commission(quantity * quote['current_price'])
        self.assertEqual(commission, 5.0, "小额交易应该收取固定5元手续费")
        
        # 执行买入
        total_cost = self.selector._execute_buy(stock_code, quantity, quote, portfolio, trade_log_file, current_time)
        expected_cost = 10000 + 5.0  # 交易金额 + 固定手续费
        self.assertEqual(total_cost, expected_cost, "小额买入总成本应该正确")
        
        # 测试大额交易（≥20000元）- 按0.025%收取
        large_stock_code = 'SH600001'
        large_quantity = 2000  # 2000股 × 20元 = 40000元 ≥ 20000元
        large_quote = {'current_price': 20.0}
        large_portfolio = {}
        large_trade_log_file = Path(self.config['output_path']) / 'test_large_trade.csv'
        with open(large_trade_log_file, 'w') as f:
            f.write('date,time,code,price,qty,action,cash,position_value\n')
        
        # 测试买入费用计算
        large_trade_value = large_quantity * large_quote['current_price']  # 40000元
        large_commission = self.selector._calculate_commission(large_trade_value)
        expected_large_commission = 40000 * 0.00025  # 40000 × 0.025% = 10元
        self.assertEqual(large_commission, expected_large_commission, "大额交易应该按0.025%收取手续费")
        
        # 执行买入
        large_total_cost = self.selector._execute_buy(large_stock_code, large_quantity, large_quote, 
                                                    large_portfolio, large_trade_log_file, current_time)
        expected_large_cost = 40000 + 10.0  # 交易金额 + 0.025%手续费
        self.assertEqual(large_total_cost, expected_large_cost, "大额买入总成本应该正确")
        
        # 验证边界值（正好20000元）
        boundary_commission = self.selector._calculate_commission(20000.0)
        expected_boundary_commission = 20000 * 0.00025  # 20000 × 0.025% = 5元
        self.assertEqual(boundary_commission, expected_boundary_commission, "边界值20000元应该按0.025%收取")
        
        # 验证持仓
        self.assertEqual(portfolio[stock_code], quantity, "买入后持仓数量应该正确")
        self.assertEqual(large_portfolio[large_stock_code], large_quantity, "大额买入后持仓数量应该正确")


if __name__ == '__main__':
    # 设置测试环境
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
    
    # 运行测试
    unittest.main(verbosity=2)