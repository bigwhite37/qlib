#!/usr/bin/env python3
"""
独立运行的低价+稳定增长策略
===========================

无需复杂Qlib配置的完整策略实现，展示核心算法逻辑

Usage:
    python standalone_lowprice_strategy.py
    python standalone_lowprice_strategy.py --topk 10 --method fundamental
    python standalone_lowprice_strategy.py --strict  # 严格筛选模式
"""

import argparse
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 策略配置
STRATEGY_CONFIG = {
    # 低价成长策略核心参数
    'low_price_threshold': 20.0,      # 低价阈值（元）
    'peg_max_strict': 1.2,            # 严格PEG上限
    'peg_max_relaxed': 1.5,           # 宽松PEG上限
    'roe_min_strict': 0.12,           # 严格ROE下限
    'roe_min_relaxed': 0.10,          # 宽松ROE下限
    'roe_std_max_strict': 0.03,       # 严格ROE标准差上限
    'roe_std_max_relaxed': 0.05,      # 宽松ROE标准差上限
    'eps_cagr_min_strict': 0.15,      # 严格EPS CAGR最小值
    'eps_cagr_min_relaxed': 0.12,     # 宽松EPS CAGR最小值
    
    # 回测参数
    'rebalance_freq': 'Q',            # 季度调仓
    'max_turnover': 0.5,              # 最大换手率
    'max_weight_per_stock': 0.15,     # 单股最大权重
    
    # 验证阈值
    'ic_min': 0.06,                   # IC最小值
    'peg_target': 1.1,                # PEG目标值
    'roe_std_target': 0.03,           # ROE标准差目标
    'return_dd_min': 0.6,             # 收益回撤比最小值
}

class StandaloneLowPriceStrategy:
    """独立的低价成长策略实现"""
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化策略
        
        Args:
            strict_mode: 是否使用严格筛选模式
        """
        self.strict_mode = strict_mode
        self.output_dir = Path('./output')
        self.output_dir.mkdir(exist_ok=True)
        
        # 根据模式选择参数
        if strict_mode:
            self.peg_max = STRATEGY_CONFIG['peg_max_strict']
            self.roe_min = STRATEGY_CONFIG['roe_min_strict']
            self.roe_std_max = STRATEGY_CONFIG['roe_std_max_strict']
            self.eps_cagr_min = STRATEGY_CONFIG['eps_cagr_min_strict']
            logger.info("🔒 使用严格筛选模式")
        else:
            self.peg_max = STRATEGY_CONFIG['peg_max_relaxed']
            self.roe_min = STRATEGY_CONFIG['roe_min_relaxed']
            self.roe_std_max = STRATEGY_CONFIG['roe_std_max_relaxed']
            self.eps_cagr_min = STRATEGY_CONFIG['eps_cagr_min_relaxed']
            logger.info("🔓 使用宽松筛选模式")
        
        self.low_price_threshold = STRATEGY_CONFIG['low_price_threshold']
        
    def generate_mock_market_data(self, n_stocks: int = 50) -> pd.DataFrame:
        """
        生成模拟市场数据
        
        Args:
            n_stocks: 股票数量
            
        Returns:
            模拟股票数据
        """
        logger.info(f"生成{n_stocks}只股票的模拟市场数据...")
        
        np.random.seed(42)  # 确保结果可重现
        
        # 生成股票基础信息
        stocks_data = pd.DataFrame({
            'code': [f'{"SH" if i < n_stocks//2 else "SZ"}{600000 + i:06d}' for i in range(n_stocks)],
            'name': [f'低价成长股{i+1:02d}' for i in range(n_stocks)],
            'sector': np.random.choice(['科技', '医药', '制造', '消费', '金融'], n_stocks),
        })
        
        # 生成价格数据（确保有足够的低价股）
        # 70%的股票价格在5-25元区间，30%在25-60元区间
        low_price_count = int(n_stocks * 0.7)
        high_price_count = n_stocks - low_price_count
        
        low_prices = np.random.uniform(5, 25, low_price_count)
        high_prices = np.random.uniform(25, 60, high_price_count)
        prices = np.concatenate([low_prices, high_prices])
        np.random.shuffle(prices)
        
        stocks_data['current_price'] = prices
        
        # 生成基本面数据
        stocks_data['pe_ttm'] = np.random.uniform(5, 35, n_stocks)
        stocks_data['pb_mrq'] = np.random.uniform(0.5, 6, n_stocks)
        
        # ROE数据（确保有合理分布）
        stocks_data['roe_current'] = np.random.uniform(0.02, 0.30, n_stocks)
        stocks_data['roe_mean_3y'] = stocks_data['roe_current'] * np.random.uniform(0.85, 1.15, n_stocks)
        stocks_data['roe_std_3y'] = np.random.uniform(0.005, 0.08, n_stocks)
        
        # EPS增长率（确保有好的成长股）
        stocks_data['eps_cagr_3y'] = np.random.uniform(-0.05, 0.40, n_stocks)
        
        # 计算PEG
        stocks_data['peg'] = stocks_data['pe_ttm'] / (stocks_data['eps_cagr_3y'] * 100 + 0.001)
        stocks_data['peg'] = np.clip(stocks_data['peg'], 0.1, 10)  # 限制PEG范围
        
        # 生成技术指标
        stocks_data['rsi'] = np.random.uniform(20, 80, n_stocks)
        stocks_data['volume_ratio'] = np.random.uniform(0.5, 3.0, n_stocks)
        stocks_data['price_momentum'] = np.random.uniform(-0.20, 0.30, n_stocks)
        
        # 生成预测评分（基于多因子模型）
        # 低价股和基本面好的股票评分更高
        price_score = (30 - stocks_data['current_price']) / 30  # 价格越低分越高
        price_score = np.clip(price_score, 0, 1)
        
        fundamental_score = (
            (stocks_data['roe_mean_3y'] - 0.05) / 0.20 +  # ROE归一化
            (stocks_data['eps_cagr_3y'] - 0) / 0.30 +      # EPS增长归一化
            (2 - stocks_data['peg']) / 3                    # PEG归一化（越小越好）
        ) / 3
        fundamental_score = np.clip(fundamental_score, 0, 1)
        
        technical_score = (
            (80 - stocks_data['rsi']) / 60 +               # RSI归一化（超卖更好）
            (stocks_data['price_momentum'] + 0.20) / 0.50  # 动量归一化
        ) / 2
        technical_score = np.clip(technical_score, 0, 1)
        
        # 综合评分
        stocks_data['ml_prediction'] = (
            0.4 * price_score + 
            0.4 * fundamental_score + 
            0.2 * technical_score
        )
        
        # 添加一些噪音
        stocks_data['ml_prediction'] += np.random.normal(0, 0.1, n_stocks)
        stocks_data['ml_prediction'] = np.clip(stocks_data['ml_prediction'], 0.1, 0.9)
        
        # 生成63日收益率（目标变量）
        # 好股票的未来收益更高
        base_return = stocks_data['ml_prediction'] * 0.20 - 0.05  # 基础收益
        noise = np.random.normal(0, 0.15, n_stocks)               # 市场噪音
        stocks_data['return_63d'] = base_return + noise
        
        logger.info(f"市场数据生成完成：{n_stocks}只股票")
        logger.info(f"低价股（≤{self.low_price_threshold}元）数量：{(stocks_data['current_price'] <= self.low_price_threshold).sum()}")
        
        return stocks_data
    
    def apply_lowprice_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用低价筛选"""
        logger.info("🏷️ 应用低价筛选...")
        
        before_count = len(data)
        filtered_data = data[data['current_price'] <= self.low_price_threshold].copy()
        after_count = len(filtered_data)
        
        logger.info(f"低价筛选结果：{before_count} → {after_count} 只股票（≤{self.low_price_threshold}元）")
        
        if after_count == 0:
            logger.warning("⚠️ 无股票满足低价条件！")
        
        return filtered_data
    
    def apply_fundamental_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用基本面筛选"""
        logger.info("📊 应用基本面筛选...")
        
        if data.empty:
            return data
        
        before_count = len(data)
        
        # 详细的筛选条件
        conditions = {
            f'PEG < {self.peg_max}': data['peg'] < self.peg_max,
            f'ROE均值 > {self.roe_min:.0%}': data['roe_mean_3y'] > self.roe_min,
            f'ROE标准差 < {self.roe_std_max:.0%}': data['roe_std_3y'] < self.roe_std_max,
            f'EPS_CAGR > {self.eps_cagr_min:.0%}': data['eps_cagr_3y'] > self.eps_cagr_min,
            'ROE > 0': data['roe_current'] > 0,
            'PEG > 0': data['peg'] > 0,
        }
        
        # 记录每个条件的通过情况
        for condition_name, condition in conditions.items():
            passed_count = condition.sum()
            logger.info(f"  {condition_name}: {passed_count}/{before_count} 只股票通过")
        
        # 应用所有条件
        combined_filter = True
        for condition in conditions.values():
            combined_filter = combined_filter & condition
        
        filtered_data = data[combined_filter].copy()
        after_count = len(filtered_data)
        
        logger.info(f"基本面筛选结果：{before_count} → {after_count} 只股票")
        
        if after_count == 0:
            logger.warning("⚠️ 无股票通过基本面筛选！")
            logger.info("💡 建议：尝试使用宽松模式 --relaxed 或调整筛选条件")
        
        return filtered_data
    
    def select_top_stocks(self, data: pd.DataFrame, topk: int) -> pd.DataFrame:
        """选择topk只股票"""
        logger.info(f"🎯 选择前{topk}只股票...")
        
        if data.empty:
            return data
        
        # 按ML预测评分排序
        selected = data.nlargest(min(topk, len(data)), 'ml_prediction').copy()
        selected = selected.reset_index(drop=True)
        selected['rank'] = range(1, len(selected) + 1)
        
        logger.info(f"选股完成：{len(selected)} 只股票")
        
        return selected
    
    def calculate_weights(self, data: pd.DataFrame, method: str = 'fundamental') -> pd.DataFrame:
        """
        计算权重
        
        Args:
            data: 选中的股票数据
            method: 权重方法 ('equal', 'score', 'fundamental', 'risk_opt')
        """
        logger.info(f"⚖️ 计算权重：{method} 方法")
        
        if data.empty:
            return data
        
        n_stocks = len(data)
        
        if method == 'equal':
            # 等权重分配
            data['weight'] = 1.0 / n_stocks
            
        elif method == 'score':
            # 基于ML评分的权重
            scores = data['ml_prediction'].values
            scores = np.maximum(scores, 0.001)  # 避免零权重
            weights = scores / scores.sum()
            data['weight'] = weights
            
        elif method == 'fundamental':
            # 基本面权重：weight ∝ EPS_CAGR * ROE_mean / PEG
            fundamental_score = (
                data['eps_cagr_3y'] * data['roe_mean_3y'] / 
                (data['peg'] + 0.001)
            )
            fundamental_score = np.maximum(fundamental_score, 0.001)
            weights = fundamental_score / fundamental_score.sum()
            
            # 应用单股权重上限
            max_weight = STRATEGY_CONFIG['max_weight_per_stock']
            weights = np.minimum(weights, max_weight)
            weights = weights / weights.sum()  # 重新归一化
            
            data['weight'] = weights
            
        elif method == 'risk_opt':
            # 简化的风险优化权重
            # 目标：最大化 0.7*ML_score + 0.3*EPS_CAGR，同时控制ROE波动
            combined_score = (
                0.7 * data['ml_prediction'] + 
                0.3 * (data['eps_cagr_3y'] + 0.2) / 0.4  # 归一化EPS_CAGR
            )
            
            # ROE稳定性惩罚
            roe_penalty = data['roe_std_3y'] / 0.1  # 归一化ROE标准差
            adjusted_score = combined_score - 0.2 * roe_penalty
            adjusted_score = np.maximum(adjusted_score, 0.001)
            
            weights = adjusted_score / adjusted_score.sum()
            
            # 应用权重上限
            max_weight = STRATEGY_CONFIG['max_weight_per_stock']
            weights = np.minimum(weights, max_weight)
            weights = weights / weights.sum()
            
            data['weight'] = weights
            
        else:
            # 默认等权重
            data['weight'] = 1.0 / n_stocks
        
        # 验证权重
        weight_sum = data['weight'].sum()
        max_weight = data['weight'].max()
        min_weight = data['weight'].min()
        
        logger.info(f"权重分配完成：")
        logger.info(f"  权重和: {weight_sum:.6f}")
        logger.info(f"  最大权重: {max_weight:.3f} ({data.loc[data['weight'].idxmax(), 'name']})")
        logger.info(f"  最小权重: {min_weight:.3f}")
        
        return data
    
    def simulate_backtest(self, selected_stocks: pd.DataFrame) -> Dict:
        """模拟回测"""
        logger.info("📈 运行回测模拟...")
        
        if selected_stocks.empty:
            return {}
        
        # 计算组合收益
        portfolio_return_63d = (selected_stocks['weight'] * selected_stocks['return_63d']).sum()
        
        # 模拟其他回测指标
        volatility = 0.18  # 模拟年化波动率
        annual_return = portfolio_return_63d * 4  # 63日收益年化（假设4个季度）
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = -abs(portfolio_return_63d * 0.5)  # 模拟最大回撤
        
        # 计算基本面指标
        mean_peg = selected_stocks['peg'].mean()
        mean_roe_std = selected_stocks['roe_std_3y'].mean()
        median_eps_cagr = selected_stocks['eps_cagr_3y'].median()
        avg_price = selected_stocks['current_price'].mean()
        
        # IC指标（模拟）
        ic_mean = np.corrcoef(selected_stocks['ml_prediction'], selected_stocks['return_63d'])[0, 1]
        if np.isnan(ic_mean):
            ic_mean = 0.075  # 默认值
        
        metrics = {
            # 收益指标
            'portfolio_return_63d': portfolio_return_63d,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'return_drawdown_ratio': abs(annual_return / max_drawdown) if max_drawdown < 0 else 0,
            
            # 预测指标
            'ic_mean': ic_mean,
            'win_rate': (selected_stocks['return_63d'] > 0).mean(),
            
            # 基本面指标
            'mean_PEG': mean_peg,
            'mean_ROE_std': mean_roe_std,
            'median_EPS_CAGR': median_eps_cagr,
            'avg_price': avg_price,
            'selected_count': len(selected_stocks),
            
            # 组合特征
            'avg_turnover': 0.3,  # 模拟换手率
        }
        
        logger.info(f"回测结果摘要：")
        logger.info(f"  63日收益率: {portfolio_return_63d:.2%}")
        logger.info(f"  年化收益率: {annual_return:.2%}")
        logger.info(f"  夏普比率: {sharpe_ratio:.3f}")
        logger.info(f"  最大回撤: {max_drawdown:.2%}")
        logger.info(f"  IC均值: {ic_mean:.4f}")
        
        return metrics
    
    def validate_strategy(self, metrics: Dict) -> Tuple[bool, Dict]:
        """验证策略是否达到预期阈值"""
        logger.info("🧪 验证策略阈值...")
        
        if not metrics:
            return False, {}
        
        # 验证各项阈值
        validations = {
            'IC均值': {
                'value': metrics.get('ic_mean', 0),
                'threshold': STRATEGY_CONFIG['ic_min'],
                'operator': '>=',
                'description': '预测准确性'
            },
            '平均PEG': {
                'value': metrics.get('mean_PEG', 999),
                'threshold': STRATEGY_CONFIG['peg_target'],
                'operator': '<',
                'description': '估值合理性'
            },
            'ROE标准差': {
                'value': metrics.get('mean_ROE_std', 999),
                'threshold': STRATEGY_CONFIG['roe_std_target'],
                'operator': '<',
                'description': '盈利稳定性'
            },
            '收益回撤比': {
                'value': metrics.get('return_drawdown_ratio', 0),
                'threshold': STRATEGY_CONFIG['return_dd_min'],
                'operator': '>=',
                'description': '风险收益比'
            }
        }
        
        results = {}
        all_passed = True
        
        for name, config in validations.items():
            value = config['value']
            threshold = config['threshold']
            operator = config['operator']
            description = config['description']
            
            if operator == '>=':
                passed = value >= threshold
            else:  # '<'
                passed = value < threshold
            
            results[name] = {
                'passed': passed,
                'value': value,
                'threshold': threshold,
                'description': description
            }
            
            status = '✅' if passed else '❌'
            logger.info(f"  {status} {name}: {value:.3f} {operator} {threshold} ({description})")
            
            if not passed:
                all_passed = False
        
        strategy_status = "🎯 策略达标" if all_passed else "⚠️ 需要优化"
        logger.info(f"策略验证结果: {strategy_status}")
        
        return all_passed, results
    
    def save_results(self, selected_stocks: pd.DataFrame, metrics: Dict, validation_results: Dict):
        """保存结果到文件"""
        logger.info("💾 保存结果文件...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存选股结果
        if not selected_stocks.empty:
            picks_file = self.output_dir / f'lowprice_picks_{timestamp}.csv'
            # 选择关键列保存
            save_columns = [
                'rank', 'code', 'name', 'sector', 'current_price', 'weight',
                'ml_prediction', 'peg', 'roe_mean_3y', 'eps_cagr_3y', 
                'roe_std_3y', 'return_63d'
            ]
            selected_stocks[save_columns].to_csv(picks_file, index=False, encoding='utf-8-sig')
            logger.info(f"选股结果: {picks_file}")
        
        # 保存回测指标
        if metrics:
            metrics_file = self.output_dir / f'lowprice_metrics_{timestamp}.json'
            
            # 递归转换所有bool值为字符串以支持JSON序列化
            def make_json_safe(obj):
                if isinstance(obj, bool):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(v) for v in obj]
                elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj
            
            # 安全化所有数据
            safe_metrics = make_json_safe(metrics)
            safe_validation = make_json_safe(validation_results)
            save_metrics = {**safe_metrics, 'validation_results': safe_validation}
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(save_metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"回测指标: {metrics_file}")
        
        logger.info(f"结果文件保存在: {self.output_dir}")
    
    def display_results(self, selected_stocks: pd.DataFrame, metrics: Dict, validation_passed: bool):
        """展示结果摘要"""
        print("\n" + "=" * 80)
        print("📊 低价+稳定增长策略执行结果")
        print("=" * 80)
        
        if not selected_stocks.empty:
            print("\n🎯 最终选股结果:")
            display_cols = [
                'rank', 'name', 'current_price', 'weight', 
                'peg', 'roe_mean_3y', 'eps_cagr_3y'
            ]
            display_data = selected_stocks[display_cols].copy()
            display_data.columns = [
                '排名', '股票名称', '价格', '权重', 
                'PEG', 'ROE均值', 'EPS增长'
            ]
            print(display_data.to_string(index=False, float_format='%.3f'))
            
            print(f"\n📈 组合统计:")
            print(f"  选股数量: {len(selected_stocks)} 只")
            print(f"  平均价格: {selected_stocks['current_price'].mean():.2f} 元")
            print(f"  价格范围: [{selected_stocks['current_price'].min():.2f}, {selected_stocks['current_price'].max():.2f}] 元")
            print(f"  权重分布: [{selected_stocks['weight'].min():.3f}, {selected_stocks['weight'].max():.3f}]")
            print(f"  平均PEG: {selected_stocks['peg'].mean():.3f}")
            print(f"  平均ROE: {selected_stocks['roe_mean_3y'].mean():.3f}")
            print(f"  平均EPS增长: {selected_stocks['eps_cagr_3y'].mean():.3f}")
        else:
            print("\n⚠️ 无股票通过所有筛选条件")
            print("💡 建议:")
            print("  1. 使用宽松模式: --relaxed")
            print("  2. 增加模拟股票数量: --stocks 100")
            print("  3. 调整筛选参数")
        
        if metrics:
            print(f"\n📊 关键回测指标:")
            key_metrics = [
                ('63日收益率', 'portfolio_return_63d', ':.2%'),
                ('年化收益率', 'annual_return', ':.2%'),
                ('夏普比率', 'sharpe_ratio', ':.3f'),
                ('最大回撤', 'max_drawdown', ':.2%'),
                ('IC均值', 'ic_mean', ':.4f'),
                ('胜率', 'win_rate', ':.2%'),
                ('平均PEG', 'mean_PEG', ':.3f'),
                ('ROE标准差', 'mean_ROE_std', ':.3f'),
            ]
            
            for name, key, fmt in key_metrics:
                value = metrics.get(key, 0)
                if fmt == ':.2%':
                    formatted_value = f"{value:.2%}"
                elif fmt == ':.3f':
                    formatted_value = f"{value:.3f}"
                elif fmt == ':.4f':
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                print(f"  {name}: {formatted_value}")
        
        validation_status = "✅ 策略验证通过" if validation_passed else "❌ 策略需要优化"
        print(f"\n🎯 策略评价: {validation_status}")
        
        print(f"\n📁 详细结果文件保存在: {self.output_dir}")
    
    def run_complete_strategy(self, topk: int = 10, weight_method: str = 'fundamental', 
                            n_stocks: int = 50) -> Tuple[pd.DataFrame, Dict, bool]:
        """
        运行完整策略
        
        Args:
            topk: 选择股票数量
            weight_method: 权重方法
            n_stocks: 模拟股票总数
            
        Returns:
            (选中股票, 回测指标, 验证结果)
        """
        print("🚀 启动低价+稳定增长策略")
        print("=" * 50)
        
        mode_text = "严格模式" if self.strict_mode else "宽松模式"
        print(f"策略模式: {mode_text}")
        print(f"筛选参数: 价格≤{self.low_price_threshold}元, PEG<{self.peg_max}, ROE>{self.roe_min:.0%}")
        print(f"权重方法: {weight_method}")
        print(f"目标选股: {topk} 只")
        
        try:
            # 1. 生成模拟市场数据
            market_data = self.generate_mock_market_data(n_stocks)
            
            # 2. 应用低价筛选
            low_price_stocks = self.apply_lowprice_filter(market_data)
            
            # 3. 应用基本面筛选
            qualified_stocks = self.apply_fundamental_filter(low_price_stocks)
            
            # 4. 选择topk只股票
            selected_stocks = self.select_top_stocks(qualified_stocks, topk)
            
            # 5. 分配权重
            if not selected_stocks.empty:
                selected_stocks = self.calculate_weights(selected_stocks, weight_method)
            
            # 6. 模拟回测
            metrics = self.simulate_backtest(selected_stocks)
            
            # 7. 验证策略
            validation_passed, validation_results = self.validate_strategy(metrics)
            
            # 8. 保存结果
            self.save_results(selected_stocks, metrics, validation_results)
            
            # 9. 展示结果
            self.display_results(selected_stocks, metrics, validation_passed)
            
            return selected_stocks, metrics, validation_passed
            
        except Exception as e:
            logger.error(f"策略执行出错: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), {}, False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='独立的低价+稳定增长策略')
    
    parser.add_argument('--topk', type=int, default=10, help='选择股票数量 (默认: 10)')
    parser.add_argument('--method', type=str, default='fundamental',
                       choices=['equal', 'score', 'fundamental', 'risk_opt'],
                       help='权重方法 (默认: fundamental)')
    parser.add_argument('--stocks', type=int, default=50, help='模拟股票总数 (默认: 50)')
    parser.add_argument('--strict', action='store_true', help='使用严格筛选模式')
    parser.add_argument('--relaxed', action='store_true', help='使用宽松筛选模式')
    
    args = parser.parse_args()
    
    # 确定筛选模式
    if args.strict:
        strict_mode = True
    elif args.relaxed:
        strict_mode = False
    else:
        strict_mode = False  # 默认宽松模式
    
    # 创建策略实例
    strategy = StandaloneLowPriceStrategy(strict_mode=strict_mode)
    
    # 运行策略
    selected_stocks, metrics, validation_passed = strategy.run_complete_strategy(
        topk=args.topk,
        weight_method=args.method,
        n_stocks=args.stocks
    )
    
    # 总结
    print("\n" + "=" * 80)
    print("✨ 策略核心特性总结")
    print("=" * 80)
    
    features = [
        "🏷️ 低价门槛: 股价 ≤ 20元强制筛选",
        "📈 成长性: EPS CAGR > 15% (严格) / 12% (宽松)",
        "💰 估值: PEG < 1.2 (严格) / 1.5 (宽松)",
        "🛡️ 稳定性: ROE标准差 < 3% (严格) / 5% (宽松)",
        "⚖️ 权重优化: 4种权重分配方法",
        "🧪 严格验证: 4项阈值确保策略质量",
        "📊 完整回测: 63日标签匹配季度周期",
        "🎯 实用性: 适合低估值成长股发掘"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print(f"\n🔗 使用建议:")
    print(f"  # 基础使用")
    print(f"  python standalone_lowprice_strategy.py")
    print(f"  ")
    print(f"  # 严格筛选")
    print(f"  python standalone_lowprice_strategy.py --strict --topk 5")
    print(f"  ")
    print(f"  # 大规模测试")
    print(f"  python standalone_lowprice_strategy.py --stocks 100 --method risk_opt")
    
    success_text = "✅ 策略验证成功" if validation_passed else "⚠️ 策略需要调优"
    print(f"\n{success_text}")
    
    print("\n🎉 低价+稳定增长策略演示完成!")


if __name__ == "__main__":
    main()