#!/usr/bin/env python3
"""
低价+稳定增长选股器 - 内存友好版本
=============================================

基于LowPriceGrowthSelector的内存优化版本，支持：
- 滚动窗口训练（每次只加载2-3年数据）
- Parquet分区存储基本面数据
- 内存限制和监控
- 支持完整11年数据训练，内存控制在4-8GB

Usage:
    # 滚动训练（推荐）
    python lowprice_growth_selector_light.py --rolling_train --window_years 3 --max_ram_gb 4
    
    # 推理最新窗口
    python lowprice_growth_selector_light.py --predict_latest
    
    # 完整回测
    python lowprice_growth_selector_light.py --backtest_stocks --topk 10
"""

import os
import sys
import argparse
import warnings
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import gc
import glob

# 添加父类路径
sys.path.append(str(Path(__file__).parent))
from lowprice_growth_selector import LowPriceGrowthSelector, CONFIG

# 新增依赖
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    print("✓ PyArrow导入成功")
except ImportError as e:
    print("❌ PyArrow导入失败，请安装: pip install pyarrow>=15.0")
    sys.exit(1)

warnings.filterwarnings('ignore')

# 内存友好配置
LIGHT_CONFIG = {
    **CONFIG,
    'chunk_size_years': 2,            # 分块加载年数
    'enable_chunked_loading': True,   # 启用分块加载
    'parquet_partition_cols': ['code'],  # Parquet分区列
    'max_ram_gb': 4,                  # 默认最大内存限制
    'dtype_optimization': 'float32',  # 数据类型优化
}

logger = logging.getLogger(__name__)


class LowPriceGrowthSelectorLight(LowPriceGrowthSelector):
    """内存友好的低价成长选股器"""
    
    def __init__(self, 
                 start_date: str = LIGHT_CONFIG['start_date'],
                 end_date: str = LIGHT_CONFIG['end_date'],
                 universe: str = LIGHT_CONFIG['universe'],
                 max_ram_gb: int = LIGHT_CONFIG['max_ram_gb']):
        """
        初始化轻量级选股器
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            universe: 股票池
            max_ram_gb: 最大内存限制(GB)
        """
        # 更新内存配置
        CONFIG['max_memory_gb'] = max_ram_gb
        LIGHT_CONFIG['max_ram_gb'] = max_ram_gb
        
        # 调用父类初始化
        super().__init__(start_date, end_date, universe)
        
        # 内存友好设置
        self.max_ram_gb = max_ram_gb
        self.fundamental_parquet_dir = self.output_dir / 'fundamental_parquet'
        self.fundamental_parquet_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 内存友好模式已启用，最大内存限制: {max_ram_gb}GB")
    
    def fetch_fundamental_data_parquet(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        获取基本面数据并保存为Parquet分区格式
        
        Args:
            force_refresh: 是否强制刷新数据
            
        Returns:
            基本面数据DataFrame
        """
        logger.info("🗃️ 开始获取基本面数据（Parquet格式）...")
        
        # 检查Parquet缓存
        if not force_refresh and len(list(self.fundamental_parquet_dir.glob('*'))) > 0:
            logger.info("加载缓存的Parquet基本面数据...")
            return self._load_fundamental_parquet()
        
        # 获取股票列表
        stock_codes = self._get_stock_list()
        
        # 分批处理股票并保存为Parquet
        failed_stocks = []
        
        for i, code in enumerate(tqdm(stock_codes, desc="获取基本面数据")):
            try:
                self._fetch_and_save_stock_fundamental(code)
                
                # 内存管理：每100只股票强制垃圾回收
                if i % 100 == 0 and i > 0:
                    self._log_memory_usage(f"已处理{i}只股票")
                    self._force_gc()
                    
            except Exception as e:
                logger.warning(f"股票 {code} 基本面数据获取失败: {e}")
                failed_stocks.append(code)
        
        logger.info(f"基本面数据获取完成，失败股票数: {len(failed_stocks)}")
        
        # 返回完整数据
        return self._load_fundamental_parquet()
    
    def _get_stock_list(self) -> List[str]:
        """获取股票列表"""
        if self.universe == 'all':
            # 扩展预定义股票池
            stock_codes = [
                'SH600000', 'SH600036', 'SH600519', 'SZ000001', 'SZ000002',
                'SZ300015', 'SZ002415', 'SH600276', 'SH600030', 'SZ000858',
                'SH600585', 'SH601318', 'SZ002594', 'SH600900', 'SZ002230',
                'SH601398', 'SH601939', 'SZ000002', 'SH600887', 'SZ002304'
            ]
            logger.info(f"使用扩展预定义股票池: {len(stock_codes)}只股票")
            return stock_codes
        else:
            # 使用父类方法
            try:
                from qlib.data import D
                instruments = D.list_instruments(
                    instruments=self.universe, 
                    start_time=self.start_date, 
                    end_time=self.end_date
                )
                if isinstance(instruments, (list, tuple)):
                    return [inst for inst in instruments if isinstance(inst, str)]
                else:
                    return ['SH600000', 'SH600036', 'SH600519', 'SZ000001', 'SZ000002']
            except:
                return ['SH600000', 'SH600036', 'SH600519', 'SZ000001', 'SZ000002']
    
    def _fetch_and_save_stock_fundamental(self, code: str):
        """获取单个股票的基本面数据并保存为Parquet"""
        try:
            import akshare as ak
            
            # 转换股票代码格式
            ak_code = code.replace('SH', '').replace('SZ', '')
            
            # 获取财务指标数据
            financial_data = ak.stock_financial_analysis_indicator_em(symbol=ak_code)
            if financial_data.empty:
                return
            
            # 数据清理
            financial_data['date'] = pd.to_datetime(financial_data['date'])
            financial_data['code'] = code
            
            # 打印列名以便调试
            logger.info(f"获取到的列名: {list(financial_data.columns)}")
            
            # 更灵活的字段匹配
            # 尝试匹配常见的列名变体
            column_mappings = {
                # 原始名称 -> 标准名称
                '净利润': 'net_profit',
                '净利润-单季度': 'net_profit',
                '净利润(万元)': 'net_profit',
                '净利润同比增长': 'net_profit_yoy',
                '净利润同比增长率': 'net_profit_yoy',
                '营业总收入': 'revenue', 
                '营业收入': 'revenue',
                '营业收入(万元)': 'revenue',
                '营业总收入同比增长': 'revenue_yoy',
                '营业收入同比增长率': 'revenue_yoy',
                'ROE': 'roe',
                '净资产收益率': 'roe',
                '加权净资产收益率': 'roe',
                '基本每股收益': 'eps_basic',
                '每股收益': 'eps_basic',
                '基本每股收益(元)': 'eps_basic',
                '摊薄每股收益': 'eps_basic'
            }
            
            # 筛选可用字段
            available_mappings = {}
            for orig_col in financial_data.columns:
                if orig_col in column_mappings:
                    available_mappings[orig_col] = column_mappings[orig_col]
            
            if len(available_mappings) == 0:
                logger.warning(f"股票 {code} 未找到匹配的财务指标列")
                return
            
            # 保留需要的列
            keep_cols = ['date', 'code'] + list(available_mappings.keys())
            financial_data = financial_data[keep_cols]
            
            # 重命名列
            for old_name, new_name in available_mappings.items():
                if old_name in financial_data.columns:
                    financial_data[new_name] = pd.to_numeric(
                        financial_data[old_name], errors='coerce'
                    )
            
            # 数据类型优化
            if LIGHT_CONFIG['dtype_optimization'] == 'float32':
                standard_cols = list(available_mappings.values())
                for col in standard_cols:
                    if col in financial_data.columns:
                        financial_data[col] = financial_data[col].astype('float32')
            
            # 过滤日期范围
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
            financial_data = financial_data[
                (financial_data['date'] >= start_dt) & 
                (financial_data['date'] <= end_dt)
            ]
            
            if financial_data.empty:
                return
            
            # 保存为Parquet分区
            stock_dir = self.fundamental_parquet_dir / f'code={code}'
            stock_dir.mkdir(exist_ok=True)
            
            # 写入Parquet文件
            parquet_file = stock_dir / f'part-{code}.parquet'
            financial_data.to_parquet(parquet_file, index=False, compression='snappy')
            
        except Exception as e:
            logger.warning(f"股票 {code} 基本面数据处理失败: {e}")
    
    def _load_fundamental_parquet(self, 
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载Parquet分区的基本面数据
        
        Args:
            start_date: 开始日期过滤
            end_date: 结束日期过滤
            
        Returns:
            基本面数据DataFrame
        """
        if not self.fundamental_parquet_dir.exists():
            logger.warning("Parquet基本面数据目录不存在")
            return pd.DataFrame()
        
        # 构建过滤条件
        filters = []
        if start_date:
            filters.append(('date', '>', pd.to_datetime(start_date)))
        if end_date:
            filters.append(('date', '<=', pd.to_datetime(end_date)))
        
        try:
            # 检查是否存在parquet文件
            parquet_files = list(self.fundamental_parquet_dir.glob('**/*.parquet'))
            if not parquet_files:
                logger.info("未找到Parquet文件，返回空DataFrame")
                return pd.DataFrame()
            
            # 读取所有分区数据
            dataset = pq.ParquetDataset(
                self.fundamental_parquet_dir,
                filters=filters if filters else None
            )
            
            table = dataset.read()
            df = table.to_pandas()
            
            logger.info(f"从Parquet加载基本面数据: {len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载Parquet基本面数据失败: {e}")
            return pd.DataFrame()
    
    def rolling_train(self, 
                     window_years: int = 3,
                     step_years: int = 1,
                     dtype: str = 'float32') -> Dict[str, str]:
        """
        滚动窗口训练
        
        Args:
            window_years: 训练窗口年数
            step_years: 步长年数
            dtype: 数据类型优化
            
        Returns:
            训练模型文件路径字典
        """
        logger.info(f"🔄 开始滚动训练，窗口: {window_years}年，步长: {step_years}年")
        self._log_memory_usage("滚动训练开始")
        
        # 生成所有滚动窗口
        windows = self._generate_rolling_windows(window_years, step_years)
        logger.info(f"生成 {len(windows)} 个训练窗口")
        
        model_paths = {}
        
        for i, (train_start, train_end) in enumerate(windows):
            logger.info(f"📈 训练窗口 {i+1}/{len(windows)}: {train_start} - {train_end}")
            
            try:
                # 临时调整日期范围
                original_start = self.start_date
                original_end = self.end_date
                
                self.start_date = train_start
                self.end_date = train_end
                
                # 准备窗口数据集
                self._log_memory_usage(f"准备窗口{i+1}数据集")
                dataset = self._prepare_dataset_optimized(dtype=dtype, drop_raw=True)
                
                # 训练模型
                self._log_memory_usage(f"训练窗口{i+1}模型")
                models = self.train_model(dataset)
                
                # 保存模型
                model_name = f"lowprice_{train_start[:4]}_{train_end[:4]}.pkl"
                model_path = self.model_dir / model_name
                
                with open(model_path, 'wb') as f:
                    pickle.dump(models['lgb'], f)
                
                model_paths[f"{train_start}_{train_end}"] = str(model_path)
                logger.info(f"💾 模型已保存: {model_path}")
                
                # 强制清理内存
                del dataset, models
                self._force_gc()
                self._log_memory_usage(f"窗口{i+1}训练完成")
                
                # 恢复原始日期范围
                self.start_date = original_start
                self.end_date = original_end
                
            except Exception as e:
                logger.error(f"窗口 {train_start}-{train_end} 训练失败: {e}")
                continue
        
        logger.info(f"✅ 滚动训练完成，共训练 {len(model_paths)} 个模型")
        
        # 保存模型索引
        model_index_path = self.model_dir / 'rolling_models_index.json'
        with open(model_index_path, 'w') as f:
            json.dump(model_paths, f, indent=2)
        
        return model_paths
    
    def _generate_rolling_windows(self, window_years: int, step_years: int) -> List[Tuple[str, str]]:
        """生成滚动窗口列表"""
        windows = []
        
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        current_start = start_dt
        
        while True:
            # 计算窗口结束时间
            window_end = current_start + timedelta(days=window_years * 365)
            
            # 如果窗口结束时间超过总结束时间，调整为总结束时间
            if window_end > end_dt:
                window_end = end_dt
            
            # 添加窗口
            windows.append((
                current_start.strftime('%Y-%m-%d'),
                window_end.strftime('%Y-%m-%d')
            ))
            
            # 如果已经到达总结束时间，停止
            if window_end >= end_dt:
                break
            
            # 移动到下一个窗口
            current_start = current_start + timedelta(days=step_years * 365)
        
        return windows
    
    def _prepare_dataset_optimized(self, dtype: str = 'float32', drop_raw: bool = True):
        """
        内存优化的数据集准备
        
        Args:
            dtype: 数据类型
            drop_raw: 是否删除原始数据以节省内存
            
        Returns:
            优化的数据集
        """
        logger.info("📊 准备优化数据集...")
        self._log_memory_usage("开始数据集优化")
        
        # 获取基本面数据（时间范围过滤）
        fundamental_data = self._load_fundamental_parquet(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if not fundamental_data.empty:
            fundamental_with_metrics = self._calculate_rolling_metrics(fundamental_data)
            if drop_raw:
                del fundamental_data
                gc.collect()
        else:
            fundamental_with_metrics = pd.DataFrame()
            logger.warning("基本面数据为空，跳过基本面特征计算")
        
        # 精简技术指标特征（内存优化）
        essential_features = [
            # 核心价格动量（必需）
            "Rsi()",
            "($close-Ref($close,20))/Ref($close,20)",
            "($close-Ref($close,5))/Ref($close,5)",
            
            # 核心估值（必需）
            "PeTtm()",
            "PbMrq()",
            "RoeTtm()",
            
            # 低价标记（核心）
            "If($close <= 20, 1, 0)",
            
            # 成长性指标
            "($close / Ref($close, 63) - 1)",
        ]
        
        # 63日收益标签
        label_expr = "Ref($close, -63) / Ref($close, -1) - 1"
        
        # 使用优化的handler配置
        try:
            from qlib.contrib.data.handler import Alpha158
            
            # 内存优化的预处理器
            optimized_processors = []
            if dtype == 'float32':
                optimized_processors.append({
                    "class": "Fillna", 
                    "kwargs": {"fields_group": "feature"}
                })
            else:
                optimized_processors.extend([
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ])
            
            handler = Alpha158(
                instruments=self.universe,
                start_time=self.start_date,
                end_time=self.end_date,
                fit_start_time=self.start_date,
                fit_end_time=self.end_date,
                infer_processors=optimized_processors,
                learn_processors=[
                    {"class": "DropnaLabel"},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
            )
            
            # 分段配置
            mid_date = (datetime.strptime(self.start_date, '%Y-%m-%d') + 
                       timedelta(days=(datetime.strptime(self.end_date, '%Y-%m-%d') - 
                                     datetime.strptime(self.start_date, '%Y-%m-%d')).days * 0.8)).strftime('%Y-%m-%d')
            
            segments = {
                "train": (self.start_date, mid_date),
                "valid": (mid_date, self.end_date),
                "test": (mid_date, self.end_date),
            }
            
            # 创建数据集
            self._log_memory_usage("开始创建优化DatasetH")
            from qlib.data.dataset import DatasetH
            dataset = DatasetH(handler, segments)
            self._log_memory_usage("优化DatasetH创建完成")
            
            return dataset
            
        except Exception as e:
            logger.error(f"优化数据集创建失败: {e}")
            # 回退到父类方法
            return super()._prepare_dataset()
    
    def predict_latest_window(self) -> Dict[str, pd.DataFrame]:
        """
        使用最新训练的模型进行推理
        
        Returns:
            预测结果字典
        """
        logger.info("🔮 使用最新窗口模型进行推理...")
        
        # 查找最新的模型
        model_files = list(self.model_dir.glob('lowprice_*.pkl'))
        if not model_files:
            raise FileNotFoundError("未找到滚动训练的模型文件，请先运行 --rolling_train")
        
        # 找到最新的模型（按文件名中的年份排序）
        latest_model = max(model_files, key=lambda x: x.stem.split('_')[-1])
        logger.info(f"使用最新模型: {latest_model}")
        
        # 加载模型
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        
        models = {'lgb': model}
        
        # 准备最新数据进行推理
        self._log_memory_usage("准备推理数据")
        
        # 使用最近2年数据进行推理
        inference_start = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        inference_end = self.end_date
        
        original_start = self.start_date
        self.start_date = inference_start
        
        try:
            dataset = self._prepare_dataset_optimized(drop_raw=True)
            
            # 执行推理
            self._log_memory_usage("开始模型推理")
            pred = model.predict(dataset)
            pred_df = pred.reset_index()
            pred_df.columns = ['date', 'instrument', 'prediction']
            
            # 保存预测结果
            output_path = self.output_dir / 'signal_latest_window.csv'
            pred_df.to_csv(output_path, index=False)
            logger.info(f"💾 最新推理结果保存至: {output_path}")
            
            self._log_memory_usage("推理完成")
            
            return {'latest': pred_df}
            
        finally:
            self.start_date = original_start
    
    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'current_mb': memory_info.rss / 1024 / 1024,
                'peak_mb': memory_info.peak_wss / 1024 / 1024 if hasattr(memory_info, 'peak_wss') else 0,
                'percent': process.memory_percent(),
                'limit_gb': self.max_ram_gb,
                'status': 'OK' if memory_info.rss / 1024 / 1024 / 1024 < self.max_ram_gb else 'WARNING'
            }
        except:
            return {'status': 'UNKNOWN'}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='低价+稳定增长选股器 - 内存友好版本')
    
    # 主要操作
    parser.add_argument('--rolling_train', action='store_true', help='滚动窗口训练')
    parser.add_argument('--predict_latest', action='store_true', help='使用最新模型推理')
    parser.add_argument('--backtest_stocks', action='store_true', help='回测')
    
    # 滚动训练参数
    parser.add_argument('--window_years', type=int, default=3, help='训练窗口年数')
    parser.add_argument('--step_years', type=int, default=1, help='步长年数')
    parser.add_argument('--dtype', type=str, default='float32', 
                       choices=['float32', 'float64'], help='数据类型优化')
    
    # 内存管理
    parser.add_argument('--max_ram_gb', type=int, default=4, help='最大内存限制(GB)')
    
    # 其他参数
    parser.add_argument('--topk', type=int, default=10, help='选择股票数量')
    parser.add_argument('--start_date', type=str, default=LIGHT_CONFIG['start_date'], help='开始日期')
    parser.add_argument('--end_date', type=str, default=LIGHT_CONFIG['end_date'], help='结束日期')
    parser.add_argument('--universe', type=str, default=LIGHT_CONFIG['universe'], help='股票池')
    
    args = parser.parse_args()
    
    print("✓ 内存友好版本启动")
    
    # 创建轻量级选股器
    selector = LowPriceGrowthSelectorLight(
        start_date=args.start_date,
        end_date=args.end_date,
        universe=args.universe,
        max_ram_gb=args.max_ram_gb
    )
    
    try:
        if args.rolling_train:
            # 首先获取基本面数据
            selector.fetch_fundamental_data_parquet()
            
            # 执行滚动训练
            model_paths = selector.rolling_train(
                window_years=args.window_years,
                step_years=args.step_years,
                dtype=args.dtype
            )
            
            print(f"\n=== 滚动训练完成 ===")
            print(f"训练模型数量: {len(model_paths)}")
            for window, path in model_paths.items():
                print(f"  {window}: {path}")
        
        if args.predict_latest:
            predictions = selector.predict_latest_window()
            latest_pred = predictions['latest']
            print(f"\n=== 最新推理完成 ===")
            print(f"预测样本数: {len(latest_pred)}")
            print(f"时间范围: {latest_pred['date'].min()} - {latest_pred['date'].max()}")
        
        if args.backtest_stocks:
            # 使用父类的回测方法
            equity_curve, metrics = selector.backtest(topk=args.topk)
            
            print(f"\n=== 回测结果 ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        # 显示内存摘要
        memory_summary = selector.get_memory_summary()
        print(f"\n=== 内存使用摘要 ===")
        print(f"当前内存: {memory_summary.get('current_mb', 0):.1f} MB")
        print(f"内存占用: {memory_summary.get('percent', 0):.1f}%")
        print(f"内存状态: {memory_summary.get('status', 'UNKNOWN')}")
        
    except Exception as e:
        logger.error(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()