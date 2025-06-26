#!/usr/bin/env python3
"""
调试数据集问题
"""

import os
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd

# 初始化Qlib
qlib.init(region=REG_CN)

def debug_dataset():
    """调试数据集创建过程"""
    
    print("1. 检查基础数据可用性...")
    
    # 检查股票列表
    instruments = D.instruments(market='all')
    print(f"可用股票数量: {len(instruments)}")
    
    # 检查交易日历
    calendar = D.calendar(start_time='2020-01-01', end_time='2023-12-31')
    print(f"交易日数量: {len(calendar)}")
    
    # 选择一个简单的特征列表进行测试
    simple_features = [
        "$close",
        "$volume",
        "Mean($close, 5)",
        "Std($close, 5)"
    ]
    
    print("\n2. 测试简单特征...")
    try:
        # 选择前10只股票进行测试
        test_instruments = instruments[:10] if len(instruments) > 10 else instruments
        print(f"测试股票: {test_instruments}")
        
        # 测试获取特征数据
        for feature in simple_features:
            try:
                data = D.features(test_instruments, [feature], 
                                start_time='2023-01-01', end_time='2023-01-31')
                print(f"特征 {feature}: 数据形状 {data.shape if data is not None else 'None'}")
            except Exception as e:
                print(f"特征 {feature} 失败: {e}")
                
    except Exception as e:
        print(f"特征测试失败: {e}")
    
    print("\n3. 测试数据集创建...")
    
    # 创建简化的数据集配置
    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": simple_features,  # 使用简单特征
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            },
            "freq": "day",
        },
    }
    
    handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "instruments": "csi300",  # 使用csi300而不是all
            "start_time": "2020-01-01",
            "end_time": "2023-12-31",
            "data_loader": data_loader_config,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {
                    "fields_group": "feature",
                    "fit_start_time": "2020-01-01",
                    "fit_end_time": "2022-12-31"
                }},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
            ],
        }
    }
    
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": ("2020-01-01", "2022-12-31"),
                "valid": ("2022-12-31", "2023-06-30"),
                "test": ("2023-06-30", "2023-12-31"),
            },
        }
    }
    
    try:
        print("创建数据集...")
        dataset = init_instance_by_config(dataset_config)
        print("数据集创建成功")
        
        # 检查训练数据
        print("\n4. 检查训练数据...")
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        print(f"训练数据形状: {train_data.shape}")
        print(f"训练数据列数: {len(train_data.columns)}")
        print(f"训练数据前5行:")
        print(train_data.head())
        
        # 检查是否有数据
        if len(train_data) == 0:
            print("⚠️ 训练数据为空!")
        else:
            print(f"✓ 训练数据包含 {len(train_data)} 行")
            
    except Exception as e:
        print(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()