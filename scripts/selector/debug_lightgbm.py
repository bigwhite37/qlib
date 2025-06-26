#!/usr/bin/env python3
"""
调试LightGBM训练问题
"""

import os
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
import pandas as pd

# 初始化Qlib
qlib.init(region=REG_CN)

def debug_lightgbm():
    """调试LightGBM训练过程"""
    
    print("=== 调试LightGBM训练问题 ===")
    
    # 使用简化的特征
    simple_features = [
        "Mean($close,5)/Ref($close,1)-1",
        "Mean($close,10)/Ref($close,1)-1", 
        "($close-Mean($close,20))/Mean($close,20)",
        "Std($close/Ref($close,1),5)",
        "$volume/Mean($volume,20)"
    ]
    
    # 数据加载器配置
    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": simple_features,
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            },
            "freq": "day",
        },
    }
    
    # 简化的数据处理器
    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {
            "fields_group": "feature",
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2022-12-31"
        }},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
    ]
    
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    ]
    
    # 数据处理器配置
    handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "instruments": "csi300",
            "start_time": "2020-01-01",
            "end_time": "2023-12-31",
            "data_loader": data_loader_config,
            "infer_processors": infer_processors,
            "learn_processors": learn_processors,
        }
    }
    
    # 数据集配置
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
        print("1. 创建数据集...")
        dataset = init_instance_by_config(dataset_config)
        print("✓ 数据集创建成功")
        
        print("\n2. 检查训练数据...")
        train_data = dataset.prepare("train", col_set=["feature", "label"])
        print(f"训练数据形状: {train_data.shape}")
        print(f"训练数据列: {train_data.columns.tolist()}")
        print(f"数据类型: {train_data.dtypes}")
        print(f"是否包含NaN: {train_data.isna().any().any()}")
        print(f"数据前5行:")
        print(train_data.head())
        
        print("\n3. 创建LightGBM模型...")
        lgb_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "boosting_type": "gbdt",
                "objective": "regression",
                "num_leaves": 32,  # 减少复杂度
                "learning_rate": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "num_boost_round": 10,  # 快速测试
                "early_stopping_rounds": 5,
                "verbosity": 1,  # 显示详细信息
                "seed": 42,
                "n_jobs": 1
            }
        }
        
        lgb_model = init_instance_by_config(lgb_config)
        print("✓ LightGBM模型创建成功")
        
        print("\n4. 开始训练...")
        try:
            lgb_model.fit(dataset)
            print("✓ LightGBM训练成功!")
        except Exception as e:
            print(f"✗ LightGBM训练失败: {e}")
            
            # 尝试直接使用训练数据
            print("\n5. 尝试直接训练...")
            import lightgbm as lgb
            
            # 分离特征和标签
            feature_data = train_data.iloc[:, :-1]  # 除最后一列外的所有列
            label_data = train_data.iloc[:, -1]     # 最后一列
            
            print(f"特征数据形状: {feature_data.shape}")
            print(f"标签数据形状: {label_data.shape}")
            print(f"特征数据是否有NaN: {feature_data.isna().any().any()}")
            print(f"标签数据是否有NaN: {label_data.isna().any()}")
            
            # 删除包含NaN的行
            valid_mask = ~(feature_data.isna().any(axis=1) | label_data.isna())
            clean_features = feature_data[valid_mask]
            clean_labels = label_data[valid_mask]
            
            print(f"清理后特征数据形状: {clean_features.shape}")
            print(f"清理后标签数据形状: {clean_labels.shape}")
            
            if len(clean_features) > 0:
                try:
                    train_dataset = lgb.Dataset(clean_features, label=clean_labels)
                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': 32,
                        'learning_rate': 0.1,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'verbose': 1
                    }
                    
                    model = lgb.train(params, train_dataset, num_boost_round=10)
                    print("✓ 直接LightGBM训练成功!")
                except Exception as e:
                    print(f"✗ 直接LightGBM训练也失败: {e}")
            else:
                print("✗ 清理后没有有效数据")
                
    except Exception as e:
        print(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lightgbm()