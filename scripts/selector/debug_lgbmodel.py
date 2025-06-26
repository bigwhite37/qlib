#!/usr/bin/env python3
"""
专门调试LGBModel的_prepare_data方法
"""

import os
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd

# 初始化Qlib
qlib.init(region=REG_CN)

def debug_lgbmodel_prepare_data():
    """调试LGBModel的数据准备过程"""
    
    print("=== 调试LGBModel数据准备问题 ===")
    
    # 使用与ai_stock_selector.py相同的配置
    features = [
        # 价格动量因子
        "Mean($close,5)/Ref($close,1)-1",
        "Mean($close,10)/Ref($close,1)-1", 
        "Mean($close,20)/Ref($close,1)-1",
        "($close-Mean($close,20))/Mean($close,20)",
        "($close-Mean($close,20))/(Std($close,20)*2)",
        "(Mean($close,12)-Mean($close,26))/Mean($close,26)",
        "Mean((Mean($close,12)-Mean($close,26))/Mean($close,26),9)",
        "Ref($close,5)/$close-1",
        "Ref($close,10)/$close-1",

        # 量价波动因子
        "($high-$low)/$low",
        "($high-$low)/$close",
        "Std($close/Ref($close,1),5)",
        "Std($close/Ref($close,1),10)",
        "Std($close/Ref($close,1),20)",
        "$volume/Mean($volume,20)",
        "Std($volume,10)/Mean($volume,10)",
        "Corr($close,$volume,10)",

        # 替代估值质量因子
        "Log($close)",
        "$close/Mean($close,60)",
        "Mean($amount,5)",
        "Log($amount+1)",
        "$amount/Mean($amount,20)",

        # 技术指标
        "($close-Mean($close,6))/(Std($close,6)+1e-12)",
        "($close-Mean($close,14))/(Std($close,14)+1e-12)",
        "($close-Min($low,14))/(Max($high,14)-Min($low,14)+1e-12)",
        "($close-Min($close,20))/(Max($close,20)-Min($close,20)+1e-12)",
    ]
    
    # 数据加载器配置
    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": features,
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            },
            "freq": "day",
        },
    }
    
    # 数据处理器
    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {
            "fields_group": "feature",
            "fit_start_time": "2014-01-01",
            "fit_end_time": "2023-12-31"
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
            "start_time": "2014-01-01",
            "end_time": "2025-06-04",
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
                "train": ("2014-01-01", "2023-12-31"),
                "valid": ("2023-12-31", "2023-12-31"),
                "test": ("2023-12-31", "2025-06-04"),
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
        print(f"训练数据列数: {len(train_data.columns)}")
        print(f"是否包含NaN: {train_data.isna().any().any()}")
        
        # 检查数据内容
        feature_cols = [col for col in train_data.columns if col[0] == 'feature']
        label_cols = [col for col in train_data.columns if col[0] == 'label']
        print(f"特征列数: {len(feature_cols)}")
        print(f"标签列数: {len(label_cols)}")
        
        if len(train_data) > 0:
            print("\n3. 创建LGBModel并测试_prepare_data...")
            
            # 创建LGBModel
            lgb_model = LGBModel(
                loss="mse",
                boosting_type="gbdt",
                objective="regression",
                num_leaves=32,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                num_boost_round=10,
                early_stopping_rounds=5,
                verbosity=1,
                seed=42,
                n_jobs=1,
                force_col_wise=True
            )
            
            print("✓ LGBModel创建成功")
            
            # 测试_prepare_data方法
            try:
                print("调用_prepare_data方法...")
                ds_l = lgb_model._prepare_data(dataset, None)  # None reweighter
                print(f"✓ _prepare_data成功，返回: {type(ds_l)}")
                
                # 如果返回的是字典，查看其内容
                if isinstance(ds_l, dict):
                    for key, value in ds_l.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: 形状 {value.shape}")
                        else:
                            print(f"  {key}: 类型 {type(value)}")
                
            except Exception as e:
                print(f"✗ _prepare_data失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 尝试手动调用prepare方法
                print("\n4. 手动调用数据集prepare方法...")
                try:
                    # 检查不同的数据准备方式
                    train_set = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                    print(f"DK_L训练数据形状: {train_set.shape}")
                    print(f"DK_L训练数据是否为空: {len(train_set) == 0}")
                    
                    if len(train_set) == 0:
                        print("DK_L数据为空！这是问题的根源。")
                        
                        # 尝试其他数据key
                        try:
                            train_set_default = dataset.prepare("train", col_set=["feature", "label"])
                            print(f"默认训练数据形状: {train_set_default.shape}")
                        except Exception as e2:
                            print(f"默认prepare也失败: {e2}")
                    else:
                        print("DK_L数据不为空，这很奇怪...")
                        
                except Exception as e2:
                    print(f"手动prepare失败: {e2}")
        else:
            print("✗ 训练数据为空")
                
    except Exception as e:
        print(f"总体失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lgbmodel_prepare_data()