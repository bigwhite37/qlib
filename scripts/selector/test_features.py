#!/usr/bin/env python3
"""
测试哪些特征在Qlib中可用
"""

import qlib
from qlib.config import REG_CN
from qlib.data import D

# 初始化Qlib
qlib.init(region=REG_CN)

def test_features():
    """测试特征可用性"""
    
    # 原始特征列表
    original_features = [
        # 价格动量因子
        "Mean($close,5)/Ref($close,1)-1",  # 5日动量
        "Mean($close,10)/Ref($close,1)-1", # 10日动量
        "Mean($close,20)/Ref($close,1)-1", # 20日动量
        "($close-Mean($close,20))/Mean($close,20)",
        "($close-Mean($close,20))/(Std($close,20)*2)",
        "(Mean($close,12)-Mean($close,26))/Mean($close,26)",  # DIF
        "Mean((Mean($close,12)-Mean($close,26))/Mean($close,26),9)",  # DEA
        "Ref($close,5)/$close-1",  # 5日反转
        "Ref($close,10)/$close-1", # 10日反转

        # 量价波动因子
        "($high-$low)/$low",
        "($high-$low)/$close",
        "Std($close/Ref($close,1),5)",   # 5日波动
        "Std($close/Ref($close,1),10)",  # 10日波动
        "Std($close/Ref($close,1),20)",  # 20日波动
        "$volume/Mean($volume,20)",      # 量比
        "Std($volume,10)/Mean($volume,10)", # 量能变异
        "Corr($close,$volume,10)",       # 量价相关性

        # 估值质量因子（可能有问题的字段）
        "$market_cap",    # 可能不存在
        "Log($market_cap)", # 可能不存在
        "Mean($turn,5)",   # 可能不存在
        "Mean($turn,20)",  # 可能不存在
        "$turn/Mean($turn,20)", # 可能不存在

        # 技术指标
        "($close-Mean($close,6))/(Std($close,6)+1e-12)",
        "($close-Mean($close,14))/(Std($close,14)+1e-12)",
        "($close-Min($low,14))/(Max($high,14)-Min($low,14)+1e-12)",
        "($close-Min($close,20))/(Max($close,20)-Min($close,20)+1e-12)",
    ]
    
    # 测试股票列表
    test_instruments = ['SH600000', 'SH600004']
    
    # 基础字段测试
    basic_fields = ['$close', '$open', '$high', '$low', '$volume', '$amount', '$market_cap', '$turn']
    
    print("=== 基础字段测试 ===")
    for field in basic_fields:
        try:
            data = D.features(test_instruments, [field], 
                            start_time='2023-01-01', end_time='2023-01-31')
            print(f"✓ {field}: 数据形状 {data.shape}")
        except Exception as e:
            print(f"✗ {field}: 失败 - {e}")
    
    print("\n=== 特征表达式测试 ===")
    valid_features = []
    invalid_features = []
    
    for i, feature in enumerate(original_features):
        try:
            data = D.features(test_instruments, [feature], 
                            start_time='2023-01-01', end_time='2023-01-31')
            print(f"✓ 特征 {i+1}: {feature[:50]}... - 数据形状 {data.shape}")
            valid_features.append(feature)
        except Exception as e:
            print(f"✗ 特征 {i+1}: {feature[:50]}... - 失败: {str(e)[:100]}")
            invalid_features.append(feature)
    
    print(f"\n=== 总结 ===")
    print(f"有效特征数量: {len(valid_features)}")
    print(f"无效特征数量: {len(invalid_features)}")
    
    print(f"\n=== 推荐的特征配置 ===")
    print("valid_features = [")
    for feature in valid_features:
        print(f'    "{feature}",')
    print("]")
    
    if invalid_features:
        print(f"\n=== 需要修正的特征 ===")
        for feature in invalid_features:
            print(f"- {feature}")

if __name__ == "__main__":
    test_features()