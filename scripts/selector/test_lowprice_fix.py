#!/usr/bin/env python3
"""
测试低价选股器的修复
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

# 简单测试数据生成
def test_column_fix():
    """测试列名修复是否正确"""
    print("测试开始...")
    
    # 模拟D.features返回的数据格式
    mock_price_data = pd.DataFrame({
        'instrument': ['SH600000', 'SH600036', 'SZ000001'],
        '$close': [10.5, 15.2, 8.9]
    })
    
    print("原始数据列:", mock_price_data.columns.tolist())
    
    # 应用修复逻辑
    if '$close' in mock_price_data.columns:
        mock_price_data.rename(columns={'$close': 'close'}, inplace=True)
        print("修复后列:", mock_price_data.columns.tolist())
    
    # 测试访问
    try:
        result = mock_price_data[['instrument', 'close']]
        print("✓ 列访问成功")
        print(result)
    except KeyError as e:
        print(f"✗ 列访问失败: {e}")
    
    print("测试完成")

if __name__ == "__main__":
    test_column_fix()