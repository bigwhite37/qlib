#!/usr/bin/env python3
"""
测试完整时间范围的内存使用
"""

import sys
import psutil
import time
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_full_range_memory():
    """测试完整时间范围的内存使用"""
    process = psutil.Process()
    print(f"开始完整范围内存测试...")
    print(f"初始内存: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    try:
        # 导入模块
        from lowprice_growth_selector import LowPriceGrowthSelector
        
        # 创建选股器（使用完整时间范围）
        print("创建选股器（2014-2025完整范围）...")
        selector = LowPriceGrowthSelector(
            start_date='2014-01-01',  # 完整时间范围
            end_date='2025-06-04',
            universe='all'
        )
        memory_after_init = process.memory_info().rss / 1024 / 1024
        print(f"初始化后内存: {memory_after_init:.1f} MB")
        
        # 测试数据准备（这里会触发内存限制和优化）
        print("开始数据准备...")
        start_time = time.time()
        
        try:
            # 这里会触发时间范围优化
            dataset = selector._prepare_dataset()
            memory_after_dataset = process.memory_info().rss / 1024 / 1024
            elapsed_time = time.time() - start_time
            print(f"数据集准备完成，内存: {memory_after_dataset:.1f} MB, 耗时: {elapsed_time:.1f}s")
            
            # 检查内存使用
            memory_gb = memory_after_dataset / 1024
            if memory_gb > 10:  # 超过10GB算高内存
                print(f"⚠️ 高内存使用: {memory_gb:.1f} GB")
                if memory_gb > 20:  # 超过20GB算内存泄露
                    print("❌ 可能存在内存泄露！")
                    return False
                else:
                    print("⚠️ 内存使用较高但在可接受范围")
                    return True
            else:
                print(f"✓ 内存使用正常: {memory_gb:.1f} GB")
                return True
                
        except Exception as e:
            print(f"数据集准备失败: {e}")
            memory_after_error = process.memory_info().rss / 1024 / 1024
            print(f"错误时内存: {memory_after_error:.1f} MB")
            return False
            
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_full_range_memory()
    if success:
        print("✓ 完整范围内存测试通过")
    else:
        print("✗ 完整范围内存测试失败")