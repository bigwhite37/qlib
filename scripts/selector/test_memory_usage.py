#!/usr/bin/env python3
"""
测试内存使用情况
"""

import sys
import psutil
import time
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process()
    print(f"开始内存监控...")
    print(f"初始内存: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    try:
        # 导入lowprice_growth_selector
        print("导入模块...")
        from lowprice_growth_selector import LowPriceGrowthSelector
        memory_after_import = process.memory_info().rss / 1024 / 1024
        print(f"导入后内存: {memory_after_import:.1f} MB")
        
        # 创建选股器
        print("创建选股器...")
        selector = LowPriceGrowthSelector(
            start_date='2023-01-01',  # 缩短时间范围测试
            end_date='2024-01-01',
            universe='all'
        )
        memory_after_init = process.memory_info().rss / 1024 / 1024
        print(f"初始化后内存: {memory_after_init:.1f} MB")
        
        # 开始数据准备
        print("准备数据集（这是高内存消耗环节）...")
        start_time = time.time()
        
        try:
            dataset = selector._prepare_dataset()
            memory_after_dataset = process.memory_info().rss / 1024 / 1024
            elapsed_time = time.time() - start_time
            print(f"数据集准备完成，内存: {memory_after_dataset:.1f} MB, 耗时: {elapsed_time:.1f}s")
            
            # 检查是否超过20GB
            if memory_after_dataset > 20 * 1024:
                print("⚠️ 内存使用超过20GB，存在内存问题！")
                return False
            else:
                print("✓ 内存使用正常")
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
    success = monitor_memory()
    if success:
        print("✓ 内存测试通过")
    else:
        print("✗ 内存测试失败，需要优化")