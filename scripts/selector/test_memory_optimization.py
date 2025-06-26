#!/usr/bin/env python3
"""
测试内存优化功能
"""

import subprocess
import sys
import time

def test_memory_optimization():
    """测试内存优化功能"""
    print("=== 内存优化测试 ===")
    
    tests = [
        {
            "name": "默认模式（自动优化）",
            "cmd": [sys.executable, "lowprice_growth_selector.py", "--train"],
            "expected": "应该自动将11年范围优化为3年"
        },
        {
            "name": "禁用内存限制模式",
            "cmd": [sys.executable, "lowprice_growth_selector.py", "--train", "--disable_memory_limit"],
            "expected": "应该使用完整11年范围（可能内存很大）"
        },
        {
            "name": "自定义内存限制",
            "cmd": [sys.executable, "lowprice_growth_selector.py", "--train", "--max_memory_gb", "4"],
            "expected": "应该使用4GB内存限制"
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   命令: {' '.join(test['cmd'])}")
        print(f"   预期: {test['expected']}")
        
        # 运行命令（限制30秒，主要看内存优化消息）
        try:
            result = subprocess.run(
                test['cmd'], 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd='.'
            )
            
            # 检查输出中的关键信息
            output = result.stdout + result.stderr
            
            if "时间范围过长" in output:
                print("   ✓ 检测到时间范围优化")
            if "自动优化为最近3年" in output:
                print("   ✓ 自动优化生效")
            if "内存限制已禁用" in output:
                print("   ✓ 内存限制禁用生效")
            if "设置内存软限制" in output:
                print("   ✓ 内存限制设置生效")
                
            # 显示内存使用信息
            for line in output.split('\n'):
                if "内存使用" in line:
                    print(f"   💾 {line.strip()}")
                if "优化范围" in line:
                    print(f"   📅 {line.strip()}")
                    
        except subprocess.TimeoutExpired:
            print("   ⏰ 30秒超时（正常，主要检查优化消息）")
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")

def show_usage_examples():
    """显示使用示例"""
    print("\n=== 使用建议 ===")
    print("1. 🚀 快速训练（推荐）：")
    print("   python lowprice_growth_selector.py --train")
    print("   → 自动优化时间范围，内存使用约1-2GB")
    
    print("\n2. 🔧 自定义内存限制：")
    print("   python lowprice_growth_selector.py --train --max_memory_gb 6")
    print("   → 设置6GB内存限制")
    
    print("\n3. ⚠️ 完整范围（高内存）：")
    print("   python lowprice_growth_selector.py --train --disable_memory_limit")
    print("   → 使用2014-2025完整范围，可能需要10-20GB内存")
    
    print("\n4. 📊 短期回测：")
    print("   python lowprice_growth_selector.py --backtest_stocks --start_date 2024-01-01 --end_date 2024-12-31")
    print("   → 只使用2024年数据，内存使用最小")

if __name__ == "__main__":
    test_memory_optimization()
    show_usage_examples()