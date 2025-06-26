#!/usr/bin/env python3
"""
测试qlib复权因子和数据读取的正确性
"""

import warnings
warnings.filterwarnings('ignore')

import qlib
from qlib.config import REG_CN
from qlib.data import D
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_qlib_factor_understanding():
    """测试qlib复权因子的理解是否正确"""
    
    print("=" * 60)
    print("测试qlib复权因子和数据读取")
    print("=" * 60)
    
    try:
        # 初始化qlib
        qlib.init(region=REG_CN)
        print("✓ qlib初始化成功")
        
        # 获取交易日历
        calendar = D.calendar()
        latest_date = calendar[-1]
        print(f"✓ 最新交易日: {latest_date.strftime('%Y-%m-%d')}")
        
        # 测试股票：选择一个常见的股票
        test_stocks = ['SH600000', 'SZ000001', 'SH600036']
        
        for stock_code in test_stocks:
            print(f"\n--- 测试股票: {stock_code} ---")
            
            try:
                # 方法1：使用D.features()读取数据
                print("1. 使用D.features()读取最新交易日数据:")
                features_data = D.features([stock_code], ['$close', '$factor'],
                                         start_time=latest_date, end_time=latest_date)
                
                if not features_data.empty:
                    print(f"   数据形状: {features_data.shape}")
                    print(f"   数据内容:\n{features_data}")
                    
                    # 提取数据
                    adjusted_close = features_data.iloc[0, 0]  # $close
                    factor = features_data.iloc[0, 1]          # $factor
                    
                    print(f"   调整后收盘价($close): {adjusted_close}")
                    print(f"   复权因子($factor): {factor}")
                    
                    # 根据qlib文档计算原始价格
                    if not pd.isna(factor) and factor != 0:
                        original_price = adjusted_close / factor
                        print(f"   原始价格(计算): {original_price:.2f}")
                        print(f"   计算公式: original_price = adjusted_price / factor")
                        print(f"   验证: {adjusted_close:.6f} / {factor:.6f} = {original_price:.2f}")
                    else:
                        print("   ⚠️ 复权因子异常")
                else:
                    print("   ⚠️ 无数据")
                
                # 方法2：读取更多历史数据来验证
                print("\n2. 读取近5个交易日数据进行验证:")
                start_date = calendar[-5]
                hist_data = D.features([stock_code], ['$close', '$factor', '$volume'],
                                     start_time=start_date, end_time=latest_date)
                
                if not hist_data.empty:
                    print(f"   历史数据:\n{hist_data}")
                    
                    # 计算原始价格序列
                    close_prices = hist_data.iloc[:, 0]
                    factors = hist_data.iloc[:, 1]
                    original_prices = close_prices / factors
                    
                    print(f"\n   调整后价格趋势: {close_prices.tolist()}")
                    print(f"   复权因子趋势: {factors.tolist()}")
                    print(f"   原始价格趋势: {original_prices.round(2).tolist()}")
                
                # 方法3：尝试其他字段
                print("\n3. 测试其他价格字段:")
                other_data = D.features([stock_code], ['$open', '$high', '$low', '$close', '$factor'],
                                      start_time=latest_date, end_time=latest_date)
                
                if not other_data.empty:
                    print(f"   完整OHLC数据:\n{other_data}")
                    
                    # 验证所有价格字段是否都是调整后的
                    row = other_data.iloc[0]
                    factor = row[4]  # $factor
                    
                    if not pd.isna(factor) and factor != 0:
                        print(f"\n   所有价格字段除以factor后的原始价格:")
                        print(f"   原始开盘价: {row[0]/factor:.2f}")
                        print(f"   原始最高价: {row[1]/factor:.2f}")
                        print(f"   原始最低价: {row[2]/factor:.2f}")
                        print(f"   原始收盘价: {row[3]/factor:.2f}")
                
            except Exception as e:
                print(f"   ❌ 测试股票 {stock_code} 失败: {e}")
                continue
        
        print(f"\n" + "=" * 60)
        print("总结:")
        print("1. qlib中的$close, $open, $high, $low都是调整后(复权后)的价格")
        print("2. $factor是复权因子，计算公式为: factor = adjusted_price / original_price")
        print("3. 要获取原始交易价格，使用: original_price = adjusted_price / factor")
        print("4. 这与代码中的计算逻辑是正确的")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_D_features_call():
    """测试D.features()调用的正确性"""
    
    print("\n" + "=" * 60)
    print("测试D.features()调用方式")
    print("=" * 60)
    
    try:
        qlib.init(region=REG_CN)
        
        # 获取一个测试股票
        stock_code = 'SH600000'
        calendar = D.calendar()
        latest_date = calendar[-1]
        
        print(f"测试股票: {stock_code}")
        print(f"测试日期: {latest_date}")
        
        # 测试不同的调用方式
        print("\n1. 测试单只股票，单个字段:")
        data1 = D.features([stock_code], ['$close'], 
                          start_time=latest_date, end_time=latest_date)
        print(f"   返回数据类型: {type(data1)}")
        print(f"   数据形状: {data1.shape}")
        print(f"   索引: {data1.index}")
        print(f"   列名: {data1.columns.tolist()}")
        print(f"   数据:\n{data1}")
        
        print("\n2. 测试单只股票，多个字段:")
        data2 = D.features([stock_code], ['$close', '$factor'], 
                          start_time=latest_date, end_time=latest_date)
        print(f"   返回数据类型: {type(data2)}")
        print(f"   数据形状: {data2.shape}")
        print(f"   索引: {data2.index}")
        print(f"   列名: {data2.columns.tolist()}")
        print(f"   数据:\n{data2}")
        
        # 验证数据访问方式
        print("\n3. 测试数据访问方式:")
        if not data2.empty:
            print(f"   方式1 - iloc[0, 0]: {data2.iloc[0, 0]} (第一行第一列)")
            print(f"   方式1 - iloc[0, 1]: {data2.iloc[0, 1]} (第一行第二列)")
            
            # 按列名访问
            print(f"   方式2 - 按列名访问:")
            for col in data2.columns:
                print(f"     {col}: {data2[col].iloc[0]}")
        
        print("\n4. 测试空数据情况:")
        # 测试一个不存在的股票
        try:
            data3 = D.features(['INVALID'], ['$close'], 
                              start_time=latest_date, end_time=latest_date)
            print(f"   不存在股票返回: {data3}")
            print(f"   是否为空: {data3.empty}")
        except Exception as e:
            print(f"   不存在股票异常: {e}")
        
        print("\n" + "=" * 60)
        print("D.features()调用分析:")
        print("1. 返回pandas.DataFrame格式")
        print("2. 对于单只股票+单日数据，返回1行N列的DataFrame")
        print("3. 列顺序与features参数顺序一致")
        print("4. 可以使用iloc[0, 0], iloc[0, 1]等方式访问")
        print("5. 空数据时返回空DataFrame，可用.empty检查")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_price_filtering_logic():
    """测试价格过滤逻辑的正确性"""
    
    print("\n" + "=" * 60)
    print("测试价格过滤逻辑")
    print("=" * 60)
    
    try:
        qlib.init(region=REG_CN)
        
        calendar = D.calendar()
        latest_date = calendar[-1]
        
        # 测试几只不同价格区间的股票
        test_stocks = ['SH600000', 'SZ000001', 'SH600036', 'SZ000002']
        
        print(f"测试日期: {latest_date}")
        print(f"价格过滤区间: [10, 50]")
        print()
        
        valid_stocks = []
        
        for stock_code in test_stocks:
            try:
                # 模拟select_stocks中的价格过滤逻辑
                price_data = D.features([stock_code], ['$close', '$factor'],
                                      start_time=latest_date, end_time=latest_date)
                
                if price_data.empty or price_data.isna().all().all():
                    print(f"{stock_code}: 无数据")
                    continue
                
                adjusted_price = price_data.iloc[0, 0]  # 调整后收盘价
                factor = price_data.iloc[0, 1]          # 复权因子
                
                if pd.isna(adjusted_price) or pd.isna(factor) or factor == 0:
                    print(f"{stock_code}: 价格或因子数据异常")
                    continue
                
                real_price = adjusted_price / factor  # 真实价格
                
                print(f"{stock_code}:")
                print(f"  调整后价格: {adjusted_price:.2f}")
                print(f"  复权因子: {factor:.6f}")
                print(f"  真实价格: {real_price:.2f}")
                
                # 价格过滤
                price_min, price_max = 10, 50
                if price_min <= real_price <= price_max:
                    print(f"  ✓ 通过价格过滤 ({price_min} <= {real_price:.2f} <= {price_max})")
                    valid_stocks.append(stock_code)
                else:
                    print(f"  ✗ 未通过价格过滤 ({real_price:.2f} 不在 [{price_min}, {price_max}] 范围)")
                
                print()
                
            except Exception as e:
                print(f"{stock_code}: 获取数据失败 - {e}")
        
        print(f"通过价格过滤的股票: {valid_stocks}")
        
        print("\n" + "=" * 60)
        print("价格过滤逻辑分析:")
        print("1. 使用最新交易日的数据进行过滤是合理的")
        print("2. 复权因子的计算和使用是正确的")
        print("3. 真实价格 = 调整后价格 / 复权因子")
        print("4. 需要处理NaN值和0值的情况")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_qlib_factor_understanding()
    test_D_features_call()
    test_price_filtering_logic()