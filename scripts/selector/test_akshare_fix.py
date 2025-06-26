#!/usr/bin/env python3
"""
测试AkShare函数修复
"""

def test_akshare_function():
    """测试AkShare函数是否正常工作"""
    try:
        import akshare as ak
        print("✅ AkShare导入成功")
        
        # 测试股票代码 - 茅台
        test_code = '600519'
        print(f"测试股票代码: {test_code}")
        
        # 测试正确的函数名
        print("尝试调用: ak.stock_financial_analysis_indicator_em(symbol='600519')")
        data = ak.stock_financial_analysis_indicator_em(symbol=test_code)
        
        if data.empty:
            print("❌ 返回数据为空")
            return False
        else:
            print(f"✅ 成功获取数据，行数: {len(data)}")
            print(f"✅ 列名: {list(data.columns)}")
            print(f"✅ 数据类型: {data.dtypes}")
            print(f"✅ 前3行数据:")
            print(data.head(3))
            return True
            
    except ImportError as e:
        print(f"❌ AkShare导入失败: {e}")
        return False
    except AttributeError as e:
        print(f"❌ 函数不存在: {e}")
        print("可用的AkShare函数:")
        try:
            import akshare as ak
            # 查找相关函数
            ak_functions = [attr for attr in dir(ak) if 'financial' in attr.lower() or 'indicator' in attr.lower()]
            for func in ak_functions[:10]:  # 只显示前10个
                print(f"  - {func}")
        except:
            pass
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_alternative_functions():
    """测试替代函数"""
    try:
        import akshare as ak
        
        # 测试其他可能的函数
        alternative_functions = [
            'stock_financial_abstract',
            'stock_financial_analysis_indicator_em',
            'stock_individual_info_em',
            'stock_zh_a_hist',
        ]
        
        for func_name in alternative_functions:
            if hasattr(ak, func_name):
                print(f"✅ 找到函数: {func_name}")
                try:
                    if func_name == 'stock_zh_a_hist':
                        data = getattr(ak, func_name)(symbol="600519", period="daily", start_date="2024-01-01", end_date="2024-12-31")
                    elif func_name == 'stock_individual_info_em':
                        data = getattr(ak, func_name)(symbol="600519")
                    else:
                        data = getattr(ak, func_name)(symbol="600519")
                    print(f"  - 数据形状: {data.shape}")
                    print(f"  - 列名: {list(data.columns)[:5]}...")  # 前5列
                except Exception as e:
                    print(f"  - 调用失败: {e}")
            else:
                print(f"❌ 未找到函数: {func_name}")
                
    except Exception as e:
        print(f"❌ 测试替代函数失败: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 AkShare函数修复测试")
    print("=" * 50)
    
    success = test_akshare_function()
    
    if not success:
        print("\n" + "=" * 50)
        print("🔍 测试替代函数")
        print("=" * 50)
        test_alternative_functions()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成" if success else "❌ 需要进一步调试")
    print("=" * 50)