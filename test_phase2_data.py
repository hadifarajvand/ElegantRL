#!/usr/bin/env python3
"""
Test Phase 2: Enhanced Data Management Implementations
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_realtime_data():
    """Test real-time data integration"""
    print("📡 Testing Real-time Data Integration...")
    
    try:
        from paper_trading.data.realtime_data import (
            RealTimeDataProvider, WebSocketManager, DataAggregator,
            create_realtime_provider, create_websocket_manager, create_data_aggregator
        )
        
        # Test real-time data provider
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        provider = create_realtime_provider(symbols, ['yahoo', 'alpaca'])
        print("✅ RealTimeDataProvider created successfully")
        
        # Test callback functionality
        data_received = []
        def test_callback(symbol, data):
            data_received.append((symbol, data))
        
        provider.add_callback(test_callback)
        print("✅ Callback added successfully")
        
        # Test data generation
        test_data = provider._generate_mock_realtime_data('AAPL')
        print(f"✅ Mock data generated: {test_data}")
        
        # Test data aggregator
        aggregator = create_data_aggregator(symbols)
        print("✅ DataAggregator created successfully")
        
        # Test data aggregation
        data_sources = {
            'yahoo': {'price': 150.0, 'volume': 1000, 'timestamp': datetime.now()},
            'alpaca': {'price': 150.1, 'volume': 1100, 'timestamp': datetime.now()}
        }
        
        aggregated = aggregator.aggregate_data('AAPL', data_sources)
        print(f"✅ Data aggregated: {aggregated}")
        
        # Test WebSocket manager
        ws_manager = create_websocket_manager(max_connections=5)
        print("✅ WebSocketManager created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time data test failed: {e}")
        return False

def test_advanced_indicators():
    """Test advanced technical indicators"""
    print("\n📊 Testing Advanced Technical Indicators...")
    
    try:
        from paper_trading.data.advanced_indicators import AdvancedTechnicalIndicators, create_advanced_indicators
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.randn(100) * 5 + 150,
            'high': np.random.randn(100) * 5 + 155,
            'low': np.random.randn(100) * 5 + 145,
            'close': np.random.randn(100) * 5 + 150,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Create indicators calculator
        indicators = create_advanced_indicators()
        print("✅ AdvancedTechnicalIndicators created successfully")
        
        # Calculate all indicators
        result_df = indicators.calculate_all_indicators(sample_data)
        print(f"✅ Calculated {len(result_df.columns)} total columns")
        
        # Check specific indicators
        expected_indicators = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_upper_20',
            'atr', 'adx', 'obv', 'mfi', 'vwap'
        ]
        
        for indicator in expected_indicators:
            if indicator in result_df.columns:
                print(f"✅ {indicator} calculated")
            else:
                print(f"⚠️ {indicator} not found")
        
        # Test individual indicator calculations
        rsi = indicators._calculate_rsi(sample_data['close'], 14)
        print(f"✅ RSI calculated - Range: {rsi.min():.2f} to {rsi.max():.2f}")
        
        macd, signal, hist = indicators._calculate_macd(sample_data['close'])
        print(f"✅ MACD calculated - MACD range: {macd.min():.2f} to {macd.max():.2f}")
        
        bb_upper, bb_middle, bb_lower = indicators._calculate_bollinger_bands(sample_data['close'], 20)
        print(f"✅ Bollinger Bands calculated")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced indicators test failed: {e}")
        return False

def test_data_integration():
    """Test data integration workflow"""
    print("\n🔄 Testing Data Integration Workflow...")
    
    try:
        from paper_trading.data.realtime_data import RealTimeDataProvider, DataAggregator
        from paper_trading.data.advanced_indicators import AdvancedTechnicalIndicators
        from paper_trading.data.market_data import YahooFinanceProvider
        
        # Create components
        symbols = ['AAPL', 'GOOGL']
        provider = RealTimeDataProvider(symbols)
        aggregator = DataAggregator(symbols)
        indicators = AdvancedTechnicalIndicators()
        
        print("✅ All data components created")
        
        # Test data flow
        # 1. Generate real-time data
        realtime_data = provider._generate_mock_realtime_data('AAPL')
        print(f"✅ Real-time data generated: {realtime_data['price']:.2f}")
        
        # 2. Aggregate data
        data_sources = {
            'yahoo': realtime_data,
            'alpaca': provider._generate_mock_realtime_data('AAPL', 'alpaca')
        }
        aggregated = aggregator.aggregate_data('AAPL', data_sources)
        print(f"✅ Data aggregated: {aggregated['price']:.2f}")
        
        # 3. Create historical data for indicators
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        hist_data = pd.DataFrame({
            'open': np.random.randn(50) * 5 + 150,
            'high': np.random.randn(50) * 5 + 155,
            'low': np.random.randn(50) * 5 + 145,
            'close': np.random.randn(50) * 5 + 150,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # 4. Calculate indicators
        indicators_df = indicators.calculate_all_indicators(hist_data)
        print(f"✅ Indicators calculated: {len(indicators_df.columns)} features")
        
        # 5. Test data quality
        quality_score = aggregator._calculate_quality_score(data_sources)
        print(f"✅ Data quality score: {quality_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data integration test failed: {e}")
        return False

def test_performance_metrics():
    """Test performance and efficiency"""
    print("\n⚡ Testing Performance Metrics...")
    
    try:
        from paper_trading.data.advanced_indicators import AdvancedTechnicalIndicators
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        large_data = pd.DataFrame({
            'open': np.random.randn(1000) * 5 + 150,
            'high': np.random.randn(1000) * 5 + 155,
            'low': np.random.randn(1000) * 5 + 145,
            'close': np.random.randn(1000) * 5 + 150,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        indicators = AdvancedTechnicalIndicators()
        
        # Time the calculation
        start_time = time.time()
        result_df = indicators.calculate_all_indicators(large_data)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        print(f"✅ Calculated {len(result_df.columns)} indicators in {calculation_time:.2f} seconds")
        
        # Check memory usage
        memory_usage = result_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        print(f"✅ Memory usage: {memory_usage:.2f} MB")
        
        # Performance thresholds
        if calculation_time < 5.0:  # Should complete within 5 seconds
            print("✅ Performance test passed")
        else:
            print("⚠️ Performance test slow")
        
        if memory_usage < 100:  # Should use less than 100MB
            print("✅ Memory usage test passed")
        else:
            print("⚠️ Memory usage high")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Phase 2: Enhanced Data Management Implementations")
    print("=" * 70)
    
    tests = [
        ("Real-time Data Integration", test_realtime_data),
        ("Advanced Technical Indicators", test_advanced_indicators),
        ("Data Integration Workflow", test_data_integration),
        ("Performance Metrics", test_performance_metrics)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PHASE 2 TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Phase 2 implementation completed successfully!")
    else:
        print("⚠️ Some Phase 2 tests failed. Please check the errors above.") 