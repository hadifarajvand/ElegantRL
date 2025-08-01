#!/usr/bin/env python3
"""
Comprehensive Test Script for Paper Trading System
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_configs():
    """Test configuration management"""
    print("\n🔧 Testing Configuration Management...")
    
    try:
        from paper_trading.configs.trading_config import TradingConfig, DEFAULT_US_MARKET_CONFIG
        from paper_trading.configs.model_config import ModelConfig, PPO_CONFIG
        from paper_trading.configs.data_config import DataConfig
        
        # Test trading config
        trading_config = TradingConfig()
        print(f"✅ Trading config created - Capital: ${trading_config.initial_capital:,.0f}")
        
        # Test model config
        model_config = PPO_CONFIG
        print(f"✅ Model config loaded - Agent: {model_config.agent_type}")
        
        # Test data config
        data_config = DataConfig()
        print(f"✅ Data config created - Source: {data_config.data_source}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_management():
    """Test data management components"""
    print("\n📊 Testing Data Management...")
    
    try:
        from paper_trading.data.market_data import YahooFinanceProvider
        from paper_trading.data.data_manager import DataManager
        from paper_trading.data.data_processor import DataProcessor
        
        # Test data provider
        provider = YahooFinanceProvider()
        print("✅ Data provider initialized")
        
        # Test data manager
        manager = DataManager(provider)
        print("✅ Data manager initialized")
        
        # Test data processor
        processor = DataProcessor()
        print("✅ Data processor initialized")
        
        # Test technical indicators calculation
        sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
        })
        
        indicators = processor.calculate_technical_indicators(sample_data)
        print(f"✅ Technical indicators calculated: {list(indicators.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Data management test failed: {e}")
        return False

def test_models():
    """Test model components"""
    print("\n🤖 Testing Models...")
    
    try:
        from paper_trading.models.trading_env import EnhancedStockTradingEnv
        from paper_trading.models.agents import RandomAgent, SimplePPOAgent, SimpleDQNAgent
        from paper_trading.models.portfolio import Portfolio
        
        # Test portfolio
        portfolio = Portfolio(initial_capital=100000)
        print("✅ Portfolio initialized")
        
        # Test agents
        state_dim, action_dim = 10, 3
        random_agent = RandomAgent(state_dim, action_dim)
        ppo_agent = SimplePPOAgent(state_dim, action_dim)
        dqn_agent = SimpleDQNAgent(state_dim, action_dim)
        print("✅ All agents initialized")
        
        # Test agent actions
        test_state = np.random.randn(state_dim)
        random_action = random_agent.get_action(test_state)
        print(f"✅ Agent actions generated - Random: {random_action.shape}")
        
        # Test trading environment (with mock data)
        mock_data = {
            'AAPL': pd.DataFrame({
                'close': np.random.randn(100) * 10 + 150,
                'volume': np.random.randint(1000, 10000, 100),
                'open': np.random.randn(100) * 5 + 150,
                'high': np.random.randn(100) * 5 + 155,
                'low': np.random.randn(100) * 5 + 145
            })
        }
        
        env = EnhancedStockTradingEnv(
            data=mock_data,
            initial_capital=100000,
            max_stock_quantity=100
        )
        print("✅ Trading environment created")
        
        # Test environment reset and step
        state, info = env.reset()
        print(f"✅ Environment reset - State shape: {state.shape}")
        
        action = np.random.randn(1) * 0.1  # Small random action for single stock
        next_state, reward, done, truncated, info = env.step(action)
        print(f"✅ Environment step - Reward: {reward:.4f}, Done: {done}")
        
        return True
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False

def test_trading_engine():
    """Test trading engine components"""
    print("\n⚙️ Testing Trading Engine...")
    
    try:
        from paper_trading.paper_trading_engine.risk_manager import RiskManager
        from paper_trading.paper_trading_engine.order_manager import OrderManager
        from paper_trading.paper_trading_engine.trading_engine import PaperTradingEngine
        from paper_trading.configs.trading_config import TradingConfig
        from paper_trading.configs.model_config import ModelConfig
        
        # Test risk manager
        risk_manager = RiskManager(
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            max_leverage=1.5,
            min_cash_reserve=0.1
        )
        print("✅ Risk manager initialized")
        
        # Test order manager
        order_manager = OrderManager(
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )
        print("✅ Order manager initialized")
        
        # Test risk validation
        portfolio_value = 100000
        positions = {'AAPL': {'quantity': 100, 'avg_price': 150, 'current_value': 15500, 'cost_basis': 15000, 'unrealized_pnl': 500}}
        
        validated_action = risk_manager.validate_action(
            action=np.array([0.1]),
            positions=positions,
            portfolio_value=portfolio_value
        )
        print(f"✅ Risk validation test - Action shape: {validated_action.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Trading engine test failed: {e}")
        return False

def test_backtesting():
    """Test backtesting components"""
    print("\n📈 Testing Backtesting...")
    
    try:
        from paper_trading.backtesting.backtest_engine import BacktestEngine
        from paper_trading.backtesting.performance_analyzer import PerformanceAnalyzer
        
        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        print("✅ Performance analyzer initialized")
        
        # Test metrics calculation
        portfolio_values = np.array([100000, 101000, 102500, 101800, 103200, 104500])
        
        analysis = analyzer.analyze_performance(portfolio_values)
        print(f"✅ Performance analysis completed: {list(analysis.keys())}")
        
        # Test specific metrics
        basic_metrics = analysis.get('basic_metrics', {})
        sharpe = basic_metrics.get('sharpe_ratio', 0)
        max_dd = basic_metrics.get('max_drawdown', 0)
        print(f"✅ Sharpe ratio: {sharpe:.4f}, Max drawdown: {max_dd:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Backtesting test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n🛠️ Testing Utilities...")
    
    try:
        from paper_trading.utils.helpers import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown
        from paper_trading.utils.metrics import PerformanceMetrics, TradingMetrics, RiskMetrics
        
        # Test helper functions
        prices = np.array([100, 101, 102, 101, 103, 104, 105])
        returns = calculate_returns(prices)
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(prices)
        
        print(f"✅ Helper functions - Returns: {len(returns)}, Sharpe: {sharpe:.4f}, Max DD: {max_dd:.4f}")
        
        # Test metrics classes
        perf_metrics = PerformanceMetrics()
        trading_metrics = TradingMetrics()
        risk_metrics = RiskMetrics()
        
        print("✅ All metrics classes initialized")
        
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_main_functionality():
    """Test main functionality"""
    print("\n🎯 Testing Main Functionality...")
    
    try:
        from paper_trading.main import create_config_file
        
        # Test config creation
        config_path = "test_config.yaml"
        create_config_file(config_path)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config file created with {len(config)} sections")
        
        # Clean up
        os.remove(config_path)
        print("✅ Test config file cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ Main functionality test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive Paper Trading System Test")
    print("=" * 60)
    
    setup_logging()
    
    tests = [
        ("Configuration Management", test_configs),
        ("Data Management", test_data_management),
        ("Models", test_models),
        ("Trading Engine", test_trading_engine),
        ("Backtesting", test_backtesting),
        ("Utilities", test_utils),
        ("Main Functionality", test_main_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Paper trading system is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test() 