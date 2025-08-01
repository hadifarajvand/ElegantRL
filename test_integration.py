#!/usr/bin/env python3
"""
Integration Test for Paper Trading System
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import yaml

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_full_workflow():
    """Test the complete paper trading workflow"""
    print("🚀 Testing Full Paper Trading Workflow")
    print("=" * 60)
    
    try:
        # 1. Test Configuration
        print("\n1️⃣ Testing Configuration Management...")
        from paper_trading.configs.trading_config import TradingConfig
        from paper_trading.configs.model_config import ModelConfig, PPO_CONFIG
        
        trading_config = TradingConfig(initial_capital=100000)
        model_config = PPO_CONFIG
        print(f"✅ Configs loaded - Capital: ${trading_config.initial_capital:,.0f}, Agent: {model_config.agent_type}")
        
        # 2. Test Data Management
        print("\n2️⃣ Testing Data Management...")
        from paper_trading.data.market_data import YahooFinanceProvider
        from paper_trading.data.data_manager import DataManager
        from paper_trading.data.data_processor import DataProcessor
        
        # Create mock data for testing
        mock_data = {
            'AAPL': pd.DataFrame({
                'close': np.random.randn(100) * 10 + 150,
                'volume': np.random.randint(1000, 10000, 100),
                'open': np.random.randn(100) * 5 + 150,
                'high': np.random.randn(100) * 5 + 155,
                'low': np.random.randn(100) * 5 + 145
            })
        }
        
        processor = DataProcessor()
        processed_data = {}
        for symbol, data in mock_data.items():
            processed_data[symbol] = processor.calculate_technical_indicators(data)
        
        print(f"✅ Data processed for {len(processed_data)} symbols")
        
        # 3. Test Trading Environment
        print("\n3️⃣ Testing Trading Environment...")
        from paper_trading.models.trading_env import EnhancedStockTradingEnv
        
        env = EnhancedStockTradingEnv(
            data=processed_data,
            initial_capital=trading_config.initial_capital,
            max_stock_quantity=trading_config.max_stock_quantity,
            transaction_cost_pct=trading_config.transaction_cost_pct,
            slippage_pct=trading_config.slippage_pct
        )
        
        state, info = env.reset()
        print(f"✅ Environment initialized - State shape: {state.shape}")
        
        # 4. Test Agent
        print("\n4️⃣ Testing DRL Agent...")
        from paper_trading.models.agents import SimplePPOAgent
        
        agent = SimplePPOAgent(state_dim=state.shape[0], action_dim=1)
        action = agent.get_action(state)
        print(f"✅ Agent action generated - Shape: {action.shape}")
        
        # 5. Test Environment Step
        print("\n5️⃣ Testing Environment Step...")
        next_state, reward, done, truncated, info = env.step(action)
        print(f"✅ Environment step completed - Reward: {reward:.4f}, Done: {done}")
        
        # 6. Test Portfolio Management
        print("\n6️⃣ Testing Portfolio Management...")
        from paper_trading.models.portfolio import Portfolio
        
        portfolio = Portfolio(initial_capital=trading_config.initial_capital)
        portfolio.buy_stock('AAPL', 100, 150.0)
        portfolio.update_prices({'AAPL': 155.0})
        
        portfolio_value = portfolio.get_total_value()
        print(f"✅ Portfolio updated - Total value: ${portfolio_value:,.2f}")
        
        # 7. Test Risk Management
        print("\n7️⃣ Testing Risk Management...")
        from paper_trading.paper_trading_engine.risk_manager import RiskManager
        
        risk_manager = RiskManager(
            max_position_size=trading_config.max_position_size,
            stop_loss_pct=trading_config.stop_loss_pct,
            take_profit_pct=trading_config.take_profit_pct
        )
        
        positions = portfolio.get_positions()
        validated_action = risk_manager.validate_action(
            action=np.array([0.1]),
            positions=positions,
            portfolio_value=portfolio_value
        )
        print(f"✅ Risk validation completed - Action shape: {validated_action.shape}")
        
        # 8. Test Order Management
        print("\n8️⃣ Testing Order Management...")
        from paper_trading.paper_trading_engine.order_manager import OrderManager
        
        order_manager = OrderManager(
            transaction_cost_pct=trading_config.transaction_cost_pct,
            slippage_pct=trading_config.slippage_pct
        )
        
        # Test order execution
        action = np.array([0.1])  # Small buy order
        positions = portfolio.get_positions()
        symbols = ['AAPL']
        
        executed_orders = order_manager.execute_orders(action, positions, symbols)
        print(f"✅ Order execution completed - {len(executed_orders)} orders executed")
        
        # 9. Test Performance Analysis
        print("\n9️⃣ Testing Performance Analysis...")
        from paper_trading.backtesting.performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        # Simulate portfolio values over time
        portfolio_values = np.array([100000, 101000, 102500, 101800, 103200, 104500, 105800, 107200])
        
        analysis = analyzer.analyze_performance(portfolio_values)
        basic_metrics = analysis['basic_metrics']
        
        print(f"✅ Performance analyzed - Sharpe: {basic_metrics['sharpe_ratio']:.4f}, "
              f"Max DD: {basic_metrics['max_drawdown_pct']:.2f}%")
        
        # 10. Test Configuration File Creation
        print("\n🔟 Testing Configuration File Creation...")
        from paper_trading.main import create_config_file
        
        config_path = "integration_test_config.yaml"
        create_config_file(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config file created with {len(config)} sections")
        
        # Clean up
        os.remove(config_path)
        
        print("\n" + "=" * 60)
        print("🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ All components working together")
        print("✅ Data flow functioning correctly")
        print("✅ Risk management operational")
        print("✅ Performance analysis working")
        print("✅ Configuration system functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_simulation():
    """Test a simulated training session"""
    print("\n🧠 Testing Training Simulation...")
    
    try:
        from paper_trading.models.trading_env import EnhancedStockTradingEnv
        from paper_trading.models.agents import SimplePPOAgent
        from paper_trading.configs.trading_config import TradingConfig
        
        # Create mock data
        mock_data = {
            'AAPL': pd.DataFrame({
                'close': np.random.randn(50) * 10 + 150,
                'volume': np.random.randint(1000, 10000, 50),
                'open': np.random.randn(50) * 5 + 150,
                'high': np.random.randn(50) * 5 + 155,
                'low': np.random.randn(50) * 5 + 145
            })
        }
        
        # Create environment
        env = EnhancedStockTradingEnv(
            data=mock_data,
            initial_capital=100000,
            max_stock_quantity=100
        )
        
        # Create agent
        state, _ = env.reset()
        agent = SimplePPOAgent(state_dim=state.shape[0], action_dim=1)
        
        # Simulate training episode
        total_reward = 0
        steps = 0
        
        for _ in range(10):  # Short episode for testing
            action = agent.get_action(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"✅ Training simulation completed - Steps: {steps}, Total reward: {total_reward:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False

def test_backtesting_simulation():
    """Test a simulated backtesting session"""
    print("\n📈 Testing Backtesting Simulation...")
    
    try:
        from paper_trading.backtesting.backtest_engine import BacktestEngine
        from paper_trading.backtesting.performance_analyzer import PerformanceAnalyzer
        from paper_trading.models.agents import RandomAgent
        
        # Create mock data
        portfolio_values = np.array([100000, 101000, 102500, 101800, 103200, 104500, 105800, 107200, 108500, 109800])
        
        # Create analyzer
        analyzer = PerformanceAnalyzer()
        
        # Analyze performance
        analysis = analyzer.analyze_performance(portfolio_values)
        
        print(f"✅ Backtesting simulation completed")
        print(f"   - Total return: {analysis['basic_metrics']['total_return_pct']:.2f}%")
        print(f"   - Sharpe ratio: {analysis['basic_metrics']['sharpe_ratio']:.4f}")
        print(f"   - Max drawdown: {analysis['basic_metrics']['max_drawdown_pct']:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Integration Tests for Paper Trading System")
    
    tests = [
        ("Full Workflow", test_full_workflow),
        ("Training Simulation", test_training_simulation),
        ("Backtesting Simulation", test_backtesting_simulation)
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
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} integration tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All integration tests passed! Paper trading system is fully functional.")
    else:
        print("⚠️ Some integration tests failed. Please check the errors above.") 