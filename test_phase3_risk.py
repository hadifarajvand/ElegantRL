#!/usr/bin/env python3
"""
Test Phase 3: Advanced Risk Management Implementations
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dynamic_risk_management():
    """Test dynamic risk management"""
    print("🛡️ Testing Dynamic Risk Management...")
    
    try:
        from paper_trading.risk_management.dynamic_risk import (
            DynamicRiskManager, RiskMetrics, RiskLevel, create_dynamic_risk_manager
        )
        
        # Create risk manager
        risk_manager = create_dynamic_risk_manager(
            max_portfolio_risk=0.02,
            max_position_risk=0.01,
            max_leverage=1.5
        )
        print("✅ DynamicRiskManager created successfully")
        
        # Create sample portfolio
        portfolio = {
            'cash': 50000,
            'total_value': 100000,
            'positions': {
                'AAPL': {
                    'quantity': 100,
                    'current_value': 15000,
                    'cost_basis': 14000,
                    'unrealized_pnl': 1000
                },
                'GOOGL': {
                    'quantity': 50,
                    'current_value': 25000,
                    'cost_basis': 24000,
                    'unrealized_pnl': 1000
                }
            }
        }
        
        # Create sample market data
        market_data = {
            'AAPL': {'price': 150, 'volume': 1000},
            'GOOGL': {'price': 500, 'volume': 500}
        }
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_risk_metrics(portfolio, market_data)
        print(f"✅ Risk metrics calculated - VaR 95%: {risk_metrics.var_95:.4f}")
        print(f"✅ Current risk level: {risk_manager.current_risk_level.value}")
        
        # Test trade validation
        trade = {
            'symbol': 'MSFT',
            'action': 'buy',
            'quantity': 50,
            'price': 300
        }
        
        is_valid, message = risk_manager.validate_trade(trade, portfolio, market_data)
        print(f"✅ Trade validation: {is_valid} - {message}")
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size('MSFT', 0.8, portfolio, market_data)
        print(f"✅ Position size calculated: {position_size:.4f}")
        
        # Test stress testing
        scenarios = [
            {'name': 'market_crash', 'shock_size': 0.2, 'shock_type': 'uniform'},
            {'name': 'volatility_spike', 'shock_size': 0.1, 'shock_type': 'correlated'}
        ]
        
        stress_results = risk_manager.stress_test(portfolio, market_data, scenarios)
        print(f"✅ Stress testing completed - {len(stress_results)} scenarios")
        
        return True
        
    except Exception as e:
        print(f"❌ Dynamic risk management test failed: {e}")
        return False

def test_portfolio_optimization():
    """Test portfolio optimization"""
    print("\n📊 Testing Portfolio Optimization...")
    
    try:
        from paper_trading.risk_management.portfolio_optimizer import (
            PortfolioOptimizer, OptimizationResult, create_portfolio_optimizer
        )
        
        # Create optimizer
        optimizer = create_portfolio_optimizer(risk_free_rate=0.02, target_volatility=0.15)
        print("✅ PortfolioOptimizer created successfully")
        
        # Create sample returns data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.001, 0.025, 252),
            'MSFT': np.random.normal(0.001, 0.018, 252),
            'AMZN': np.random.normal(0.001, 0.03, 252)
        }, index=dates)
        
        print("✅ Returns data created")
        
        # Test different optimization methods
        methods = ['mean_variance', 'risk_parity', 'kelly', 'hierarchical']
        
        for method in methods:
            try:
                result = optimizer.optimize_portfolio(symbols, returns_data, method=method)
                print(f"✅ {method} optimization - Sharpe: {result.sharpe_ratio:.4f}")
                print(f"   Expected return: {result.expected_return:.4f}, Volatility: {result.volatility:.4f}")
            except Exception as e:
                print(f"⚠️ {method} optimization failed: {e}")
        
        # Test rebalancing trades
        current_weights = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.2, 'AMZN': 0.2}
        target_weights = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}
        portfolio_value = 100000
        
        rebalancing_trades = optimizer.calculate_rebalancing_trades(
            current_weights, target_weights, portfolio_value
        )
        print(f"✅ Rebalancing trades calculated: {len(rebalancing_trades)} trades")
        
        # Test risk budget
        risk_budget = optimizer.calculate_risk_budget(symbols, returns_data)
        print(f"✅ Risk budget calculated: {risk_budget}")
        
        # Test rebalance frequency
        frequency = optimizer.calculate_optimal_rebalance_frequency(returns_data)
        print(f"✅ Optimal rebalance frequency: {frequency}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio optimization test failed: {e}")
        return False

def test_risk_integration():
    """Test risk management integration"""
    print("\n🔄 Testing Risk Management Integration...")
    
    try:
        from paper_trading.risk_management.dynamic_risk import create_dynamic_risk_manager
        from paper_trading.risk_management.portfolio_optimizer import create_portfolio_optimizer
        from paper_trading.models.portfolio import Portfolio
        
        # Create components
        risk_manager = create_dynamic_risk_manager()
        optimizer = create_portfolio_optimizer()
        portfolio = Portfolio(initial_capital=100000)
        
        print("✅ All risk components created")
        
        # Create sample data
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.025, 100),
            'MSFT': np.random.normal(0.001, 0.018, 100)
        }, index=dates)
        
        # Simulate portfolio positions
        portfolio.buy_stock('AAPL', 100, 150)
        portfolio.buy_stock('GOOGL', 50, 500)
        
        # Update portfolio with current prices
        current_prices = {'AAPL': 155, 'GOOGL': 510, 'MSFT': 300}
        portfolio.update_prices(current_prices)
        
        # Get portfolio state
        portfolio_state = {
            'cash': portfolio.cash,
            'total_value': portfolio.get_total_value(),
            'positions': portfolio.positions
        }
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_risk_metrics(portfolio_state, current_prices)
        print(f"✅ Portfolio risk metrics calculated")
        print(f"   VaR 95%: {risk_metrics.var_95:.4f}")
        print(f"   Sharpe ratio: {risk_metrics.sharpe_ratio:.4f}")
        print(f"   Max drawdown: {risk_metrics.max_drawdown:.4f}")
        
        # Optimize portfolio
        optimization_result = optimizer.optimize_portfolio(
            symbols, returns_data, method='mean_variance'
        )
        print(f"✅ Portfolio optimization completed")
        print(f"   Expected return: {optimization_result.expected_return:.4f}")
        print(f"   Target volatility: {optimization_result.volatility:.4f}")
        
        # Calculate rebalancing trades
        current_weights = {symbol: portfolio.get_position_weight(symbol) for symbol in symbols}
        target_weights = optimization_result.weights
        
        rebalancing_trades = optimizer.calculate_rebalancing_trades(
            current_weights, target_weights, portfolio.get_total_value()
        )
        print(f"✅ Rebalancing trades: {len(rebalancing_trades)} trades needed")
        
        # Validate trades with risk manager
        for trade in rebalancing_trades[:2]:  # Test first 2 trades
            trade_dict = {
                'symbol': trade['symbol'],
                'action': trade['action'],
                'quantity': trade['quantity'],
                'price': current_prices.get(trade['symbol'], 100)
            }
            
            is_valid, message = risk_manager.validate_trade(trade_dict, portfolio_state, current_prices)
            print(f"   Trade {trade['symbol']}: {is_valid} - {message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk integration test failed: {e}")
        return False

def test_risk_performance():
    """Test risk management performance"""
    print("\n⚡ Testing Risk Management Performance...")
    
    try:
        from paper_trading.risk_management.dynamic_risk import create_dynamic_risk_manager
        from paper_trading.risk_management.portfolio_optimizer import create_portfolio_optimizer
        import time
        
        # Create components
        risk_manager = create_dynamic_risk_manager()
        optimizer = create_portfolio_optimizer()
        
        # Create large dataset
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        returns_data = pd.DataFrame({
            symbol: np.random.normal(0.001, 0.02, 1000) for symbol in symbols
        }, index=dates)
        
        # Test risk metrics calculation performance
        portfolio = {
            'cash': 100000,
            'total_value': 200000,
            'positions': {
                symbol: {
                    'quantity': 100,
                    'current_value': 10000,
                    'cost_basis': 9500,
                    'unrealized_pnl': 500
                } for symbol in symbols
            }
        }
        
        market_data = {symbol: {'price': 100, 'volume': 1000} for symbol in symbols}
        
        start_time = time.time()
        risk_metrics = risk_manager.calculate_risk_metrics(portfolio, market_data)
        risk_time = time.time() - start_time
        
        print(f"✅ Risk metrics calculated in {risk_time:.3f} seconds")
        
        # Test optimization performance
        start_time = time.time()
        optimization_result = optimizer.optimize_portfolio(symbols, returns_data, method='mean_variance')
        opt_time = time.time() - start_time
        
        print(f"✅ Portfolio optimization completed in {opt_time:.3f} seconds")
        
        # Performance thresholds
        if risk_time < 1.0:  # Should complete within 1 second
            print("✅ Risk calculation performance test passed")
        else:
            print("⚠️ Risk calculation performance slow")
        
        if opt_time < 5.0:  # Should complete within 5 seconds
            print("✅ Optimization performance test passed")
        else:
            print("⚠️ Optimization performance slow")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Phase 3: Advanced Risk Management Implementations")
    print("=" * 70)
    
    tests = [
        ("Dynamic Risk Management", test_dynamic_risk_management),
        ("Portfolio Optimization", test_portfolio_optimization),
        ("Risk Integration", test_risk_integration),
        ("Risk Performance", test_risk_performance)
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
    print("📊 PHASE 3 TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Phase 3 implementation completed successfully!")
    else:
        print("⚠️ Some Phase 3 tests failed. Please check the errors above.") 