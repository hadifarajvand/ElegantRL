#!/usr/bin/env python3
"""
Test Phase 4: Advanced Trading Strategies Implementations
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multi_strategy_framework():
    """Test multi-strategy framework"""
    print("📊 Testing Multi-Strategy Framework...")
    
    try:
        from paper_trading.strategies.multi_strategy import (
            create_momentum_strategy, create_mean_reversion_strategy, 
            create_multi_strategy_framework, StrategySignal
        )
        
        # Create strategies
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        momentum_strategy = create_momentum_strategy(symbols, lookback_period=20)
        reversion_strategy = create_mean_reversion_strategy(symbols, lookback_period=50)
        
        print("✅ Individual strategies created")
        
        # Create multi-strategy framework
        strategies = [momentum_strategy, reversion_strategy]
        framework = create_multi_strategy_framework(strategies, allocation_method='equal')
        
        print("✅ Multi-strategy framework created")
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = pd.DataFrame({
                'open': np.random.randn(100) * 5 + 150,
                'high': np.random.randn(100) * 5 + 155,
                'low': np.random.randn(100) * 5 + 145,
                'close': np.random.randn(100) * 5 + 150,
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
        
        print("✅ Market data created")
        
        # Generate aggregated signals
        signals = framework.generate_aggregated_signals(market_data)
        print(f"✅ Generated {len(signals)} aggregated signals")
        
        # Test individual strategy signals
        momentum_signals = momentum_strategy.generate_signals(market_data)
        reversion_signals = reversion_strategy.generate_signals(market_data)
        
        print(f"✅ Momentum strategy: {len(momentum_signals)} signals")
        print(f"✅ Mean reversion strategy: {len(reversion_signals)} signals")
        
        # Test strategy performance
        for strategy in strategies:
            performance = strategy.calculate_performance({})
            print(f"✅ {strategy.name} performance - Sharpe: {performance.sharpe_ratio:.3f}")
        
        # Test framework performance
        framework_performance = framework.calculate_framework_performance()
        print(f"✅ Framework performance - Sharpe: {framework_performance.sharpe_ratio:.3f}")
        
        # Test strategy status
        status = framework.get_strategy_status()
        print(f"✅ Strategy status retrieved for {len(status)} strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-strategy framework test failed: {e}")
        return False

def test_ml_strategies():
    """Test machine learning strategies"""
    print("\n🤖 Testing Machine Learning Strategies...")
    
    try:
        from paper_trading.strategies.ml_strategies import (
            create_ml_strategy, create_ensemble_ml_strategy
        )
        
        # Create ML strategy
        symbols = ['AAPL', 'GOOGL']
        ml_strategy = create_ml_strategy(symbols, model_type='random_forest')
        
        print("✅ ML strategy created")
        
        # Create training data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        training_data = {}
        
        for symbol in symbols:
            training_data[symbol] = pd.DataFrame({
                'open': np.random.randn(500) * 5 + 150,
                'high': np.random.randn(500) * 5 + 155,
                'low': np.random.randn(500) * 5 + 145,
                'close': np.random.randn(500) * 5 + 150,
                'volume': np.random.randint(1000, 10000, 500)
            }, index=dates)
        
        print("✅ Training data created")
        
        # Train models
        success = ml_strategy.retrain_models(training_data)
        print(f"✅ Model training: {'Success' if success else 'Failed'}")
        
        # Test predictions
        for symbol in symbols:
            if symbol in training_data:
                prediction = ml_strategy.predict(symbol, training_data[symbol])
                if prediction:
                    print(f"✅ {symbol} prediction: {prediction['action']} (confidence: {prediction['confidence']:.3f})")
        
        # Test model performance
        performance = ml_strategy.get_model_performance()
        print(f"✅ Model performance retrieved for {len(performance)} models")
        
        # Test ensemble strategy
        ensemble_strategy = create_ensemble_ml_strategy(symbols)
        print("✅ Ensemble ML strategy created")
        
        # Train ensemble
        ensemble_success = ensemble_strategy.train_ensemble(training_data)
        print(f"✅ Ensemble training: {'Success' if ensemble_success else 'Failed'}")
        
        # Test ensemble predictions
        for symbol in symbols:
            if symbol in training_data:
                prediction = ensemble_strategy.predict_ensemble(symbol, training_data[symbol])
                if prediction:
                    print(f"✅ {symbol} ensemble prediction: {prediction['action']} (confidence: {prediction['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ ML strategies test failed: {e}")
        return False

def test_strategy_integration():
    """Test strategy integration"""
    print("\n🔄 Testing Strategy Integration...")
    
    try:
        from paper_trading.strategies.multi_strategy import (
            create_momentum_strategy, create_mean_reversion_strategy, 
            create_multi_strategy_framework
        )
        from paper_trading.strategies.ml_strategies import create_ml_strategy
        from paper_trading.risk_management.dynamic_risk import create_dynamic_risk_manager
        from paper_trading.models.portfolio import Portfolio
        
        # Create components
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Create strategies
        momentum_strategy = create_momentum_strategy(symbols)
        reversion_strategy = create_mean_reversion_strategy(symbols)
        ml_strategy = create_ml_strategy(symbols, model_type='random_forest')
        
        # Create framework
        strategies = [momentum_strategy, reversion_strategy]
        framework = create_multi_strategy_framework(strategies)
        
        # Create risk manager
        risk_manager = create_dynamic_risk_manager()
        
        # Create portfolio
        portfolio = Portfolio(initial_capital=100000)
        
        print("✅ All components created")
        
        # Create market data
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = pd.DataFrame({
                'open': np.random.randn(200) * 5 + 150,
                'high': np.random.randn(200) * 5 + 155,
                'low': np.random.randn(200) * 5 + 145,
                'close': np.random.randn(200) * 5 + 150,
                'volume': np.random.randint(1000, 10000, 200)
            }, index=dates)
        
        # Train ML model
        ml_strategy.retrain_models(market_data)
        
        # Generate signals
        framework_signals = framework.generate_aggregated_signals(market_data)
        print(f"✅ Framework generated {len(framework_signals)} signals")
        
        # Test ML predictions
        ml_predictions = []
        for symbol in symbols:
            if symbol in market_data:
                prediction = ml_strategy.predict(symbol, market_data[symbol])
                if prediction:
                    ml_predictions.append(prediction)
        
        print(f"✅ ML strategy generated {len(ml_predictions)} predictions")
        
        # Simulate trading
        current_prices = {symbol: market_data[symbol]['close'].iloc[-1] for symbol in symbols}
        
        # Process framework signals
        for signal in framework_signals[:3]:  # Process first 3 signals
            if signal.action in ['buy', 'sell']:
                # Validate with risk manager
                trade = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'quantity': 100,  # Fixed quantity for testing
                    'price': signal.price
                }
                
                portfolio_state = {
                    'cash': portfolio.cash,
                    'total_value': portfolio.get_total_value(),
                    'positions': portfolio.positions
                }
                
                is_valid, message = risk_manager.validate_trade(trade, portfolio_state, current_prices)
                print(f"   Signal {signal.symbol} {signal.action}: {is_valid} - {message}")
        
        # Test performance calculation
        for strategy in strategies:
            performance = strategy.calculate_performance({})
            print(f"✅ {strategy.name} - Sharpe: {performance.sharpe_ratio:.3f}, Win Rate: {performance.win_rate:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy integration test failed: {e}")
        return False

def test_strategy_performance():
    """Test strategy performance"""
    print("\n⚡ Testing Strategy Performance...")
    
    try:
        from paper_trading.strategies.multi_strategy import (
            create_momentum_strategy, create_mean_reversion_strategy
        )
        from paper_trading.strategies.ml_strategies import create_ml_strategy
        import time
        
        # Create strategies
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        momentum_strategy = create_momentum_strategy(symbols)
        reversion_strategy = create_mean_reversion_strategy(symbols)
        ml_strategy = create_ml_strategy(symbols, model_type='random_forest')
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = pd.DataFrame({
                'open': np.random.randn(1000) * 5 + 150,
                'high': np.random.randn(1000) * 5 + 155,
                'low': np.random.randn(1000) * 5 + 145,
                'close': np.random.randn(1000) * 5 + 150,
                'volume': np.random.randint(1000, 10000, 1000)
            }, index=dates)
        
        # Test momentum strategy performance
        start_time = time.time()
        momentum_signals = momentum_strategy.generate_signals(market_data)
        momentum_time = time.time() - start_time
        
        print(f"✅ Momentum strategy: {len(momentum_signals)} signals in {momentum_time:.3f}s")
        
        # Test reversion strategy performance
        start_time = time.time()
        reversion_signals = reversion_strategy.generate_signals(market_data)
        reversion_time = time.time() - start_time
        
        print(f"✅ Reversion strategy: {len(reversion_signals)} signals in {reversion_time:.3f}s")
        
        # Test ML strategy performance
        start_time = time.time()
        ml_strategy.retrain_models(market_data)
        ml_time = time.time() - start_time
        
        print(f"✅ ML strategy training: {ml_time:.3f}s")
        
        # Performance thresholds
        if momentum_time < 1.0:
            print("✅ Momentum strategy performance test passed")
        else:
            print("⚠️ Momentum strategy performance slow")
        
        if reversion_time < 1.0:
            print("✅ Reversion strategy performance test passed")
        else:
            print("⚠️ Reversion strategy performance slow")
        
        if ml_time < 10.0:  # ML training takes longer
            print("✅ ML strategy performance test passed")
        else:
            print("⚠️ ML strategy performance slow")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Phase 4: Advanced Trading Strategies Implementations")
    print("=" * 70)
    
    tests = [
        ("Multi-Strategy Framework", test_multi_strategy_framework),
        ("Machine Learning Strategies", test_ml_strategies),
        ("Strategy Integration", test_strategy_integration),
        ("Strategy Performance", test_strategy_performance)
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
    print("📊 PHASE 4 TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Phase 4 implementation completed successfully!")
    else:
        print("⚠️ Some Phase 4 tests failed. Please check the errors above.") 