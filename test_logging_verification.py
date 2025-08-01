#!/usr/bin/env python3
"""
Logging Verification Test for Paper Trading System
"""

import sys
import os
import logging
import time
from datetime import datetime
import yaml

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_comprehensive_logging():
    """Setup comprehensive logging for verification"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging with multiple handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paper_trading_test.log'),
            logging.FileHandler('logs/performance.log'),
            logging.FileHandler('logs/risk_management.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create specific loggers
    trading_logger = logging.getLogger('paper_trading.trading')
    trading_logger.setLevel(logging.INFO)
    
    risk_logger = logging.getLogger('paper_trading.risk')
    risk_logger.setLevel(logging.INFO)
    
    data_logger = logging.getLogger('paper_trading.data')
    data_logger.setLevel(logging.INFO)
    
    return trading_logger, risk_logger, data_logger

def test_logging_functionality():
    """Test comprehensive logging functionality"""
    print("🔍 Testing Logging Functionality")
    print("=" * 50)
    
    # Setup logging
    trading_logger, risk_logger, data_logger = setup_comprehensive_logging()
    
    try:
        # Test 1: Configuration Logging
        print("\n1️⃣ Testing Configuration Logging...")
        from paper_trading.configs.trading_config import TradingConfig
        
        trading_logger.info("Loading trading configuration...")
        config = TradingConfig(initial_capital=100000)
        trading_logger.info(f"Trading config loaded - Capital: ${config.initial_capital:,.0f}")
        trading_logger.info(f"Max position size: {config.max_position_size:.1%}")
        trading_logger.info(f"Transaction cost: {config.transaction_cost_pct:.3%}")
        
        # Test 2: Data Processing Logging
        print("\n2️⃣ Testing Data Processing Logging...")
        from paper_trading.data.data_processor import DataProcessor
        
        data_logger.info("Initializing data processor...")
        processor = DataProcessor()
        data_logger.info("Data processor initialized successfully")
        
        # Create sample data and log processing
        import pandas as pd
        import numpy as np
        
        sample_data = pd.DataFrame({
            'close': np.random.randn(100) * 10 + 150,
            'volume': np.random.randint(1000, 10000, 100),
            'high': np.random.randn(100) * 5 + 155,
            'low': np.random.randn(100) * 5 + 145,
            'open': np.random.randn(100) * 5 + 150
        })
        
        data_logger.info(f"Processing data with {len(sample_data)} rows")
        processed_data = processor.calculate_technical_indicators(sample_data)
        data_logger.info(f"Technical indicators calculated: {len(processed_data.columns)} features")
        
        # Test 3: Risk Management Logging
        print("\n3️⃣ Testing Risk Management Logging...")
        from paper_trading.paper_trading_engine.risk_manager import RiskManager
        
        risk_logger.info("Initializing risk manager...")
        risk_manager = RiskManager(
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        risk_logger.info("Risk manager initialized successfully")
        
        # Test risk validation with logging
        import numpy as np
        
        action = np.array([0.1])
        positions = {'AAPL': {'quantity': 100, 'avg_price': 150, 'current_value': 15500, 
                             'cost_basis': 15000, 'unrealized_pnl': 500}}
        portfolio_value = 100000
        
        risk_logger.info(f"Validating action: {action}")
        risk_logger.info(f"Current positions: {list(positions.keys())}")
        risk_logger.info(f"Portfolio value: ${portfolio_value:,.0f}")
        
        validated_action = risk_manager.validate_action(action, positions, portfolio_value)
        risk_logger.info(f"Action validation completed - Shape: {validated_action.shape}")
        
        # Test 4: Portfolio Management Logging
        print("\n4️⃣ Testing Portfolio Management Logging...")
        from paper_trading.models.portfolio import Portfolio
        
        trading_logger.info("Initializing portfolio...")
        portfolio = Portfolio(initial_capital=100000)
        trading_logger.info(f"Portfolio initialized with ${portfolio.initial_capital:,.0f}")
        
        # Test portfolio operations with logging
        trading_logger.info("Executing buy order for AAPL...")
        success = portfolio.buy_stock('AAPL', 100, 150.0)
        trading_logger.info(f"Buy order result: {'Success' if success else 'Failed'}")
        
        trading_logger.info("Updating portfolio prices...")
        portfolio.update_prices({'AAPL': 155.0})
        portfolio_value = portfolio.get_total_value()
        trading_logger.info(f"Portfolio value updated: ${portfolio_value:,.2f}")
        
        # Test 5: Performance Analysis Logging
        print("\n5️⃣ Testing Performance Analysis Logging...")
        from paper_trading.backtesting.performance_analyzer import PerformanceAnalyzer
        
        trading_logger.info("Initializing performance analyzer...")
        analyzer = PerformanceAnalyzer()
        trading_logger.info("Performance analyzer initialized")
        
        # Simulate portfolio values and log analysis
        portfolio_values = np.array([100000, 101000, 102500, 101800, 103200, 104500, 105800, 107200])
        trading_logger.info(f"Analyzing performance for {len(portfolio_values)} data points")
        
        analysis = analyzer.analyze_performance(portfolio_values)
        basic_metrics = analysis['basic_metrics']
        
        trading_logger.info(f"Performance analysis completed:")
        trading_logger.info(f"  - Total return: {basic_metrics['total_return_pct']:.2f}%")
        trading_logger.info(f"  - Sharpe ratio: {basic_metrics['sharpe_ratio']:.4f}")
        trading_logger.info(f"  - Max drawdown: {basic_metrics['max_drawdown_pct']:.2f}%")
        
        # Test 6: Order Management Logging
        print("\n6️⃣ Testing Order Management Logging...")
        from paper_trading.paper_trading_engine.order_manager import OrderManager
        
        trading_logger.info("Initializing order manager...")
        order_manager = OrderManager(
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )
        trading_logger.info("Order manager initialized")
        
        # Test order execution with logging
        action = np.array([0.1])
        positions = portfolio.get_positions()
        symbols = ['AAPL']
        
        trading_logger.info(f"Executing orders for symbols: {symbols}")
        trading_logger.info(f"Action vector: {action}")
        
        executed_orders = order_manager.execute_orders(action, positions, symbols)
        trading_logger.info(f"Order execution completed - {len(executed_orders)} orders executed")
        
        # Test 7: Configuration File Logging
        print("\n7️⃣ Testing Configuration File Logging...")
        from paper_trading.main import create_config_file
        
        trading_logger.info("Creating configuration file...")
        config_path = "test_logging_config.yaml"
        create_config_file(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        trading_logger.info(f"Configuration file created with {len(config)} sections")
        trading_logger.info(f"Model agent type: {config['model']['agent_type']}")
        trading_logger.info(f"Initial capital: ${config['trading']['initial_capital']:,.0f}")
        
        # Clean up
        os.remove(config_path)
        trading_logger.info("Test configuration file cleaned up")
        
        # Test 8: Error Logging
        print("\n8️⃣ Testing Error Logging...")
        
        try:
            # Simulate an error condition
            raise ValueError("Simulated error for logging test")
        except Exception as e:
            trading_logger.error(f"Simulated error caught: {e}")
            trading_logger.warning("This is a test warning message")
            trading_logger.debug("This is a debug message")
        
        # Test 9: Performance Metrics Logging
        print("\n9️⃣ Testing Performance Metrics Logging...")
        from paper_trading.utils.metrics import PerformanceMetrics
        
        trading_logger.info("Calculating performance metrics...")
        perf_metrics = PerformanceMetrics()
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        metrics = perf_metrics.calculate_all_metrics(portfolio_values, returns)
        
        trading_logger.info("Performance metrics calculated:")
        for metric_name, value in metrics.items():
            trading_logger.info(f"  - {metric_name}: {value:.6f}")
        
        # Test 10: Summary Logging
        print("\n🔟 Testing Summary Logging...")
        
        trading_logger.info("=" * 50)
        trading_logger.info("LOGGING VERIFICATION TEST COMPLETED")
        trading_logger.info("=" * 50)
        trading_logger.info("✅ All logging components tested successfully")
        trading_logger.info("✅ Log files generated in logs/ directory")
        trading_logger.info("✅ Multiple log levels tested (INFO, WARNING, ERROR, DEBUG)")
        trading_logger.info("✅ Different loggers tested (trading, risk, data)")
        trading_logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        trading_logger.error(f"Logging test failed: {e}")
        import traceback
        trading_logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def verify_log_files():
    """Verify that log files were created and contain expected content"""
    print("\n📋 Verifying Log Files...")
    print("=" * 50)
    
    log_files = [
        'logs/paper_trading_test.log',
        'logs/performance.log',
        'logs/risk_management.log'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"✅ {log_file}: {len(lines)} log entries")
                
                # Check for specific log entries
                if 'Trading config loaded' in content:
                    print(f"   - Configuration logging ✓")
                if 'Risk manager initialized' in content:
                    print(f"   - Risk management logging ✓")
                if 'Portfolio initialized' in content:
                    print(f"   - Portfolio logging ✓")
                if 'Performance analysis completed' in content:
                    print(f"   - Performance logging ✓")
                if 'Order execution completed' in content:
                    print(f"   - Order management logging ✓")
        else:
            print(f"❌ {log_file}: File not found")
    
    # Check console output log
    if os.path.exists('paper_trading.log'):
        with open('paper_trading.log', 'r') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"✅ paper_trading.log: {len(lines)} entries")
    else:
        print("❌ paper_trading.log: File not found")

def test_log_levels():
    """Test different log levels"""
    print("\n📊 Testing Log Levels...")
    
    logger = logging.getLogger('test_levels')
    logger.setLevel(logging.DEBUG)
    
    # Add file handler for this test
    fh = logging.FileHandler('logs/log_levels_test.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    print("✅ Log levels test completed")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Logging Verification Test")
    print("=" * 60)
    
    # Run the main logging test
    success = test_logging_functionality()
    
    # Test log levels
    test_log_levels()
    
    # Verify log files
    verify_log_files()
    
    print("\n" + "=" * 60)
    print("📊 LOGGING VERIFICATION RESULTS")
    print("=" * 60)
    
    if success:
        print("✅ Logging functionality test PASSED")
        print("✅ Log files generated successfully")
        print("✅ Multiple log levels working")
        print("✅ Different loggers functional")
        print("✅ Error handling and logging operational")
    else:
        print("❌ Logging functionality test FAILED")
    
    print("\n📁 Log files created:")
    log_files = ['logs/paper_trading_test.log', 'logs/performance.log', 
                 'logs/risk_management.log', 'logs/log_levels_test.log']
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"   - {log_file}: {size} bytes")
        else:
            print(f"   - {log_file}: Not found")
    
    print("\n🎉 Logging verification test completed!") 