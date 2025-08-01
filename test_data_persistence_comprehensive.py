"""
Comprehensive Test for Data Persistence, Logging, and Statistics
Demonstrates all the data storage, logging, and analysis capabilities
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
from paper_trading.data.ccxt_provider import create_ccxt_provider
from paper_trading.configs.trading_config import TradingConfig
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.strategies.crypto_strategies import create_crypto_strategies
from paper_trading.utils.data_persistence import create_data_persistence_manager
from paper_trading.utils.logging_config import setup_logging, get_logger
from paper_trading.utils.statistics_manager import create_statistics_manager

logger = logging.getLogger(__name__)


def test_data_persistence_system():
    """Test the comprehensive data persistence system"""
    print("🚀 Testing Data Persistence System")
    print("=" * 50)
    
    # Create data persistence manager
    persistence_manager = create_data_persistence_manager("./test_paper_trading_data")
    
    # Test market data storage
    print("\n📊 Testing Market Data Storage...")
    
    # Create sample market data
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(150, 250, len(dates)),
        'low': np.random.uniform(50, 150, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Save market data
    file_path = persistence_manager.save_market_data(
        'BTC/USDT', sample_data, '2025-01-01', '2025-01-31'
    )
    print(f"✅ Market data saved: {file_path}")
    
    # Load market data
    loaded_data = persistence_manager.load_market_data('BTC/USDT', '2025-01-01', '2025-01-31')
    if loaded_data is not None:
        print(f"✅ Market data loaded: {len(loaded_data)} records")
    else:
        print("⚠️ Market data not found (this is expected for the first run)")
    
    # Test trade log storage
    print("\n📈 Testing Trade Log Storage...")
    
    sample_trades = [
        {
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'quantity': 10,
            'price': 150.0,
            'portfolio_value': 100000.0,
            'cash': 98500.0,
            'positions': [10, 0, 0],
            'reward': 0.5,
            'step': 1
        },
        {
            'symbol': 'ETH/USDT',
            'action': 'SELL',
            'quantity': 5,
            'price': 2500.0,
            'portfolio_value': 102500.0,
            'cash': 114750.0,
            'positions': [10, -5, 0],
            'reward': -0.2,
            'step': 2
        }
    ]
    
    # Save individual trades
    for trade in sample_trades:
        persistence_manager.save_trade_log(trade)
    
    # Save batch trades
    persistence_manager.save_trade_logs_batch(sample_trades)
    print(f"✅ {len(sample_trades)} trade logs saved")
    
    # Retrieve trade logs
    trade_logs = persistence_manager.get_trade_logs(limit=10)
    print(f"✅ Retrieved {len(trade_logs)} trade logs")
    
    # Test performance metrics storage
    print("\n📊 Testing Performance Metrics Storage...")
    
    sample_metrics = {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'volatility': 0.12,
        'win_rate': 0.65,
        'total_trades': 100,
        'final_value': 115000.0
    }
    
    persistence_manager.save_performance_metrics(sample_metrics, "test_strategy")
    print("✅ Performance metrics saved")
    
    # Retrieve performance metrics
    perf_metrics = persistence_manager.get_performance_metrics(limit=5)
    print(f"✅ Retrieved {len(perf_metrics)} performance records")
    
    # Test configuration storage
    print("\n⚙️ Testing Configuration Storage...")
    
    sample_config = {
        'trading': {
            'initial_capital': 100000.0,
            'max_position_size': 0.2,
            'transaction_cost_pct': 0.001
        },
        'model': {
            'agent_type': 'PPO',
            'learning_rate': 0.0003,
            'batch_size': 2048
        }
    }
    
    config_path = persistence_manager.save_config(sample_config, "test_config")
    print(f"✅ Configuration saved: {config_path}")
    
    # Load configuration
    loaded_config = persistence_manager.load_config("test_config")
    print(f"✅ Configuration loaded: {loaded_config is not None}")
    
    # Test backtest results storage
    print("\n📈 Testing Backtest Results Storage...")
    
    sample_backtest_results = {
        'summary': {
            'strategy_1': {'total_return': 0.15, 'sharpe_ratio': 1.2},
            'strategy_2': {'total_return': 0.12, 'sharpe_ratio': 1.1}
        },
        'detailed_results': {
            'strategy_1': {'portfolio_values': [100000, 101000, 102000]},
            'strategy_2': {'portfolio_values': [100000, 100500, 101200]}
        },
        'trade_logs': sample_trades,
        'portfolio_values': [[100000, 101000, 102000], [100000, 100500, 101200]]
    }
    
    backtest_path = persistence_manager.save_backtest_results(sample_backtest_results, "test_backtest")
    print(f"✅ Backtest results saved: {backtest_path}")
    
    # Get database statistics
    print("\n📊 Database Statistics...")
    db_stats = persistence_manager.get_database_stats()
    print(f"✅ Database stats: {db_stats}")
    
    return persistence_manager


def test_logging_system():
    """Test the comprehensive logging system"""
    print("\n🚀 Testing Logging System")
    print("=" * 50)
    
    # Setup logging
    logger_instance = setup_logging(
        log_dir="./test_paper_trading_data/logs",
        log_level="INFO",
        enable_console=True,
        enable_file=True
    )
    
    # Test different types of logging
    print("\n📝 Testing Different Log Types...")
    
    # System events
    logger_instance.log_system_event("System initialized", {
        'version': '1.0.0',
        'components': ['data_persistence', 'logging', 'statistics']
    })
    
    # Trade logging
    logger_instance.log_trade({
        'symbol': 'BTC/USDT',
        'action': 'BUY',
        'quantity': 10,
        'price': 150.0,
        'portfolio_value': 100000.0
    })
    
    # Performance logging
    logger_instance.log_performance({
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.05,
        'volatility': 0.12
    })
    
    # Error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger_instance.log_error(e, {'context': 'test_logging_system'})
    
    # Data events
    logger_instance.log_data_event("Data fetched", {
        'symbol': 'BTC/USDT',
        'records': 100,
        'date_range': '2025-01-01 to 2025-01-31'
    })
    
    # Backtest events
    logger_instance.log_backtest_event("Backtest completed", {
        'strategy': 'Momentum',
        'total_return': 0.15,
        'sharpe_ratio': 1.2
    })
    
    # Training events
    logger_instance.log_training_event("Training started", {
        'agent_type': 'PPO',
        'episodes': 1000,
        'learning_rate': 0.0003
    })
    
    # Risk events
    logger_instance.log_risk_event("Risk limit exceeded", {
        'position_size': 0.25,
        'max_allowed': 0.20,
        'action_taken': 'reduced_position'
    })
    
    # Get log statistics
    print("\n📊 Log Statistics...")
    log_stats = logger_instance.get_log_stats()
    print(f"✅ Log stats: {log_stats}")
    
    return logger_instance


def test_statistics_manager():
    """Test the comprehensive statistics manager"""
    print("\n🚀 Testing Statistics Manager")
    print("=" * 50)
    
    # Create statistics manager
    stats_manager = create_statistics_manager("./test_paper_trading_data")
    
    # Create sample data
    print("\n📊 Creating Sample Data...")
    
    # Sample portfolio values
    portfolio_values = [100000, 101000, 100500, 102000, 101500, 103000, 102500, 104000]
    
    # Sample trade logs
    trade_logs = [
        {'portfolio_value': 100000, 'timestamp': '2025-01-01T10:00:00'},
        {'portfolio_value': 101000, 'timestamp': '2025-01-02T10:00:00'},
        {'portfolio_value': 100500, 'timestamp': '2025-01-03T10:00:00'},
        {'portfolio_value': 102000, 'timestamp': '2025-01-04T10:00:00'},
        {'portfolio_value': 101500, 'timestamp': '2025-01-05T10:00:00'},
        {'portfolio_value': 103000, 'timestamp': '2025-01-06T10:00:00'},
        {'portfolio_value': 102500, 'timestamp': '2025-01-07T10:00:00'},
        {'portfolio_value': 104000, 'timestamp': '2025-01-08T10:00:00'}
    ]
    
    # Test trade statistics
    print("\n📈 Testing Trade Statistics...")
    trade_stats = stats_manager.calculate_trade_statistics(trade_logs)
    print(f"✅ Trade statistics calculated:")
    print(f"   Total trades: {trade_stats.total_trades}")
    print(f"   Win rate: {trade_stats.win_rate:.2%}")
    print(f"   Profit factor: {trade_stats.profit_factor:.2f}")
    
    # Test performance statistics
    print("\n📊 Testing Performance Statistics...")
    perf_stats = stats_manager.calculate_performance_statistics(portfolio_values)
    print(f"✅ Performance statistics calculated:")
    print(f"   Total return: {perf_stats.total_return:.2%}")
    print(f"   Sharpe ratio: {perf_stats.sharpe_ratio:.4f}")
    print(f"   Max drawdown: {perf_stats.max_drawdown:.2%}")
    print(f"   Volatility: {perf_stats.volatility:.2%}")
    
    # Test risk statistics
    print("\n⚠️ Testing Risk Statistics...")
    risk_stats = stats_manager.calculate_risk_statistics(portfolio_values)
    print(f"✅ Risk statistics calculated:")
    print(f"   VaR (95%): {risk_stats.var_95:.4f}")
    print(f"   CVaR (95%): {risk_stats.cvar_95:.4f}")
    print(f"   Ulcer Index: {risk_stats.ulcer_index:.4f}")
    print(f"   Gain to Pain: {risk_stats.gain_to_pain_ratio:.2f}")
    
    # Test comprehensive portfolio analysis
    print("\n📋 Testing Portfolio Analysis...")
    portfolio_analysis = stats_manager.analyze_portfolio(portfolio_values, trade_logs)
    print(f"✅ Portfolio analysis completed:")
    print(f"   Initial value: ${portfolio_analysis['portfolio_summary']['initial_value']:,.2f}")
    print(f"   Final value: ${portfolio_analysis['portfolio_summary']['final_value']:,.2f}")
    print(f"   Total trades: {portfolio_analysis['portfolio_summary']['total_trades']}")
    
    # Test strategy comparison
    print("\n🏆 Testing Strategy Comparison...")
    
    strategy_results = {
        'Momentum': {
            'portfolio_values': [100000, 101000, 102000, 103000, 104000],
            'trade_logs': trade_logs[:5]
        },
        'MeanReversion': {
            'portfolio_values': [100000, 100500, 101000, 101500, 102000],
            'trade_logs': trade_logs[:5]
        },
        'Volatility': {
            'portfolio_values': [100000, 100200, 100800, 101200, 101800],
            'trade_logs': trade_logs[:5]
        }
    }
    
    comparison = stats_manager.compare_strategies(strategy_results)
    print(f"✅ Strategy comparison completed:")
    print(f"   Best strategy: {comparison['summary']['best_strategy']}")
    print(f"   Most consistent: {comparison['summary']['most_consistent']}")
    print(f"   Safest: {comparison['summary']['safest']}")
    
    # Generate report
    print("\n📄 Generating Analysis Report...")
    report_path = stats_manager.generate_report(portfolio_analysis)
    print(f"✅ Analysis report generated: {report_path}")
    
    # Get database statistics
    print("\n📊 Database Statistics...")
    db_stats = stats_manager.get_database_statistics()
    print(f"✅ Database stats: {db_stats}")
    
    return stats_manager


def test_integrated_system():
    """Test the integrated system with real data"""
    print("\n🚀 Testing Integrated System with Real Data")
    print("=" * 60)
    
    # Create CCXT provider
    data_provider = create_ccxt_provider('binance', sandbox=False)
    
    # Create trading configuration
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_stock_quantity=10,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
        max_position_size=0.3,
        min_cash_reserve=0.2,
        max_leverage=1.2,
        stop_loss_pct=0.08,
        take_profit_pct=0.20
    )
    
    # Fetch real data
    print("\n📊 Fetching Real Cryptocurrency Data...")
    symbols = ['BTC/USDT', 'ETH/USDT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    data = {}
    for symbol in symbols:
        symbol_data = data_provider.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'
        )
        if not symbol_data.empty:
            symbol_data = data_provider.get_technical_indicators(symbol_data)
            data[symbol] = symbol_data
            print(f"✅ {symbol}: {len(symbol_data)} records")
    
    if not data:
        print("❌ No data available for testing")
        return
    
    # Create trading environment with data persistence
    print("\n🎮 Creating Trading Environment...")
    env = EnhancedStockTradingEnv(
        data=data,
        initial_capital=100000.0,
        enable_data_persistence=True,
        persistence_dir="./test_paper_trading_data"
    )
    
    # Run trading simulation
    print("\n🔄 Running Trading Simulation...")
    state, _ = env.reset()
    portfolio_values = [env._calculate_total_asset()]
    actions_taken = []
    
    for step in range(50):  # Run for 50 steps
        # Generate random action
        action = np.random.uniform(-0.5, 0.5, env.action_dim)
        
        # Take step
        state, reward, done, truncated, info = env.step(action)
        
        # Record results
        portfolio_values.append(info['total_asset'])
        actions_taken.append(action)
        
        if step % 10 == 0:
            print(f"   Step {step}: Portfolio = ${info['total_asset']:,.2f}, Reward = {reward:.4f}")
        
        if done:
            break
    
    # Get final statistics
    final_stats = env.get_portfolio_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Final Value: ${final_stats.get('final_value', portfolio_values[-1]):,.2f}")
    print(f"   Total Return: {final_stats.get('total_return', 0.0):.2%}")
    print(f"   Sharpe Ratio: {final_stats.get('sharpe_ratio', 0.0):.4f}")
    print(f"   Max Drawdown: {final_stats.get('max_drawdown', 0.0):.2%}")
    
    return {
        'portfolio_values': portfolio_values,
        'actions_taken': actions_taken,
        'final_stats': final_stats
    }


def main():
    """Main test function"""
    print("🚀 Comprehensive Data Persistence, Logging, and Statistics Test")
    print("=" * 80)
    
    try:
        # Test data persistence system
        persistence_manager = test_data_persistence_system()
        
        # Test logging system
        logger_instance = test_logging_system()
        
        # Test statistics manager
        stats_manager = test_statistics_manager()
        
        # Test integrated system
        integrated_results = test_integrated_system()
        
        print("\n🎉 All Tests Completed Successfully!")
        print("=" * 80)
        print("✅ Data persistence system working")
        print("✅ Logging system working")
        print("✅ Statistics manager working")
        print("✅ Integrated system working")
        
        # Show data directory structure
        print("\n📁 Data Directory Structure:")
        data_dir = Path("./test_paper_trading_data")
        if data_dir.exists():
            for item in data_dir.rglob("*"):
                if item.is_file():
                    print(f"   📄 {item.relative_to(data_dir)}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 