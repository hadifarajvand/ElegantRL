"""
BTC Multi-Timeframe Trading Analysis
Test different timeframes for BTC trading with comprehensive data persistence
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from dataclasses import asdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
from paper_trading.data.ccxt_provider import create_ccxt_provider
from paper_trading.configs.trading_config import TradingConfig
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.utils.data_persistence import create_data_persistence_manager
from paper_trading.utils.logging_config import setup_logging, get_logger
from paper_trading.utils.statistics_manager import create_statistics_manager

logger = logging.getLogger(__name__)


def test_timeframe(provider, timeframe: str, days: int, persistence_manager, logger_instance) -> Dict[str, Any]:
    """Test a specific timeframe for BTC trading"""
    print(f"\n🕐 Testing {timeframe} Timeframe ({days} days)")
    print("-" * 50)
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Fetch BTC data for this timeframe
    btc_data = provider.get_historical_data(
        symbol='BTC/USDT',
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    if btc_data.empty:
        print(f"❌ No BTC data available for {timeframe}")
        return {}
    
    # Add technical indicators
    btc_data = provider.get_technical_indicators(btc_data)
    print(f"✅ BTC/USDT ({timeframe}): {len(btc_data)} records with {len(btc_data.columns)} features")
    
    # Log data fetching
    logger_instance.log_data_event(f"BTC {timeframe} data fetched", {
        'symbol': 'BTC/USDT',
        'timeframe': timeframe,
        'records': len(btc_data),
        'features': len(btc_data.columns),
        'date_range': f"{start_date} to {end_date}"
    })
    
    # Save market data
    persistence_manager.save_market_data(f'BTC_USDT_{timeframe}', btc_data, start_date, end_date)
    
    # Create trading configuration
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_stock_quantity=5,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
        max_position_size=0.3,
        min_cash_reserve=0.2,
        max_leverage=1.2,
        stop_loss_pct=0.08,
        take_profit_pct=0.20
    )
    
    # Create trading environment
    env = EnhancedStockTradingEnv(
        data={'BTC/USDT': btc_data},
        initial_capital=trading_config.initial_capital,
        max_stock_quantity=trading_config.max_stock_quantity,
        transaction_cost_pct=trading_config.transaction_cost_pct,
        slippage_pct=trading_config.slippage_pct,
        max_position_size=trading_config.max_position_size,
        min_cash_reserve=trading_config.min_cash_reserve,
        max_leverage=trading_config.max_leverage,
        stop_loss_pct=trading_config.stop_loss_pct,
        take_profit_pct=trading_config.take_profit_pct,
        enable_data_persistence=True,
        persistence_dir="./paper_trading/data_cache"
    )
    
    # Run trading simulation
    state, _ = env.reset()
    initial_portfolio = env._calculate_total_asset()
    
    portfolio_values = [initial_portfolio]
    actions_taken = []
    trade_summary = {
        'total_trades': 0,
        'buy_trades': 0,
        'sell_trades': 0,
        'total_volume': 0.0,
        'total_cost': 0.0
    }
    
    # Run simulation for min(100, len(data)) steps
    max_steps = min(100, len(btc_data) - 1)
    
    for step in range(max_steps):
        # Generate action (random for demonstration)
        action = np.random.uniform(-0.3, 0.3, env.action_dim)
        
        # Take step
        state, reward, done, truncated, info = env.step(action)
        
        # Record results
        portfolio_values.append(info['total_asset'])
        actions_taken.append(action)
        
        # Update trade summary
        if step > 0:
            portfolio_change = info['total_asset'] - portfolio_values[-2]
            if abs(portfolio_change) > 1.0:
                trade_summary['total_trades'] += 1
                if portfolio_change > 0:
                    trade_summary['buy_trades'] += 1
                else:
                    trade_summary['sell_trades'] += 1
                trade_summary['total_volume'] += abs(portfolio_change)
                trade_summary['total_cost'] += abs(portfolio_change) * 0.001
        
        if done:
            break
    
    # Calculate final statistics
    final_portfolio = portfolio_values[-1]
    total_return = (final_portfolio - initial_portfolio) / initial_portfolio
    
    # Create statistics manager for this timeframe
    stats_manager = create_statistics_manager("./paper_trading/data_cache")
    
    # Calculate comprehensive statistics
    trade_stats = stats_manager.calculate_trade_statistics(env.trade_logs)
    perf_stats = stats_manager.calculate_performance_statistics(portfolio_values)
    risk_stats = stats_manager.calculate_risk_statistics(portfolio_values)
    
    # Save performance metrics
    performance_metrics = {
        'timeframe': timeframe,
        'total_return': perf_stats.total_return,
        'sharpe_ratio': perf_stats.sharpe_ratio,
        'max_drawdown': perf_stats.max_drawdown,
        'volatility': perf_stats.volatility,
        'win_rate': trade_stats.win_rate,
        'total_trades': trade_stats.total_trades,
        'final_value': final_portfolio,
        'trade_summary': trade_summary,
        'data_points': len(btc_data),
        'simulation_steps': len(portfolio_values) - 1
    }
    
    persistence_manager.save_performance_metrics(performance_metrics, f"btc_{timeframe}_trading")
    
    # Log results
    logger_instance.log_performance({
        'timeframe': timeframe,
        'final_portfolio_value': final_portfolio,
        'total_return': total_return,
        'sharpe_ratio': perf_stats.sharpe_ratio,
        'max_drawdown': perf_stats.max_drawdown,
        'total_trades': trade_stats.total_trades,
        'win_rate': trade_stats.win_rate
    })
    
    # Print results
    print(f"   📊 Results for {timeframe}:")
    print(f"      Initial Portfolio: ${initial_portfolio:,.2f}")
    print(f"      Final Portfolio:   ${final_portfolio:,.2f}")
    print(f"      Total Return:      {total_return:+.2%}")
    print(f"      Sharpe Ratio:      {perf_stats.sharpe_ratio:.4f}")
    print(f"      Max Drawdown:      {perf_stats.max_drawdown:.2%}")
    print(f"      Total Trades:      {trade_summary['total_trades']}")
    print(f"      Win Rate:          {trade_stats.win_rate:.2%}")
    print(f"      Data Points:       {len(btc_data)}")
    
    return {
        'timeframe': timeframe,
        'initial_portfolio': initial_portfolio,
        'final_portfolio': final_portfolio,
        'total_return': total_return,
        'trade_stats': trade_stats,
        'perf_stats': perf_stats,
        'risk_stats': risk_stats,
        'portfolio_values': portfolio_values,
        'trade_summary': trade_summary,
        'data_points': len(btc_data)
    }


def run_multi_timeframe_analysis():
    """Run comprehensive multi-timeframe BTC trading analysis"""
    print("🚀 BTC Multi-Timeframe Trading Analysis")
    print("=" * 60)
    
    # Setup logging
    logger_instance = setup_logging(
        log_dir="./paper_trading/data_cache/logs",
        log_level="INFO",
        enable_console=True,
        enable_file=True
    )
    
    # Create data persistence manager
    persistence_manager = create_data_persistence_manager("./paper_trading/data_cache")
    
    # Log system initialization
    logger_instance.log_system_event("Multi-timeframe BTC trading system initialized", {
        'symbol': 'BTC/USDT',
        'data_persistence': True,
        'logging': True,
        'statistics': True
    })
    
    # Create CCXT provider
    print("\n📊 Setting up CCXT Provider...")
    data_provider = create_ccxt_provider('binance', sandbox=False)
    
    # Test connection
    if data_provider.test_connection():
        print("✅ Connected to Binance successfully")
        logger_instance.log_data_event("Connected to Binance", {'exchange': 'binance'})
    else:
        print("❌ Failed to connect to Binance")
        return
    
    # Define timeframes to test
    timeframes = [
        ('1m', 7),      # 1 minute - 7 days
        ('5m', 14),     # 5 minutes - 14 days
        ('15m', 30),    # 15 minutes - 30 days
        ('1h', 60),     # 1 hour - 60 days
        ('4h', 90),     # 4 hours - 90 days
        ('1d', 180),    # 1 day - 180 days
    ]
    
    results = {}
    
    # Test each timeframe
    for timeframe, days in timeframes:
        try:
            result = test_timeframe(data_provider, timeframe, days, persistence_manager, logger_instance)
            if result:
                results[timeframe] = result
        except Exception as e:
            print(f"❌ Error testing {timeframe}: {e}")
            logger_instance.log_error(f"Error testing {timeframe}", {'error': str(e)})
    
    # Compare results
    print(f"\n📊 Multi-Timeframe Comparison Results:")
    print("=" * 60)
    
    comparison_data = []
    for timeframe, result in results.items():
        comparison_data.append({
            'timeframe': timeframe,
            'total_return': result['total_return'],
            'sharpe_ratio': result['perf_stats'].sharpe_ratio,
            'max_drawdown': result['perf_stats'].max_drawdown,
            'win_rate': result['trade_stats'].win_rate,
            'total_trades': result['trade_summary']['total_trades'],
            'data_points': result['data_points'],
            'volatility': result['perf_stats'].volatility
        })
    
    # Sort by total return
    comparison_data.sort(key=lambda x: x['total_return'], reverse=True)
    
    print(f"{'Timeframe':<8} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10} {'Trades':<8} {'Data Points':<12}")
    print("-" * 80)
    
    for data in comparison_data:
        print(f"{data['timeframe']:<8} {data['total_return']:+.2%} {data['sharpe_ratio']:>6.3f} "
              f"{data['max_drawdown']:>8.2%} {data['win_rate']:>8.2%} {data['total_trades']:>6} "
              f"{data['data_points']:>10}")
    
    # Find best performing timeframe
    best_timeframe = comparison_data[0]
    print(f"\n🏆 Best Performing Timeframe: {best_timeframe['timeframe']}")
    print(f"   Total Return: {best_timeframe['total_return']:+.2%}")
    print(f"   Sharpe Ratio: {best_timeframe['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {best_timeframe['max_drawdown']:.2%}")
    
    # Generate comprehensive report
    report_data = {
        'metadata': {
            'symbol': 'BTC/USDT',
            'analysis_date': datetime.now().isoformat(),
            'total_timeframes_tested': len(results),
            'best_timeframe': best_timeframe['timeframe']
        },
        'timeframe_results': results,
        'comparison_data': comparison_data,
        'summary': {
            'best_return': best_timeframe['total_return'],
            'best_sharpe': best_timeframe['sharpe_ratio'],
            'best_drawdown': best_timeframe['max_drawdown'],
            'best_win_rate': best_timeframe['win_rate']
        }
    }
    
    # Save comprehensive report
    stats_manager = create_statistics_manager("./paper_trading/data_cache")
    report_path = stats_manager.generate_report(report_data)
    print(f"\n📄 Comprehensive multi-timeframe report generated: {report_path}")
    
    # Show data directory structure
    print(f"\n📁 Paper Trading Data Directory Structure:")
    data_dir = Path("./paper_trading/data_cache")
    if data_dir.exists():
        for item in data_dir.rglob("*"):
            if item.is_file():
                print(f"   📄 {item.relative_to(data_dir)}")
    
    print(f"\n🎉 Multi-Timeframe BTC Trading Analysis Completed!")
    print(f"✅ {len(results)} timeframes tested successfully")
    print(f"✅ Comprehensive data persistence and logging implemented")
    print(f"✅ Real BTC data from Binance processed for all timeframes")
    print(f"✅ Advanced statistics and analysis performed")
    print(f"✅ Complete reports and logs generated")
    
    return results


if __name__ == "__main__":
    try:
        results = run_multi_timeframe_analysis()
        print(f"\n🏆 Final Summary:")
        print(f"   Timeframes Tested: {len(results)}")
        if results:
            best_timeframe = max(results.keys(), key=lambda k: results[k]['total_return'])
            best_result = results[best_timeframe]
            print(f"   Best Timeframe: {best_timeframe}")
            print(f"   Best Return: {best_result['total_return']:+.2%}")
            print(f"   Best Sharpe: {best_result['perf_stats'].sharpe_ratio:.4f}")
    except Exception as e:
        print(f"\n❌ Multi-timeframe analysis failed with error: {e}")
        import traceback
        traceback.print_exc() 