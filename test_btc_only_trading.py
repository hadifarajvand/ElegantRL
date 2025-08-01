"""
BTC-Only Paper Trading System Test
Comprehensive test focusing on Bitcoin trading with full data persistence and logging
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
from dataclasses import asdict

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


def run_btc_only_trading():
    """Run comprehensive BTC-only trading simulation"""
    print("🚀 BTC-Only Paper Trading System")
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
    
    # Create statistics manager
    stats_manager = create_statistics_manager("./paper_trading/data_cache")
    
    # Log system initialization
    logger_instance.log_system_event("BTC trading system initialized", {
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
    
    # Fetch BTC data
    print("\n📈 Fetching BTC/USDT Data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    btc_data = data_provider.get_historical_data(
        symbol='BTC/USDT',
        start_date=start_date,
        end_date=end_date,
        timeframe='1d'
    )
    
    if btc_data.empty:
        print("❌ No BTC data available")
        return
    
    # Add technical indicators
    btc_data = data_provider.get_technical_indicators(btc_data)
    print(f"✅ BTC/USDT: {len(btc_data)} records with {len(btc_data.columns)} features")
    
    # Log data fetching
    logger_instance.log_data_event("BTC data fetched", {
        'symbol': 'BTC/USDT',
        'records': len(btc_data),
        'features': len(btc_data.columns),
        'date_range': f"{start_date} to {end_date}"
    })
    
    # Save market data
    persistence_manager.save_market_data('BTC/USDT', btc_data, start_date, end_date)
    
    # Create trading configuration
    print("\n⚙️ Creating Trading Configuration...")
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_stock_quantity=5,  # Smaller position size for BTC
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
        max_position_size=0.3,
        min_cash_reserve=0.2,
        max_leverage=1.2,
        stop_loss_pct=0.08,
        take_profit_pct=0.20
    )
    
    # Save configuration
    persistence_manager.save_config({
        'trading': {
            'initial_capital': trading_config.initial_capital,
            'max_stock_quantity': trading_config.max_stock_quantity,
            'transaction_cost_pct': trading_config.transaction_cost_pct,
            'slippage_pct': trading_config.slippage_pct,
            'max_position_size': trading_config.max_position_size,
            'min_cash_reserve': trading_config.min_cash_reserve,
            'max_leverage': trading_config.max_leverage,
            'stop_loss_pct': trading_config.stop_loss_pct,
            'take_profit_pct': trading_config.take_profit_pct
        },
        'symbol': 'BTC/USDT',
        'date_range': f"{start_date} to {end_date}"
    }, "btc_trading_config")
    
    # Create trading environment
    print("\n🎮 Creating BTC Trading Environment...")
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
    
    print(f"✅ Environment created - State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # Run trading simulation
    print("\n🔄 Running BTC Trading Simulation...")
    state, _ = env.reset()
    initial_portfolio = env._calculate_total_asset()
    print(f"💰 Initial portfolio value: ${initial_portfolio:,.2f}")
    
    portfolio_values = [initial_portfolio]
    actions_taken = []
    trade_summary = {
        'total_trades': 0,
        'buy_trades': 0,
        'sell_trades': 0,
        'total_volume': 0.0,
        'total_cost': 0.0
    }
    
    # Run simulation for 100 steps
    for step in range(100):
        # Generate action (random for demonstration)
        action = np.random.uniform(-0.3, 0.3, env.action_dim)  # Conservative actions
        
        # Take step
        state, reward, done, truncated, info = env.step(action)
        
        # Record results
        portfolio_values.append(info['total_asset'])
        actions_taken.append(action)
        
        # Update trade summary
        if step > 0:
            portfolio_change = info['total_asset'] - portfolio_values[-2]
            if abs(portfolio_change) > 1.0:  # Significant change indicates trade
                trade_summary['total_trades'] += 1
                if portfolio_change > 0:
                    trade_summary['buy_trades'] += 1
                else:
                    trade_summary['sell_trades'] += 1
                trade_summary['total_volume'] += abs(portfolio_change)
                trade_summary['total_cost'] += abs(portfolio_change) * 0.001  # Transaction cost
        
        # Log progress
        if step % 20 == 0:
            current_return = (info['total_asset'] - initial_portfolio) / initial_portfolio
            print(f"   Step {step:3d}: Portfolio = ${info['total_asset']:,.2f}, "
                  f"Return = {current_return:+.2%}, Reward = {reward:+.4f}")
            
            # Log performance
            logger_instance.log_performance({
                'step': step,
                'portfolio_value': info['total_asset'],
                'return': current_return,
                'reward': reward,
                'cash': info['cash'],
                'positions': info['positions'].tolist()
            })
        
        if done:
            break
    
    # Get final statistics
    final_stats = env.get_portfolio_stats()
    final_portfolio = portfolio_values[-1]
    total_return = (final_portfolio - initial_portfolio) / initial_portfolio
    
    print(f"\n📊 BTC Trading Results:")
    print(f"   Initial Portfolio: ${initial_portfolio:,.2f}")
    print(f"   Final Portfolio:   ${final_portfolio:,.2f}")
    print(f"   Total Return:      {total_return:+.2%}")
    print(f"   Total Trades:      {trade_summary['total_trades']}")
    print(f"   Buy Trades:        {trade_summary['buy_trades']}")
    print(f"   Sell Trades:       {trade_summary['sell_trades']}")
    print(f"   Total Volume:      ${trade_summary['total_volume']:,.2f}")
    print(f"   Transaction Costs: ${trade_summary['total_cost']:,.2f}")
    
    # Calculate comprehensive statistics
    print("\n📈 Comprehensive Statistics Analysis...")
    
    # Trade statistics
    trade_stats = stats_manager.calculate_trade_statistics(env.trade_logs)
    print(f"   Trade Statistics:")
    print(f"     Total Trades: {trade_stats.total_trades}")
    print(f"     Win Rate: {trade_stats.win_rate:.2%}")
    print(f"     Profit Factor: {trade_stats.profit_factor:.2f}")
    print(f"     Average Win: ${trade_stats.average_win:,.2f}")
    print(f"     Average Loss: ${trade_stats.average_loss:,.2f}")
    
    # Performance statistics
    perf_stats = stats_manager.calculate_performance_statistics(portfolio_values)
    print(f"   Performance Statistics:")
    print(f"     Total Return: {perf_stats.total_return:.2%}")
    print(f"     Annualized Return: {perf_stats.annualized_return:.2%}")
    print(f"     Sharpe Ratio: {perf_stats.sharpe_ratio:.4f}")
    print(f"     Max Drawdown: {perf_stats.max_drawdown:.2%}")
    print(f"     Volatility: {perf_stats.volatility:.2%}")
    
    # Risk statistics
    risk_stats = stats_manager.calculate_risk_statistics(portfolio_values)
    print(f"   Risk Statistics:")
    print(f"     VaR (95%): {risk_stats.var_95:.4f}")
    print(f"     CVaR (95%): {risk_stats.cvar_95:.4f}")
    print(f"     Ulcer Index: {risk_stats.ulcer_index:.4f}")
    print(f"     Gain to Pain: {risk_stats.gain_to_pain_ratio:.2f}")
    
    # Portfolio analysis
    portfolio_analysis = stats_manager.analyze_portfolio(portfolio_values, env.trade_logs)
    
    # Save performance metrics
    performance_metrics = {
        'total_return': perf_stats.total_return,
        'sharpe_ratio': perf_stats.sharpe_ratio,
        'max_drawdown': perf_stats.max_drawdown,
        'volatility': perf_stats.volatility,
        'win_rate': trade_stats.win_rate,
        'total_trades': trade_stats.total_trades,
        'final_value': final_portfolio,
        'trade_summary': trade_summary
    }
    
    persistence_manager.save_performance_metrics(performance_metrics, "btc_trading")
    
    # Generate comprehensive report
    report_data = {
        'metadata': {
            'symbol': 'BTC/USDT',
            'date_range': f"{start_date} to {end_date}",
            'initial_capital': initial_portfolio,
            'final_capital': final_portfolio,
            'total_return': total_return,
            'simulation_steps': len(portfolio_values) - 1
        },
        'trade_statistics': asdict(trade_stats),
        'performance_statistics': asdict(perf_stats),
        'risk_statistics': asdict(risk_stats),
        'portfolio_analysis': portfolio_analysis,
        'trade_summary': trade_summary,
        'portfolio_values': portfolio_values,
        'actions_taken': [action.tolist() for action in actions_taken]
    }
    
    report_path = stats_manager.generate_report(report_data)
    print(f"\n📄 Comprehensive report generated: {report_path}")
    
    # Log final results
    logger_instance.log_performance({
        'final_portfolio_value': final_portfolio,
        'total_return': total_return,
        'sharpe_ratio': perf_stats.sharpe_ratio,
        'max_drawdown': perf_stats.max_drawdown,
        'total_trades': trade_stats.total_trades,
        'win_rate': trade_stats.win_rate
    })
    
    # Show data directory structure
    print(f"\n📁 BTC Trading Data Directory Structure:")
    data_dir = Path("./paper_trading/data_cache")
    if data_dir.exists():
        for item in data_dir.rglob("*"):
            if item.is_file():
                print(f"   📄 {item.relative_to(data_dir)}")
    
    print(f"\n🎉 BTC-Only Trading Simulation Completed!")
    print(f"✅ Comprehensive data persistence and logging implemented")
    print(f"✅ Real BTC data from Binance processed")
    print(f"✅ Advanced statistics and analysis performed")
    print(f"✅ Complete reports and logs generated")
    
    return {
        'initial_portfolio': initial_portfolio,
        'final_portfolio': final_portfolio,
        'total_return': total_return,
        'trade_stats': trade_stats,
        'perf_stats': perf_stats,
        'risk_stats': risk_stats,
        'portfolio_values': portfolio_values,
        'trade_summary': trade_summary
    }


if __name__ == "__main__":
    try:
        results = run_btc_only_trading()
        print(f"\n🏆 Final Summary:")
        print(f"   BTC Trading Return: {results['total_return']:+.2%}")
        print(f"   Portfolio Growth: ${results['initial_portfolio']:,.2f} → ${results['final_portfolio']:,.2f}")
        print(f"   Total Trades: {results['trade_summary']['total_trades']}")
        print(f"   Sharpe Ratio: {results['perf_stats'].sharpe_ratio:.4f}")
        print(f"   Max Drawdown: {results['perf_stats'].max_drawdown:.2%}")
    except Exception as e:
        print(f"\n❌ BTC trading test failed with error: {e}")
        import traceback
        traceback.print_exc() 