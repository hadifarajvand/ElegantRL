"""
Enhanced Cryptocurrency Backtesting System
Comprehensive backtesting for cryptocurrency trading strategies
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our components
from paper_trading.data.ccxt_provider import create_ccxt_provider
from paper_trading.configs.trading_config import TradingConfig
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.strategies.crypto_strategies import create_crypto_strategies
from paper_trading.utils.metrics import PerformanceMetrics

# Import data persistence
try:
    from paper_trading.utils.data_persistence import create_data_persistence_manager
    DATA_PERSISTENCE_AVAILABLE = True
except ImportError:
    DATA_PERSISTENCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    Enhanced backtesting engine for cryptocurrency trading
    """
    
    def __init__(self, 
                 data_provider,
                 trading_config: TradingConfig,
                 initial_capital: float = 100000.0,
                 enable_data_persistence: bool = True,
                 persistence_dir: str = "./paper_trading_data"):
        
        self.data_provider = data_provider
        self.trading_config = trading_config
        self.initial_capital = initial_capital
        
        # Data persistence
        self.enable_data_persistence = enable_data_persistence and DATA_PERSISTENCE_AVAILABLE
        if self.enable_data_persistence:
            self.persistence_manager = create_data_persistence_manager(persistence_dir)
        else:
            self.persistence_manager = None
        
        # Backtesting results
        self.results = {}
        self.performance_metrics = {}
        self.trade_logs = []
        
        # Strategy manager
        self.strategy_manager = create_crypto_strategies()
        
        logger.info("Crypto backtesting engine initialized")
    
    def run_backtest(self, 
                    symbols: List[str],
                    start_date: str,
                    end_date: str,
                    strategies: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest on multiple strategies
        """
        print(f"🚀 Running Cryptocurrency Backtest")
        print(f"📊 Symbols: {symbols}")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 60)
        
        # Fetch data
        data = self._fetch_data(symbols, start_date, end_date)
        if not data:
            print("❌ No data available for backtesting")
            return {}
        
        # Use all strategies if none specified
        if strategies is None:
            strategies = list(self.strategy_manager.strategies.keys())
        
        results = {}
        
        for strategy_name in strategies:
            print(f"\n📈 Testing strategy: {strategy_name}")
            
            # Set active strategy
            self.strategy_manager.set_active_strategy(strategy_name)
            
            # Run backtest for this strategy
            strategy_results = self._run_strategy_backtest(data, strategy_name)
            results[strategy_name] = strategy_results
            
            # Print summary
            if strategy_results:
                metrics = strategy_results['metrics']
                print(f"   ✅ Total Return: {metrics['total_return']:.2%}")
                print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"   💰 Final Value: ${metrics['final_value']:,.2f}")
        
        self.results = results
        return results
    
    def _fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch and prepare data for backtesting"""
        print("📊 Fetching cryptocurrency data...")
        
        data = {}
        for symbol in symbols:
            print(f"   📈 Fetching {symbol}...")
            
            # Try to load from cache first
            if self.enable_data_persistence and self.persistence_manager:
                cached_data = self.persistence_manager.load_market_data(symbol, start_date, end_date)
                if cached_data is not None:
                    data[symbol] = cached_data
                    print(f"   ✅ {symbol}: {len(cached_data)} records (cached)")
                    continue
            
            # Fetch from provider
            symbol_data = self.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1d'
            )
            
            if not symbol_data.empty:
                # Add technical indicators
                symbol_data = self.data_provider.get_technical_indicators(symbol_data)
                data[symbol] = symbol_data
                
                # Save to cache
                if self.enable_data_persistence and self.persistence_manager:
                    self.persistence_manager.save_market_data(symbol, symbol_data, start_date, end_date)
                
                print(f"   ✅ {symbol}: {len(symbol_data)} records")
            else:
                print(f"   ❌ No data for {symbol}")
        
        return data
    
    def _run_strategy_backtest(self, data: Dict[str, pd.DataFrame], strategy_name: str) -> Dict[str, Any]:
        """Run backtest for a specific strategy"""
        try:
            # Create trading environment
            env = EnhancedStockTradingEnv(
                data=data,
                initial_capital=self.initial_capital,
                max_stock_quantity=self.trading_config.max_stock_quantity,
                transaction_cost_pct=self.trading_config.transaction_cost_pct,
                slippage_pct=self.trading_config.slippage_pct,
                max_position_size=self.trading_config.max_position_size,
                min_cash_reserve=self.trading_config.min_cash_reserve,
                max_leverage=self.trading_config.max_leverage,
                stop_loss_pct=self.trading_config.stop_loss_pct,
                take_profit_pct=self.trading_config.take_profit_pct
            )
            
            # Get strategy signals
            signals = self.strategy_manager.get_signals(data)
            
            # Run simulation
            state, _ = env.reset()
            portfolio_values = [self.initial_capital]
            actions_taken = []
            trade_log = []
            
            step = 0
            while step < env.max_step:
                # Get current signals for this step
                current_signals = {}
                for symbol, signal_array in signals.items():
                    if step < len(signal_array):
                        current_signals[symbol] = signal_array[step]
                
                # Convert signals to actions
                action = self._signals_to_action(current_signals, env.action_dim)
                
                # Take step
                state, reward, done, truncated, info = env.step(action)
                
                # Record results
                portfolio_values.append(info['total_asset'])
                actions_taken.append(action)
                
                # Log trades
                if step > 0:
                    trade_log.append({
                        'step': step,
                        'portfolio_value': info['total_asset'],
                        'cash': info['cash'],
                        'positions': info['positions'].tolist(),
                        'reward': reward,
                        'action': action.tolist()
                    })
                
                if done or truncated:
                    break
                
                step += 1
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio_values, trade_log)
            
            return {
                'portfolio_values': portfolio_values,
                'actions_taken': actions_taken,
                'trade_log': trade_log,
                'metrics': metrics,
                'strategy_name': strategy_name
            }
            
        except Exception as e:
            logger.error(f"Error in strategy backtest: {e}")
            return {}
    
    def _signals_to_action(self, signals: Dict[str, float], action_dim: int) -> np.ndarray:
        """Convert strategy signals to trading actions"""
        action = np.zeros(action_dim)
        
        # Map signals to actions (simplified)
        for i, (symbol, signal) in enumerate(signals.items()):
            if i < action_dim:
                # Scale signal to action range [-1, 1]
                action[i] = np.clip(signal, -1, 1)
        
        return action
    
    def _calculate_performance_metrics(self, portfolio_values: List[float], trade_log: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not portfolio_values:
            return {}
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        final_value = portfolio_values[-1]
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Additional metrics
        positive_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Trading activity
        total_trades = len(trade_log)
        avg_trade_value = np.mean([log['portfolio_value'] for log in trade_log]) if trade_log else 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade_value': avg_trade_value,
            'total_days': total_days,
            'positive_days': positive_days
        }
    
    def generate_report(self, output_dir: str = "./backtest_results"):
        """Generate comprehensive backtest report"""
        if not self.results:
            print("❌ No backtest results available")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Generating Backtest Report")
        print("=" * 60)
        
        # Generate performance comparison
        self._generate_performance_comparison(output_dir)
        
        # Generate strategy reports
        self._generate_strategy_reports(output_dir)
        
        # Generate trade analysis
        self._generate_trade_analysis(output_dir)
        
        # Save results
        self._save_results(output_dir)
        
        print(f"✅ Report generated in {output_dir}")
    
    def _generate_performance_comparison(self, output_dir: str):
        """Generate performance comparison chart"""
        strategies = list(self.results.keys())
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cryptocurrency Trading Strategy Performance Comparison', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            values = [self.results[strategy]['metrics'].get(metric, 0) for strategy in strategies]
            
            bars = ax.bar(strategies, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace("_", " ").title())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_strategy_reports(self, output_dir: str):
        """Generate individual strategy reports"""
        for strategy_name, result in self.results.items():
            if not result:
                continue
            
            # Portfolio value plot
            portfolio_values = result['portfolio_values']
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(portfolio_values, linewidth=2, color='#1f77b4')
            plt.title(f'{strategy_name} - Portfolio Value Over Time')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            
            # Returns plot
            plt.subplot(2, 1, 2)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            plt.plot(returns, linewidth=1, color='#ff7f0e', alpha=0.7)
            plt.title(f'{strategy_name} - Daily Returns')
            plt.ylabel('Daily Return')
            plt.xlabel('Trading Day')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{strategy_name}_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_trade_analysis(self, output_dir: str):
        """Generate trade analysis"""
        for strategy_name, result in self.results.items():
            if not result or 'trade_log' not in result:
                continue
            
            trade_log = result['trade_log']
            if not trade_log:
                continue
            
            # Extract trade data
            steps = [log['step'] for log in trade_log]
            portfolio_values = [log['portfolio_value'] for log in trade_log]
            rewards = [log['reward'] for log in trade_log]
            
            # Create trade analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{strategy_name} - Trade Analysis', fontsize=16)
            
            # Portfolio value over time
            axes[0, 0].plot(steps, portfolio_values, linewidth=2, color='#1f77b4')
            axes[0, 0].set_title('Portfolio Value')
            axes[0, 0].set_ylabel('Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Reward distribution
            axes[0, 1].hist(rewards, bins=20, alpha=0.7, color='#ff7f0e')
            axes[0, 1].set_title('Reward Distribution')
            axes[0, 1].set_xlabel('Reward')
            axes[0, 1].set_ylabel('Frequency')
            
            # Cumulative rewards
            cumulative_rewards = np.cumsum(rewards)
            axes[1, 0].plot(steps, cumulative_rewards, linewidth=2, color='#2ca02c')
            axes[1, 0].set_title('Cumulative Rewards')
            axes[1, 0].set_ylabel('Cumulative Reward')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Action distribution
            actions = [log['action'] for log in trade_log]
            if actions and len(actions[0]) > 0:
                actions_array = np.array(actions)
                for i in range(min(3, actions_array.shape[1])):
                    axes[1, 1].hist(actions_array[:, i], bins=20, alpha=0.5, 
                                   label=f'Action {i+1}')
                axes[1, 1].set_title('Action Distribution')
                axes[1, 1].set_xlabel('Action Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{strategy_name}_trade_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_results(self, output_dir: str):
        """Save backtest results to files"""
        # Save summary results
        summary = {}
        for strategy_name, result in self.results.items():
            if result:
                # Convert numpy types to native Python types for JSON serialization
                metrics = result['metrics']
                summary[strategy_name] = {
                    key: float(value) if hasattr(value, 'item') else value
                    for key, value in metrics.items()
                }
        
        with open(f"{output_dir}/backtest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results (simplified to avoid serialization issues)
        detailed_results = {}
        for strategy_name, result in self.results.items():
            if result:
                detailed_results[strategy_name] = {
                    'strategy_name': result['strategy_name'],
                    'metrics': summary[strategy_name],
                    'portfolio_values_count': len(result.get('portfolio_values', [])),
                    'actions_taken_count': len(result.get('actions_taken', [])),
                    'trade_log_count': len(result.get('trade_log', []))
                }
        
        with open(f"{output_dir}/detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save to data persistence system if enabled
        if self.enable_data_persistence and self.persistence_manager:
            backtest_name = f"crypto_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.persistence_manager.save_backtest_results({
                'summary': summary,
                'detailed_results': detailed_results,
                'trade_logs': self.trade_logs,
                'portfolio_values': [result.get('portfolio_values', []) for result in self.results.values() if result]
            }, backtest_name)
            
            # Save performance metrics for each strategy
            for strategy_name, result in self.results.items():
                if result and 'metrics' in result:
                    self.persistence_manager.save_performance_metrics(
                        result['metrics'], 
                        f"crypto_backtest_{strategy_name}"
                    )
        
        print(f"📁 Results saved to {output_dir}")


def main():
    """Main backtesting function"""
    print("🚀 Cryptocurrency Backtesting System")
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
    
    # Create backtesting engine
    engine = CryptoBacktestEngine(
        data_provider=data_provider,
        trading_config=trading_config,
        initial_capital=100000.0
    )
    
    # Define backtest parameters
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Run backtest
    results = engine.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        strategies=['Momentum', 'MeanReversion', 'Volatility', 'MultiStrategy']
    )
    
    # Generate report
    engine.generate_report()
    
    print("\n🎉 Backtesting completed successfully!")
    print("✅ Comprehensive cryptocurrency backtesting performed")
    print("✅ Multiple strategies tested and compared")
    print("✅ Detailed reports generated")


if __name__ == "__main__":
    main() 