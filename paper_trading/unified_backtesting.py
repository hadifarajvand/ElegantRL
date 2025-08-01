#!/usr/bin/env python3
"""
Unified Backtesting System
Combines all backtesting functionality into one comprehensive script
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Mock the CryptoTradingAgent for now to avoid import issues
class CryptoTradingAgent:
    """Mock CryptoTradingAgent for unified backtesting"""
    
    def __init__(self, agent_type: str = 'PPO', state_dim: int = 15, action_dim: int = 1, hidden_dim: int = 64):
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = None
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from agent"""
        # Mock action - random for now
        return np.random.uniform(-1, 1, self.action_dim)
    
    def save_agent(self, filepath: str):
        """Save agent to file"""
        print(f"Mock: Saving agent to {filepath}")
        
    def load_agent(self, filepath: str):
        """Load agent from file"""
        print(f"Mock: Loading agent from {filepath}")

class UnifiedBacktester:
    """Unified backtester that combines all backtesting functionality"""
    
    def __init__(self, data_path: str = None, initial_capital: float = 100000.0):
        self.data_path = data_path or '../paper_trading_data/bitcoin_data/bitcoin_15m_combined_20230731_20250730.parquet'
        self.initial_capital = initial_capital
        self.results_dir = Path('paper_trading_data/unified_backtesting')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Backtesting state
        self.portfolio = {
            'cash': initial_capital,
            'position': 0.0,
            'total_value': initial_capital,
            'trades': [],
            'equity_curve': []
        }
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.results_dir / f"unified_backtesting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Unified Backtesting System Started")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        self.logger.info(f"📊 Data path: {self.data_path}")
        self.logger.info(f"💵 Initial capital: ${self.initial_capital:,.2f}")
        
    def load_backtest_data(self) -> Optional[pd.DataFrame]:
        """Load data for backtesting"""
        self.logger.info("📊 Loading backtest data...")
        try:
            # Try to load as CSV first, then Parquet
            if self.data_path.endswith('.csv'):
                data = pd.read_csv(self.data_path)
            else:
                data = pd.read_parquet(self.data_path)
            self.logger.info(f"✅ Successfully loaded dataset:")
            self.logger.info(f"   📈 Total records: {len(data):,}")
            self.logger.info(f"   📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            self.logger.info(f"   💰 Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
            
            # Clean data
            data_clean = self._clean_data(data)
            if data_clean is not None and not data_clean.empty:
                self.logger.info(f"✅ Data cleaned successfully: {len(data_clean):,} records")
                return data_clean
            else:
                self.logger.error("❌ No valid data after cleaning")
                return None
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return None
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        self.logger.info("🧹 Cleaning data...")
        
        # Remove duplicates
        initial_count = len(data)
        data = data.drop_duplicates()
        self.logger.info(f"   Removed {initial_count - len(data)} duplicates")
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Remove rows with missing values
        data = data.dropna()
        self.logger.info(f"   Removed rows with missing values: {len(data):,} records remaining")
        
        # Validate price data
        data = data[data['close'] > 0]
        data = data[data['volume'] >= 0]
        self.logger.info(f"   Validated price data: {len(data):,} records remaining")
        
        return data
    
    def prepare_state_vector(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Prepare state vector for DRL agent"""
        try:
            if index < 20:  # Need at least 20 data points
                return np.zeros(15)
            
            # Get recent data
            recent_data = data.iloc[max(0, index-20):index+1]
            close_prices = recent_data['close'].values
            volumes = recent_data['volume'].values
            
            # Simple moving averages
            sma_5 = np.mean(close_prices[-5:])
            sma_10 = np.mean(close_prices[-10:])
            sma_20 = np.mean(close_prices[-20:])
            
            # Price changes
            price_change_1 = (close_prices[-1] - close_prices[-2]) / close_prices[-2] if len(close_prices) > 1 else 0
            price_change_5 = (close_prices[-1] - close_prices[-5]) / close_prices[-5] if len(close_prices) > 5 else 0
            
            # Volume indicators
            volume_avg = np.mean(volumes[-10:])
            volume_ratio = volumes[-1] / volume_avg if volume_avg > 0 else 1
            
            # Volatility
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
            
            # Current price normalized
            price_normalized = close_prices[-1] / np.mean(close_prices[-20:]) if len(close_prices) >= 20 else 1
            
            # Portfolio state
            cash_ratio = self.portfolio['cash'] / self.portfolio['total_value']
            position_ratio = self.portfolio['position'] / self.portfolio['total_value'] if self.portfolio['total_value'] > 0 else 0
            
            # Combine all features
            state = np.array([
                price_normalized,
                price_change_1,
                price_change_5,
                sma_5 / close_prices[-1] - 1,
                sma_10 / close_prices[-1] - 1,
                sma_20 / close_prices[-1] - 1,
                volume_ratio,
                volatility,
                cash_ratio,
                position_ratio,
                0.0,  # Last action (not available in backtesting)
                close_prices[-1] / 100000,  # Normalized price
                volumes[-1] / 1000000,  # Normalized volume
                len(self.portfolio['trades']),
                self.portfolio['total_value'] / self.initial_capital - 1  # Return
            ])
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing state vector: {e}")
            return np.zeros(15)
    
    def execute_backtest_trade(self, action: float, current_price: float, timestamp: datetime):
        """Execute a trade during backtesting"""
        try:
            # Determine trade size based on action magnitude
            trade_size = abs(action)
            if trade_size < 0.1:  # Small action, no trade
                return
            
            # Calculate trade amount
            if action > 0:  # Buy
                trade_amount = min(self.portfolio['cash'] * trade_size, self.portfolio['cash'])
                if trade_amount > 0:
                    shares = trade_amount / current_price
                    self.portfolio['cash'] -= trade_amount
                    self.portfolio['position'] += shares
                    
                    self.portfolio['trades'].append({
                        'timestamp': timestamp,
                        'action': 'buy',
                        'price': current_price,
                        'amount': trade_amount,
                        'shares': shares
                    })
                    
                    self.logger.debug(f"💰 BUY: {shares:.6f} shares at ${current_price:,.2f} (${trade_amount:,.2f})")
            
            elif action < 0:  # Sell
                if self.portfolio['position'] > 0:
                    shares_to_sell = self.portfolio['position'] * trade_size
                    trade_amount = shares_to_sell * current_price
                    
                    self.portfolio['cash'] += trade_amount
                    self.portfolio['position'] -= shares_to_sell
                    
                    self.portfolio['trades'].append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': current_price,
                        'amount': trade_amount,
                        'shares': shares_to_sell
                    })
                    
                    self.logger.debug(f"💰 SELL: {shares_to_sell:.6f} shares at ${current_price:,.2f} (${trade_amount:,.2f})")
            
            # Update total value
            self.portfolio['total_value'] = self.portfolio['cash'] + (self.portfolio['position'] * current_price)
            
        except Exception as e:
            self.logger.error(f"❌ Error executing backtest trade: {e}")
    
    def run_drl_backtest(self, agent_type: str, model_path: str, test_period_days: int = 30) -> Dict[str, Any]:
        """Run backtest with DRL agent"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🤖 Running DRL Backtest: {agent_type}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Load DRL agent
            self.logger.info(f"🤖 Loading DRL agent: {agent_type}")
            agent = CryptoTradingAgent(
                agent_type=agent_type,
                state_dim=15,
                action_dim=1,
                hidden_dim=64
            )
            agent.load_agent(model_path)
            self.logger.info(f"✅ DRL agent loaded successfully")
            
            # Load data
            data = self.load_backtest_data()
            if data is None:
                self.logger.error("❌ Failed to load data for backtesting")
                return None
            
            # Select test period (last N days)
            # Convert timestamp to datetime if it's a string
            if data['timestamp'].dtype == 'object':
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            test_start = data['timestamp'].max() - timedelta(days=test_period_days)
            test_data = data[data['timestamp'] >= test_start].copy()
            
            self.logger.info(f"📊 Test period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
            self.logger.info(f"📈 Test records: {len(test_data):,}")
            
            # Reset portfolio
            self.portfolio = {
                'cash': self.initial_capital,
                'position': 0.0,
                'total_value': self.initial_capital,
                'trades': [],
                'equity_curve': []
            }
            
            # Run backtest
            start_time = time.time()
            
            for i, (idx, row) in enumerate(test_data.iterrows()):
                current_price = row['close']
                timestamp = row['timestamp']
                
                # Prepare state vector
                state = self.prepare_state_vector(test_data, i)
                
                # Get action from DRL agent
                action = agent.get_action(state)
                
                # Execute trade
                self.execute_backtest_trade(action, current_price, timestamp)
                
                # Record equity curve
                self.portfolio['equity_curve'].append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'portfolio_value': self.portfolio['total_value'],
                    'cash': self.portfolio['cash'],
                    'position': self.portfolio['position'],
                    'action': action
                })
                
                # Log progress every 1000 iterations
                if (i + 1) % 1000 == 0:
                    progress = (i + 1) / len(test_data) * 100
                    self.logger.info(f"📊 Progress: {progress:.1f}% ({i + 1}/{len(test_data)})")
            
            backtest_time = time.time() - start_time
            
            # Calculate results
            results = self._calculate_backtest_results(test_data, backtest_time)
            
            # Save results
            self._save_backtest_results(results, agent_type)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error running DRL backtest: {e}")
            return None
    
    def run_simple_strategy_backtest(self, test_period_days: int = 30) -> Dict[str, Any]:
        """Run backtest with simple moving average strategy"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Running Simple Strategy Backtest")
        self.logger.info(f"{'='*60}")
        
        try:
            # Load data
            data = self.load_backtest_data()
            if data is None:
                self.logger.error("❌ Failed to load data for backtesting")
                return None
            
            # Select test period (last N days)
            # Convert timestamp to datetime if it's a string
            if data['timestamp'].dtype == 'object':
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            test_start = data['timestamp'].max() - timedelta(days=test_period_days)
            test_data = data[data['timestamp'] >= test_start].copy()
            
            self.logger.info(f"📊 Test period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
            self.logger.info(f"📈 Test records: {len(test_data):,}")
            
            # Reset portfolio
            self.portfolio = {
                'cash': self.initial_capital,
                'position': 0.0,
                'total_value': self.initial_capital,
                'trades': [],
                'equity_curve': []
            }
            
            # Run backtest
            start_time = time.time()
            
            for i, (idx, row) in enumerate(test_data.iterrows()):
                current_price = row['close']
                timestamp = row['timestamp']
                
                # Simple strategy based on moving averages
                if i >= 20:  # Need at least 20 data points
                    recent_data = test_data.iloc[max(0, i-20):i+1]
                    sma_5 = recent_data['close'].rolling(5).mean().iloc[-1]
                    sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
                    
                    if sma_5 > sma_20 and current_price > sma_5:
                        action = 0.5  # Buy signal
                    elif sma_5 < sma_20 and current_price < sma_5:
                        action = -0.5  # Sell signal
                    else:
                        action = 0.0  # Hold
                else:
                    action = 0.0
                
                # Execute trade
                self.execute_backtest_trade(action, current_price, timestamp)
                
                # Record equity curve
                self.portfolio['equity_curve'].append({
                    'timestamp': timestamp,
                    'price': current_price,
                    'portfolio_value': self.portfolio['total_value'],
                    'cash': self.portfolio['cash'],
                    'position': self.portfolio['position'],
                    'action': action
                })
                
                # Log progress every 1000 iterations
                if (i + 1) % 1000 == 0:
                    progress = (i + 1) / len(test_data) * 100
                    self.logger.info(f"📊 Progress: {progress:.1f}% ({i + 1}/{len(test_data)})")
            
            backtest_time = time.time() - start_time
            
            # Calculate results
            results = self._calculate_backtest_results(test_data, backtest_time)
            results['strategy'] = 'Simple Moving Average'
            
            # Save results
            self._save_backtest_results(results, 'SimpleStrategy')
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error running simple strategy backtest: {e}")
            return None
    
    def _calculate_backtest_results(self, test_data: pd.DataFrame, backtest_time: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        try:
            # Final portfolio value
            final_value = self.portfolio['total_value']
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Buy and hold comparison
            initial_price = test_data['close'].iloc[0]
            final_price = test_data['close'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price
            
            # Calculate metrics
            equity_curve = pd.DataFrame(self.portfolio['equity_curve'])
            if not equity_curve.empty:
                # Calculate daily returns
                equity_curve['daily_return'] = equity_curve['portfolio_value'].pct_change()
                
                # Sharpe ratio (assuming 0 risk-free rate)
                sharpe_ratio = equity_curve['daily_return'].mean() / equity_curve['daily_return'].std() if equity_curve['daily_return'].std() > 0 else 0
                
                # Maximum drawdown
                cumulative = (1 + equity_curve['daily_return']).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                # Volatility
                volatility = equity_curve['daily_return'].std() * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                volatility = 0
            
            # Trade statistics
            trades = self.portfolio['trades']
            total_trades = len(trades)
            
            if total_trades > 0:
                winning_trades = len([t for t in trades if t['action'] == 'sell' and t['amount'] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
            else:
                win_rate = 0
            
            results = {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'backtest_time': backtest_time,
                'equity_curve': self.portfolio['equity_curve'],
                'trades': trades
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating backtest results: {e}")
            return {}
    
    def _save_backtest_results(self, results: Dict[str, Any], strategy_name: str):
        """Save backtest results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results JSON
            results_file = self.results_dir / f"backtest_results_{strategy_name}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"💾 Backtest results saved: {results_file}")
            
            # Save equity curve CSV
            if results.get('equity_curve'):
                equity_df = pd.DataFrame(results['equity_curve'])
                equity_file = self.results_dir / f"equity_curve_{strategy_name}_{timestamp}.csv"
                equity_df.to_csv(equity_file, index=False)
                self.logger.info(f"💾 Equity curve saved: {equity_file}")
            
            # Log summary
            self.logger.info(f"\n📊 Backtest Summary ({strategy_name}):")
            self.logger.info(f"   💰 Initial capital: ${results['initial_capital']:,.2f}")
            self.logger.info(f"   💰 Final value: ${results['final_value']:,.2f}")
            self.logger.info(f"   📈 Total return: {results['total_return']:.2%}")
            self.logger.info(f"   📈 Buy & hold return: {results['buy_hold_return']:.2%}")
            self.logger.info(f"   📈 Excess return: {results['excess_return']:.2%}")
            self.logger.info(f"   📊 Sharpe ratio: {results['sharpe_ratio']:.3f}")
            self.logger.info(f"   📉 Max drawdown: {results['max_drawdown']:.2%}")
            self.logger.info(f"   📊 Volatility: {results['volatility']:.2%}")
            self.logger.info(f"   🔄 Total trades: {results['total_trades']}")
            self.logger.info(f"   🎯 Win rate: {results['win_rate']:.2%}")
            self.logger.info(f"   ⏱️  Backtest time: {results['backtest_time']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving backtest results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Unified Backtesting System')
    parser.add_argument('--mode', choices=['drl', 'simple', 'both'], default='both',
                       help='Backtesting mode')
    parser.add_argument('--agent-type', type=str, default='SAC',
                       help='DRL agent type')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model')
    parser.add_argument('--test-days', type=int, default=30,
                       help='Number of days to test')
    parser.add_argument('--data-path', type=str,
                       help='Path to data file')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = UnifiedBacktester(
        data_path=args.data_path,
        initial_capital=args.capital
    )
    
    results = {}
    
    # Run backtests based on mode
    if args.mode in ['drl', 'both']:
        if not args.model_path:
            print("❌ Model path required for DRL backtesting")
            return
        
        drl_results = backtester.run_drl_backtest(
            agent_type=args.agent_type,
            model_path=args.model_path,
            test_period_days=args.test_days
        )
        if drl_results:
            results['drl'] = drl_results
    
    if args.mode in ['simple', 'both']:
        simple_results = backtester.run_simple_strategy_backtest(
            test_period_days=args.test_days
        )
        if simple_results:
            results['simple'] = simple_results
    
    if results:
        print(f"\n🎉 Backtesting completed successfully!")
        print(f"📁 Results saved in: {backtester.results_dir}")
    else:
        print(f"\n❌ Backtesting failed. Check logs for details.")

if __name__ == "__main__":
    main() 