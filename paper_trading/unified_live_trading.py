#!/usr/bin/env python3
"""
Unified Live Trading System
Combines all live trading functionality into one comprehensive script
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Mock the required classes for now to avoid import issues
class CCXTProvider:
    """Mock CCXTProvider for unified live trading"""
    
    def __init__(self, exchange: str = 'mexc'):
        self.exchange = exchange
        
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int = None, limit: int = 1000):
        """Mock fetch OHLCV data"""
        # Return mock data
        import time
        current_time = int(time.time() * 1000)
        mock_data = []
        for i in range(limit):
            mock_data.append([
                current_time + i * 900000,  # 15-minute intervals
                50000 + i * 10,  # open
                50100 + i * 10,  # high
                49900 + i * 10,  # low
                50050 + i * 10,  # close
                1000000 + i * 1000  # volume
            ])
        return mock_data

class CryptoTradingAgent:
    """Mock CryptoTradingAgent for unified live trading"""
    
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

class UnifiedLiveTrader:
    """Unified live trader that combines all live trading functionality"""
    
    def __init__(self, exchange: str = 'mexc', symbol: str = 'BTC/USDT:USDT', 
                 initial_capital: float = 100000.0, use_drl: bool = True):
        self.exchange = exchange
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.use_drl = use_drl
        
        # Setup directories
        self.results_dir = Path('paper_trading_data/unified_live_trading')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.data_provider = CCXTProvider(exchange=exchange)
        self.drl_agent = None
        self.portfolio = {
            'cash': initial_capital,
            'position': 0.0,
            'total_value': initial_capital,
            'trades': []
        }
        
        # Trading state
        self.is_running = False
        self.current_price = 0.0
        self.last_action = 0.0
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.results_dir / f"unified_live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Unified Live Trading System Started")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        self.logger.info(f"📊 Exchange: {self.exchange}")
        self.logger.info(f"💰 Symbol: {self.symbol}")
        self.logger.info(f"💵 Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"🤖 DRL enabled: {self.use_drl}")
        
    def load_drl_agent(self, agent_type: str, model_path: str):
        """Load a trained DRL agent"""
        try:
            self.logger.info(f"🤖 Loading DRL agent: {agent_type}")
            
            self.drl_agent = CryptoTradingAgent(
                agent_type=agent_type,
                state_dim=15,
                action_dim=1,
                hidden_dim=64
            )
            
            self.drl_agent.load_agent(model_path)
            self.logger.info(f"✅ DRL agent loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading DRL agent: {e}")
            return False
    
    def get_current_market_data(self) -> Optional[pd.DataFrame]:
        """Get current market data"""
        try:
            # Fetch recent OHLCV data
            data = self.data_provider.fetch_ohlcv(
                symbol=self.symbol,
                timeframe='15m',
                limit=100
            )
            
            if data and len(data) > 0:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching market data: {e}")
            return None
    
    def prepare_state_vector(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare state vector for DRL agent"""
        try:
            # Calculate technical indicators
            close_prices = data['close'].values
            volumes = data['volume'].values
            
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
                self.last_action,
                close_prices[-1] / 100000,  # Normalized price
                volumes[-1] / 1000000,  # Normalized volume
                len(self.portfolio['trades']),
                self.portfolio['total_value'] / self.initial_capital - 1  # Return
            ])
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing state vector: {e}")
            return np.zeros(15)
    
    def get_trading_action(self, state: np.ndarray) -> float:
        """Get trading action from DRL agent or simple strategy"""
        try:
            if self.use_drl and self.drl_agent is not None:
                # Use DRL agent
                action = self.drl_agent.get_action(state)
                self.logger.debug(f"🤖 DRL action: {action:.3f}")
                return action
            else:
                # Simple strategy based on moving averages
                sma_5 = state[3] + 1  # Convert back from ratio
                sma_20 = state[5] + 1
                current_price = state[0] * 100000  # Convert back from normalized
                
                if sma_5 > sma_20 and current_price > sma_5:
                    action = 0.5  # Buy signal
                elif sma_5 < sma_20 and current_price < sma_5:
                    action = -0.5  # Sell signal
                else:
                    action = 0.0  # Hold
                
                self.logger.debug(f"📊 Simple strategy action: {action:.3f}")
                return action
                
        except Exception as e:
            self.logger.error(f"❌ Error getting trading action: {e}")
            return 0.0
    
    def execute_trade(self, action: float, current_price: float):
        """Execute a trade based on the action"""
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
                        'timestamp': datetime.now(),
                        'action': 'buy',
                        'price': current_price,
                        'amount': trade_amount,
                        'shares': shares
                    })
                    
                    self.logger.info(f"💰 BUY: {shares:.6f} shares at ${current_price:,.2f} (${trade_amount:,.2f})")
            
            elif action < 0:  # Sell
                if self.portfolio['position'] > 0:
                    shares_to_sell = self.portfolio['position'] * trade_size
                    trade_amount = shares_to_sell * current_price
                    
                    self.portfolio['cash'] += trade_amount
                    self.portfolio['position'] -= shares_to_sell
                    
                    self.portfolio['trades'].append({
                        'timestamp': datetime.now(),
                        'action': 'sell',
                        'price': current_price,
                        'amount': trade_amount,
                        'shares': shares_to_sell
                    })
                    
                    self.logger.info(f"💰 SELL: {shares_to_sell:.6f} shares at ${current_price:,.2f} (${trade_amount:,.2f})")
            
            # Update total value
            self.portfolio['total_value'] = self.portfolio['cash'] + (self.portfolio['position'] * current_price)
            
        except Exception as e:
            self.logger.error(f"❌ Error executing trade: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            current_value = self.portfolio['total_value']
            total_return = (current_value - self.initial_capital) / self.initial_capital
            
            return {
                'cash': self.portfolio['cash'],
                'position': self.portfolio['position'],
                'total_value': current_value,
                'total_return': total_return,
                'total_trades': len(self.portfolio['trades']),
                'current_price': self.current_price
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting portfolio status: {e}")
            return {}
    
    def run_trading_session(self, duration_minutes: int = 60, interval_seconds: int = 60):
        """Run a live trading session"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 Starting Live Trading Session")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"⏱️  Duration: {duration_minutes} minutes")
        self.logger.info(f"⏰ Interval: {interval_seconds} seconds")
        self.logger.info(f"🤖 DRL enabled: {self.use_drl}")
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        session_data = []
        
        try:
            while self.is_running and datetime.now() < end_time:
                # Get current market data
                market_data = self.get_current_market_data()
                if market_data is None or market_data.empty:
                    self.logger.warning("⚠️  No market data available, skipping iteration")
                    time.sleep(interval_seconds)
                    continue
                
                # Update current price
                self.current_price = market_data['close'].iloc[-1]
                
                # Prepare state vector
                state = self.prepare_state_vector(market_data)
                
                # Get trading action
                action = self.get_trading_action(state)
                self.last_action = action
                
                # Execute trade
                self.execute_trade(action, self.current_price)
                
                # Get portfolio status
                status = self.get_portfolio_status()
                
                # Log status
                self.logger.info(f"📊 Status: Price=${self.current_price:,.2f}, "
                               f"Value=${status['total_value']:,.2f}, "
                               f"Return={status['total_return']:.2%}, "
                               f"Trades={status['total_trades']}")
                
                # Store session data
                session_data.append({
                    'timestamp': datetime.now(),
                    'price': self.current_price,
                    'action': action,
                    'portfolio_value': status['total_value'],
                    'total_return': status['total_return'],
                    'cash': status['cash'],
                    'position': status['position']
                })
                
                # Wait for next iteration
                time.sleep(interval_seconds)
            
            # Session completed
            self.logger.info(f"\n🎉 Trading session completed!")
            self.logger.info(f"⏱️  Duration: {datetime.now() - start_time}")
            
            # Save session results
            self._save_session_results(session_data, start_time, datetime.now())
            
        except KeyboardInterrupt:
            self.logger.info("⏹️  Trading session interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error during trading session: {e}")
        finally:
            self.is_running = False
    
    def _save_session_results(self, session_data: List[Dict], start_time: datetime, end_time: datetime):
        """Save trading session results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Convert to DataFrame
            df = pd.DataFrame(session_data)
            
            # Save session data
            session_file = self.results_dir / f"trading_session_{timestamp}.csv"
            df.to_csv(session_file, index=False)
            self.logger.info(f"💾 Session data saved: {session_file}")
            
            # Calculate session statistics
            if len(session_data) > 0:
                initial_value = session_data[0]['portfolio_value']
                final_value = session_data[-1]['portfolio_value']
                session_return = (final_value - initial_value) / initial_value
                
                # Count trades
                trades_count = len([d for d in session_data if abs(d['action']) > 0.1])
                
                # Save summary
                summary = {
                    'session_start': start_time.isoformat(),
                    'session_end': end_time.isoformat(),
                    'duration_minutes': (end_time - start_time).total_seconds() / 60,
                    'initial_value': initial_value,
                    'final_value': final_value,
                    'session_return': session_return,
                    'total_trades': trades_count,
                    'final_portfolio_status': self.get_portfolio_status()
                }
                
                summary_file = self.results_dir / f"session_summary_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(f"💾 Session summary saved: {summary_file}")
                
                self.logger.info(f"\n📊 Session Summary:")
                self.logger.info(f"   💰 Initial value: ${initial_value:,.2f}")
                self.logger.info(f"   💰 Final value: ${final_value:,.2f}")
                self.logger.info(f"   📈 Session return: {session_return:.2%}")
                self.logger.info(f"   🔄 Total trades: {trades_count}")
                
        except Exception as e:
            self.logger.error(f"❌ Error saving session results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Unified Live Trading System')
    parser.add_argument('--mode', choices=['live', 'demo'], default='demo',
                       help='Trading mode')
    parser.add_argument('--duration', type=int, default=60,
                       help='Trading session duration in minutes')
    parser.add_argument('--interval', type=int, default=60,
                       help='Data collection interval in seconds')
    parser.add_argument('--agent-type', type=str, default='SAC',
                       help='DRL agent type')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model')
    parser.add_argument('--exchange', type=str, default='mexc',
                       help='Exchange to use')
    parser.add_argument('--symbol', type=str, default='BTC/USDT:USDT',
                       help='Trading symbol')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--no-drl', action='store_true',
                       help='Disable DRL and use simple strategy')
    
    args = parser.parse_args()
    
    # Create trader
    trader = UnifiedLiveTrader(
        exchange=args.exchange,
        symbol=args.symbol,
        initial_capital=args.capital,
        use_drl=not args.no_drl
    )
    
    # Load DRL agent if specified
    if trader.use_drl and args.model_path:
        if not trader.load_drl_agent(args.agent_type, args.model_path):
            print("❌ Failed to load DRL agent. Exiting.")
            return
    
    # Run trading session
    trader.run_trading_session(
        duration_minutes=args.duration,
        interval_seconds=args.interval
    )

if __name__ == "__main__":
    main() 