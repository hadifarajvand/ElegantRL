"""
Enhanced Trading Environment for Paper Trading System
"""

import numpy as np
import pandas as pd
import torch as th
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import data persistence
try:
    from ..utils.data_persistence import create_data_persistence_manager
    DATA_PERSISTENCE_AVAILABLE = True
except ImportError:
    DATA_PERSISTENCE_AVAILABLE = False

ARY = np.ndarray
TEN = th.Tensor


class EnhancedStockTradingEnv(gym.Env):
    """
    Enhanced Stock Trading Environment for Paper Trading
    
    Features:
    - Multiple data sources support
    - Advanced risk management
    - Realistic transaction costs and slippage
    - Portfolio constraints
    - Technical indicators
    - Position sizing
    """
    
    def __init__(self, 
                 data: Dict[str, pd.DataFrame],
                 initial_capital: float = 1000000.0,
                 max_stock_quantity: int = 100,
                 transaction_cost_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 max_position_size: float = 0.2,
                 min_cash_reserve: float = 0.1,
                 max_leverage: float = 1.5,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 reward_scale: float = 2 ** -12,
                 gamma: float = 0.99,
                 if_random_reset: bool = True,
                 enable_data_persistence: bool = True,
                 persistence_dir: str = "./paper_trading_data"):
        
        super().__init__()
        
        # Data
        self.data = data
        self.symbols = list(data.keys())
        self.num_stocks = len(self.symbols)
        
        # Get common date range
        self._prepare_data()
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.max_stock_quantity = max_stock_quantity
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_size = max_position_size
        self.min_cash_reserve = min_cash_reserve
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.if_random_reset = if_random_reset
        
        # Environment state
        self.day = None
        self.amount = None  # Cash
        self.shares = None  # Stock positions
        self.total_asset = None
        self.rewards = None
        self.cumulative_returns = 0
        
        # Position tracking
        self.position_values = None
        self.position_returns = None
        self.entry_prices = None
        
        # Environment information
        self.env_name = 'EnhancedStockTradingEnv-v1'
        self.state_dim = self._calculate_state_dim()
        self.action_dim = self.num_stocks
        self.if_discrete = False
        self.max_step = self.close_prices.shape[0] - 1
        self.target_return = +np.inf
        
        # Data persistence
        self.enable_data_persistence = enable_data_persistence and DATA_PERSISTENCE_AVAILABLE
        if self.enable_data_persistence:
            self.persistence_manager = create_data_persistence_manager(persistence_dir)
            self.trade_logs = []
            self.performance_history = []
        else:
            self.persistence_manager = None
            self.trade_logs = []
            self.performance_history = []
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
    
    def _prepare_data(self):
        """Prepare and align data from multiple symbols"""
        # Find common date range
        start_dates = []
        end_dates = []
        
        for symbol, df in self.data.items():
            if not df.empty:
                start_dates.append(df.index.min())
                end_dates.append(df.index.max())
        
        if not start_dates:
            raise ValueError("No valid data found")
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Align data
        aligned_data = {}
        for symbol, df in self.data.items():
            if not df.empty:
                aligned_df = df[(df.index >= common_start) & (df.index <= common_end)]
                if not aligned_df.empty:
                    aligned_data[symbol] = aligned_df
        
        self.data = aligned_data
        self.symbols = list(aligned_data.keys())
        self.num_stocks = len(self.symbols)
        
        # Create price arrays
        self.close_prices = np.zeros((len(aligned_data[self.symbols[0]]), self.num_stocks))
        self.tech_features = []
        
        for i, symbol in enumerate(self.symbols):
            df = aligned_data[symbol]
            self.close_prices[:, i] = df['close'].values
            
            # Technical features (excluding OHLCV)
            tech_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            if tech_cols:
                tech_data = df[tech_cols].values
                self.tech_features.append(tech_data)
        
        if self.tech_features:
            self.tech_features = np.concatenate(self.tech_features, axis=1)
        else:
            self.tech_features = np.zeros((self.close_prices.shape[0], 0))
    
    def _calculate_state_dim(self) -> int:
        """Calculate state dimension"""
        # Cash (1) + Positions (num_stocks) + Prices (num_stocks) + Tech features
        return 1 + self.num_stocks + self.num_stocks + self.tech_features.shape[1]
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ARY, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.day = 0
        
        if self.if_random_reset:
            self.amount = self.initial_capital * np.random.uniform(0.9, 1.1)
            self.shares = (np.abs(np.random.randn(self.num_stocks).clip(-2, +2)) * 2 ** 6).astype(int)
        else:
            self.amount = self.initial_capital
            self.shares = np.zeros(self.num_stocks, dtype=np.float32)
        
        # Initialize position tracking
        self.position_values = np.zeros(self.num_stocks)
        self.position_returns = np.zeros(self.num_stocks)
        self.entry_prices = np.zeros(self.num_stocks)
        
        self.rewards = []
        self.total_asset = self._calculate_total_asset()
        
        return self.get_state(), {}
    
    def get_state(self) -> ARY:
        """Get current state"""
        # Normalize cash
        cash_norm = np.array([np.tanh(self.amount * 2 ** -16)])
        
        # Normalize positions
        positions_norm = self.shares * 2 ** -9
        
        # Normalize prices
        prices_norm = self.close_prices[self.day] * 2 ** -7
        
        # Normalize technical features
        if self.tech_features.shape[1] > 0:
            tech_norm = self.tech_features[self.day] * 2 ** -6
        else:
            tech_norm = np.array([])
        
        state = np.concatenate([cash_norm, positions_norm, prices_norm, tech_norm])
        return state.astype(np.float32)
    
    def step(self, action: ARY) -> Tuple[ARY, float, bool, bool, Dict]:
        """Execute trading action"""
        self.day += 1
        
        # Validate action
        action = np.clip(action, -1, 1)
        
        # Execute trades
        old_total_asset = self.total_asset
        trade_logs = self._execute_trades(action)
        
        # Calculate new total asset
        self.total_asset = self._calculate_total_asset()
        
        # Calculate reward
        reward = (self.total_asset - old_total_asset) * self.reward_scale
        self.rewards.append(reward)
        
        # Log trades if persistence is enabled
        if self.enable_data_persistence and trade_logs:
            for trade_log in trade_logs:
                trade_log.update({
                    'timestamp': datetime.now().isoformat(),
                    'portfolio_value': self.total_asset,
                    'reward': reward,
                    'step': self.day
                })
                self.trade_logs.append(trade_log)
                if self.persistence_manager:
                    self.persistence_manager.save_trade_log(trade_log)
        
        # Check if episode is done
        done = self.day >= self.max_step
        truncated = False
        
        # Update position tracking
        self._update_position_tracking()
        
        # Get next state
        if done:
            # Save performance statistics if persistence is enabled
            if self.enable_data_persistence and self.persistence_manager:
                stats = self.get_portfolio_stats()
                self.persistence_manager.save_performance_metrics(stats, "trading_episode")
                
                # Save all trade logs
                if self.trade_logs:
                    self.persistence_manager.save_trade_logs_batch(self.trade_logs)
            
            state = self.reset()[0]
        else:
            state = self.get_state()
        
        info = {
            'total_asset': self.total_asset,
            'cash': self.amount,
            'positions': self.shares.copy(),
            'day': self.day,
            'portfolio_return': (self.total_asset - self.initial_capital) / self.initial_capital
        }
        
        return state, reward, done, truncated, info
    
    def _execute_trades(self, action: ARY) -> List[Dict[str, Any]]:
        """Execute trading actions with risk management and return trade logs"""
        # Convert action to position changes
        action_int = (action * self.max_stock_quantity).astype(int)
        
        trade_logs = []
        
        # Apply slippage and transaction costs
        for i, symbol in enumerate(self.symbols):
            current_price = self.close_prices[self.day, i]
            
            # Calculate slippage
            slippage = current_price * self.slippage_pct
            
            if action_int[i] > 0:  # Buy
                # Check cash reserve constraint
                max_buy_amount = self.amount * (1 - self.min_cash_reserve)
                max_shares = int(max_buy_amount / (current_price * (1 + self.transaction_cost_pct)))
                
                # Check position size constraint
                current_position_value = self.shares[i] * current_price
                max_position_value = self.total_asset * self.max_position_size
                max_shares_by_position = int((max_position_value - current_position_value) / current_price)
                
                shares_to_buy = min(action_int[i], max_shares, max_shares_by_position)
                
                if shares_to_buy > 0:
                    cost = current_price * shares_to_buy * (1 + self.transaction_cost_pct + self.slippage_pct)
                    if cost <= self.amount:
                        self.amount -= cost
                        self.shares[i] += shares_to_buy
                        self.entry_prices[i] = current_price
                        
                        # Log trade
                        trade_logs.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': shares_to_buy,
                            'price': current_price,
                            'cash': self.amount,
                            'positions': self.shares.copy().tolist()
                        })
            
            elif action_int[i] < 0 and self.shares[i] > 0:  # Sell
                shares_to_sell = min(-action_int[i], self.shares[i])
                
                # Check stop loss and take profit
                if self.entry_prices[i] > 0:
                    current_return = (current_price - self.entry_prices[i]) / self.entry_prices[i]
                    
                    # Stop loss
                    if current_return <= -self.stop_loss_pct:
                        shares_to_sell = self.shares[i]  # Sell all
                    
                    # Take profit
                    elif current_return >= self.take_profit_pct:
                        shares_to_sell = int(self.shares[i] * 0.5)  # Sell half
                
                if shares_to_sell > 0:
                    proceeds = current_price * shares_to_sell * (1 - self.transaction_cost_pct - self.slippage_pct)
                    self.amount += proceeds
                    self.shares[i] -= shares_to_sell
                    
                    # Log trade
                    trade_logs.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': shares_to_sell,
                        'price': current_price,
                        'cash': self.amount,
                        'positions': self.shares.copy().tolist()
                    })
        
        return trade_logs
    
    def _calculate_total_asset(self) -> float:
        """Calculate total portfolio value"""
        stock_value = np.sum(self.close_prices[self.day] * self.shares)
        return stock_value + self.amount
    
    def _update_position_tracking(self):
        """Update position tracking information"""
        for i in range(self.num_stocks):
            if self.shares[i] > 0:
                current_price = self.close_prices[self.day, i]
                self.position_values[i] = self.shares[i] * current_price
                
                if self.entry_prices[i] > 0:
                    self.position_returns[i] = (current_price - self.entry_prices[i]) / self.entry_prices[i]
    
    def render(self):
        """Render current state"""
        print(f"Day: {self.day}")
        print(f"Total Asset: ${self.total_asset:,.2f}")
        print(f"Cash: ${self.amount:,.2f}")
        print(f"Portfolio Return: {((self.total_asset - self.initial_capital) / self.initial_capital) * 100:.2f}%")
        
        print("\nPositions:")
        for i, symbol in enumerate(self.symbols):
            if self.shares[i] > 0:
                current_price = self.close_prices[self.day, i]
                position_value = self.shares[i] * current_price
                position_return = self.position_returns[i] * 100 if self.entry_prices[i] > 0 else 0
                print(f"  {symbol}: {self.shares[i]} shares @ ${current_price:.2f} = ${position_value:,.2f} ({position_return:+.2f}%)")
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio statistics"""
        if not self.rewards:
            return {}
        
        returns = np.array(self.rewards) / self.reward_scale
        cumulative_return = (self.total_asset - self.initial_capital) / self.initial_capital
        
        stats = {
            'total_return': cumulative_return,
            'total_return_pct': cumulative_return * 100,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
            'max_drawdown': self._calculate_max_drawdown(),
            'volatility': np.std(returns),
            'final_value': self.total_asset
        }
        
        return stats
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.rewards:
            return 0.0
        
        cumulative_returns = np.cumsum(self.rewards) / self.reward_scale
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        return np.min(drawdown) 