"""
Paper Trading Engine - Main Trading System
"""

import numpy as np
import pandas as pd
import torch as th
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from datetime import datetime, timedelta
import threading
import queue
import json
import os

from ..configs.trading_config import TradingConfig
from ..configs.model_config import ModelConfig
from ..data.market_data import DataManager, YahooFinanceProvider
from ..models.trading_env import EnhancedStockTradingEnv
from .risk_manager import RiskManager
from .order_manager import OrderManager

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Main Paper Trading Engine
    
    Features:
    - Real-time market data processing
    - DRL agent inference
    - Risk management
    - Order execution simulation
    - Performance tracking
    - Portfolio management
    """
    
    def __init__(self, 
                 trading_config: TradingConfig,
                 model_config: ModelConfig,
                 model_path: str,
                 symbols: List[str],
                 data_provider: str = "yahoo"):
        
        self.trading_config = trading_config
        self.model_config = model_config
        self.model_path = model_path
        self.symbols = symbols
        
        # Initialize components
        self._initialize_data_provider(data_provider)
        self._initialize_risk_manager()
        self._initialize_order_manager()
        self._load_model()
        
        # Trading state
        self.is_running = False
        self.current_positions = {}
        self.portfolio_value = trading_config.initial_capital
        self.cash = trading_config.initial_capital
        self.trade_history = []
        self.performance_metrics = {}
        
        # Threading
        self.data_queue = queue.Queue()
        self.action_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Logging
        self.setup_logging()
    
    def _initialize_data_provider(self, provider_type: str):
        """Initialize market data provider"""
        if provider_type == "yahoo":
            self.data_provider = YahooFinanceProvider()
        elif provider_type == "alpaca":
            self.data_provider = AlpacaProvider(
                self.trading_config.api_key,
                self.trading_config.api_secret,
                self.trading_config.api_base_url
            )
        else:
            raise ValueError(f"Unsupported data provider: {provider_type}")
        
        self.data_manager = DataManager(self.data_provider)
    
    def _initialize_risk_manager(self):
        """Initialize risk management system"""
        self.risk_manager = RiskManager(
            max_position_size=self.trading_config.max_position_size,
            stop_loss_pct=self.trading_config.stop_loss_pct,
            take_profit_pct=self.trading_config.take_profit_pct,
            max_leverage=self.trading_config.max_leverage,
            min_cash_reserve=self.trading_config.min_cash_reserve
        )
    
    def _initialize_order_manager(self):
        """Initialize order management system"""
        self.order_manager = OrderManager(
            transaction_cost_pct=self.trading_config.transaction_cost_pct,
            slippage_pct=self.trading_config.slippage_pct
        )
    
    def _load_model(self):
        """Load trained DRL model"""
        try:
            # Load model weights
            self.model = self._load_model_weights()
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_model_weights(self):
        """Load model weights based on agent type"""
        # This is a placeholder - you'll need to implement based on your model format
        # For ElegantRL, you might load the actor network
        if self.model_config.agent_type == "PPO":
            return self._load_ppo_model()
        elif self.model_config.agent_type == "A2C":
            return self._load_a2c_model()
        else:
            raise ValueError(f"Unsupported agent type: {self.model_config.agent_type}")
    
    def _load_ppo_model(self):
        """Load PPO model"""
        # Placeholder implementation
        # You'll need to implement based on your specific model format
        return None
    
    def _load_a2c_model(self):
        """Load A2C model"""
        # Placeholder implementation
        return None
    
    def setup_logging(self):
        """Setup logging for trading engine"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = f"{log_dir}/paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    def start(self):
        """Start paper trading"""
        logger.info("Starting Paper Trading Engine...")
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self._data_collection_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start trading loop
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        logger.info("Paper Trading Engine started successfully")
    
    def stop(self):
        """Stop paper trading"""
        logger.info("Stopping Paper Trading Engine...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=5)
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=5)
        
        # Save final state
        self._save_trading_state()
        
        logger.info("Paper Trading Engine stopped")
    
    def _data_collection_loop(self):
        """Data collection loop"""
        while not self.stop_event.is_set():
            try:
                # Get real-time data
                current_prices = self.data_provider.get_realtime_data(self.symbols)
                
                if current_prices:
                    # Add timestamp
                    data = {
                        'timestamp': datetime.now(),
                        'prices': current_prices
                    }
                    
                    # Put data in queue
                    self.data_queue.put(data)
                
                # Sleep based on update frequency
                time.sleep(self._get_update_interval())
                
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                time.sleep(5)
    
    def _trading_loop(self):
        """Main trading loop"""
        while not self.stop_event.is_set():
            try:
                # Get latest data
                if not self.data_queue.empty():
                    data = self.data_queue.get()
                    self._process_market_data(data)
                
                # Check if it's time to trade
                if self._should_trade():
                    self._execute_trading_cycle()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)
    
    def _process_market_data(self, data: Dict):
        """Process incoming market data"""
        timestamp = data['timestamp']
        prices = data['prices']
        
        # Update current positions
        for symbol, price in prices.items():
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                position['current_price'] = price
                position['current_value'] = position['shares'] * price
                position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check risk management rules
        self.risk_manager.check_positions(self.current_positions, self.portfolio_value)
    
    def _should_trade(self) -> bool:
        """Check if it's time to trade"""
        now = datetime.now()
        
        # Check trading hours
        start_time = datetime.strptime(self.trading_config.trading_hours[0], "%H:%M").time()
        end_time = datetime.strptime(self.trading_config.trading_hours[1], "%H:%M").time()
        
        if not (start_time <= now.time() <= end_time):
            return False
        
        # Check rebalancing frequency
        if self.trading_config.rebalance_frequency == "daily":
            return True  # Trade every day during market hours
        elif self.trading_config.rebalance_frequency == "weekly":
            return now.weekday() == 0  # Monday
        elif self.trading_config.rebalance_frequency == "monthly":
            return now.day == 1  # First day of month
        
        return True
    
    def _execute_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Get current state
            state = self._get_current_state()
            
            # Get model prediction
            action = self._get_model_action(state)
            
            # Apply risk management
            action = self.risk_manager.validate_action(action, self.current_positions, self.portfolio_value)
            
            # Execute orders
            executed_orders = self.order_manager.execute_orders(
                action, self.current_positions, self.symbols
            )
            
            # Update positions
            self._update_positions(executed_orders)
            
            # Log trading activity
            self._log_trading_activity(action, executed_orders)
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _get_current_state(self) -> np.ndarray:
        """Get current market state for model"""
        # This should match the state format used during training
        # You'll need to implement based on your specific state representation
        
        # Placeholder implementation
        state_dim = self.model_config.state_dim
        state = np.random.randn(state_dim)  # Replace with actual state calculation
        
        return state
    
    def _get_model_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from trained model"""
        # Convert state to tensor
        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        
        # Get model prediction
        with th.no_grad():
            if self.model_config.agent_type == "PPO":
                action = self._get_ppo_action(state_tensor)
            elif self.model_config.agent_type == "A2C":
                action = self._get_a2c_action(state_tensor)
            else:
                raise ValueError(f"Unsupported agent type: {self.model_config.agent_type}")
        
        return action.numpy().flatten()
    
    def _get_ppo_action(self, state: th.Tensor) -> th.Tensor:
        """Get action from PPO model"""
        # Placeholder implementation
        # You'll need to implement based on your specific PPO model
        return th.randn(self.model_config.action_dim)
    
    def _get_a2c_action(self, state: th.Tensor) -> th.Tensor:
        """Get action from A2C model"""
        # Placeholder implementation
        return th.randn(self.model_config.action_dim)
    
    def _update_positions(self, executed_orders: List[Dict]):
        """Update current positions based on executed orders"""
        for order in executed_orders:
            symbol = order['symbol']
            shares = order['shares']
            price = order['price']
            order_type = order['type']
            
            if symbol not in self.current_positions:
                self.current_positions[symbol] = {
                    'shares': 0,
                    'cost_basis': 0,
                    'current_price': price,
                    'current_value': 0,
                    'unrealized_pnl': 0
                }
            
            position = self.current_positions[symbol]
            
            if order_type == 'buy':
                # Update position
                total_cost = position['cost_basis'] + (shares * price)
                total_shares = position['shares'] + shares
                position['shares'] = total_shares
                position['cost_basis'] = total_cost
                position['current_price'] = price
                position['current_value'] = total_shares * price
                position['unrealized_pnl'] = position['current_value'] - total_cost
                
                # Update cash
                self.cash -= shares * price
                
            elif order_type == 'sell':
                # Update position
                position['shares'] -= shares
                position['current_price'] = price
                position['current_value'] = position['shares'] * price
                
                # Update cost basis (FIFO method)
                avg_cost = position['cost_basis'] / (position['shares'] + shares)
                position['cost_basis'] = position['shares'] * avg_cost
                position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
                
                # Update cash
                self.cash += shares * price
                
                # Remove position if no shares left
                if position['shares'] <= 0:
                    del self.current_positions[symbol]
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        stock_value = sum(pos['current_value'] for pos in self.current_positions.values())
        self.portfolio_value = self.cash + stock_value
    
    def _log_trading_activity(self, action: np.ndarray, executed_orders: List[Dict]):
        """Log trading activity"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action.tolist(),
            'executed_orders': executed_orders,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.current_positions.copy()
        }
        
        self.trade_history.append(log_entry)
        logger.info(f"Trading activity logged: {len(executed_orders)} orders executed")
    
    def _save_trading_state(self):
        """Save current trading state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.current_positions,
            'trade_history': self.trade_history,
            'performance_metrics': self.performance_metrics
        }
        
        # Save to file
        state_file = f"trading_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Trading state saved to {state_file}")
    
    def _get_update_interval(self) -> int:
        """Get update interval in seconds based on frequency"""
        frequency_map = {
            "1min": 60,
            "5min": 300,
            "15min": 900,
            "1hour": 3600,
            "daily": 86400
        }
        
        return frequency_map.get(self.trading_config.update_frequency, 60)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.trade_history:
            return {}
        
        # Calculate metrics
        initial_value = self.trading_config.initial_capital
        final_value = self.portfolio_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = []
        dates = []
        
        for entry in self.trade_history:
            dates.append(entry['timestamp'])
            daily_returns.append(entry['portfolio_value'])
        
        if len(daily_returns) > 1:
            returns = np.diff(daily_returns) / daily_returns[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            volatility = np.std(returns)
        else:
            sharpe_ratio = 0
            volatility = 0
        
        summary = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'num_trades': len(self.trade_history),
            'current_positions': len(self.current_positions),
            'cash_ratio': self.cash / self.portfolio_value
        }
        
        return summary 