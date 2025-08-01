"""
Portfolio Management for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio management system
    
    Features:
    - Position tracking
    - Cash management
    - Risk monitoring
    - Performance calculation
    - Rebalancing
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> position info
        self.trade_history = []
        self.portfolio_values = [initial_capital]
        self.current_date = None
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        stock_value = sum(pos['current_value'] for pos in self.positions.values())
        return self.cash + stock_value
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.positions.copy()
    
    def get_cash(self) -> float:
        """Get current cash"""
        return self.cash
    
    def get_position_value(self, symbol: str) -> float:
        """Get value of specific position"""
        if symbol in self.positions:
            return self.positions[symbol]['current_value']
        return 0.0
    
    def get_position_shares(self, symbol: str) -> int:
        """Get number of shares for specific position"""
        if symbol in self.positions:
            return self.positions[symbol]['shares']
        return 0
    
    def get_position_weight(self, symbol: str) -> float:
        """Get position weight as percentage of total portfolio value"""
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
        
        position = self.positions.get(symbol)
        if position is None:
            return 0.0
        
        return position['current_value'] / total_value
    
    def buy_stock(self, symbol: str, shares: int, price: float, 
                  transaction_cost: float = 0.0, timestamp: datetime = None) -> bool:
        """Buy stock"""
        if shares <= 0 or price <= 0:
            logger.warning(f"Invalid buy order: shares={shares}, price={price}")
            return False
        
        total_cost = shares * price + transaction_cost
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for buy order: required=${total_cost:.2f}, available=${self.cash:.2f}")
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Update or create position
        if symbol in self.positions:
            # Add to existing position
            position = self.positions[symbol]
            total_shares = position['shares'] + shares
            total_cost_basis = position['cost_basis'] + (shares * price)
            position['shares'] = total_shares
            position['cost_basis'] = total_cost_basis
            position['current_price'] = price
            position['current_value'] = total_shares * price
            position['unrealized_pnl'] = position['current_value'] - total_cost_basis
        else:
            # Create new position
            self.positions[symbol] = {
                'shares': shares,
                'cost_basis': shares * price,
                'current_price': price,
                'current_value': shares * price,
                'unrealized_pnl': 0,
                'entry_price': price
            }
        
        # Record trade
        trade = {
            'timestamp': timestamp or datetime.now(),
            'symbol': symbol,
            'action': 'buy',
            'shares': shares,
            'price': price,
            'value': shares * price,
            'transaction_cost': transaction_cost,
            'cash_after': self.cash
        }
        self.trade_history.append(trade)
        
        logger.info(f"Bought {shares} shares of {symbol} at ${price:.2f}")
        return True
    
    def sell_stock(self, symbol: str, shares: int, price: float,
                   transaction_cost: float = 0.0, timestamp: datetime = None) -> bool:
        """Sell stock"""
        if shares <= 0 or price <= 0:
            logger.warning(f"Invalid sell order: shares={shares}, price={price}")
            return False
        
        if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
            logger.warning(f"Insufficient shares for sell order: required={shares}, available={self.positions[symbol]['shares'] if symbol in self.positions else 0}")
            return False
        
        position = self.positions[symbol]
        
        # Calculate proceeds
        proceeds = shares * price - transaction_cost
        
        # Update cash
        self.cash += proceeds
        
        # Update position
        position['shares'] -= shares
        position['current_price'] = price
        position['current_value'] = position['shares'] * price
        
        # Update cost basis (FIFO method)
        if position['shares'] > 0:
            # Calculate average cost
            avg_cost = position['cost_basis'] / (position['shares'] + shares)
            position['cost_basis'] = position['shares'] * avg_cost
        else:
            position['cost_basis'] = 0
        
        position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
        
        # Remove position if no shares left
        if position['shares'] <= 0:
            del self.positions[symbol]
        
        # Record trade
        trade = {
            'timestamp': timestamp or datetime.now(),
            'symbol': symbol,
            'action': 'sell',
            'shares': shares,
            'price': price,
            'value': shares * price,
            'transaction_cost': transaction_cost,
            'cash_after': self.cash
        }
        self.trade_history.append(trade)
        
        logger.info(f"Sold {shares} shares of {symbol} at ${price:.2f}")
        return True
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime = None):
        """Update stock prices"""
        self.current_date = timestamp or datetime.now()
        
        # Update position values
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position['current_price'] = price
                position['current_value'] = position['shares'] * price
                position['unrealized_pnl'] = position['current_value'] - position['cost_basis']
        
        # Record portfolio value
        self.portfolio_values.append(self.get_total_value())
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_value = self.get_total_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': total_value,
            'cash': self.cash,
            'cash_pct': self.cash / total_value if total_value > 0 else 0,
            'stock_value': total_value - self.cash,
            'stock_pct': (total_value - self.cash) / total_value if total_value > 0 else 0,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
        
        return summary
    
    def get_position_summary(self) -> Dict:
        """Get detailed position summary"""
        summary = {}
        
        for symbol, position in self.positions.items():
            summary[symbol] = {
                'shares': position['shares'],
                'current_price': position['current_price'],
                'current_value': position['current_value'],
                'cost_basis': position['cost_basis'],
                'unrealized_pnl': position['unrealized_pnl'],
                'unrealized_pnl_pct': position['unrealized_pnl'] / position['cost_basis'] if position['cost_basis'] > 0 else 0,
                'weight': position['current_value'] / self.get_total_value() if self.get_total_value() > 0 else 0
            }
        
        return summary
    
    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        if not self.positions:
            return {}
        
        position_values = [pos['current_value'] for pos in self.positions.values()]
        total_value = self.get_total_value()
        
        # Concentration metrics
        concentration = sum(position_values) / total_value if total_value > 0 else 0
        max_position_pct = max(position_values) / total_value if total_value > 0 else 0
        
        # Volatility (simplified)
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
        else:
            volatility = 0
        
        metrics = {
            'concentration': concentration,
            'max_position_pct': max_position_pct,
            'num_positions': len(self.positions),
            'volatility': volatility,
            'cash_ratio': self.cash / total_value if total_value > 0 else 0
        }
        
        return metrics
    
    def rebalance(self, target_weights: Dict[str, float], current_prices: Dict[str, float]) -> List[Dict]:
        """Rebalance portfolio to target weights"""
        trades = []
        total_value = self.get_total_value()
        
        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            target_value = total_value * target_weight
            current_value = self.get_position_value(symbol)
            
            if abs(target_value - current_value) > total_value * 0.01:  # 1% threshold
                if target_value > current_value:
                    # Need to buy
                    shares_to_buy = int((target_value - current_value) / current_price)
                    if shares_to_buy > 0:
                        success = self.buy_stock(symbol, shares_to_buy, current_price)
                        if success:
                            trades.append({
                                'symbol': symbol,
                                'action': 'buy',
                                'shares': shares_to_buy,
                                'price': current_price
                            })
                else:
                    # Need to sell
                    shares_to_sell = int((current_value - target_value) / current_price)
                    if shares_to_sell > 0:
                        success = self.sell_stock(symbol, shares_to_sell, current_price)
                        if success:
                            trades.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'shares': shares_to_sell,
                                'price': current_price
                            })
        
        return trades
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history.copy()
    
    def get_portfolio_history(self) -> List[float]:
        """Get portfolio value history"""
        return self.portfolio_values.copy()
    
    def calculate_returns(self) -> np.ndarray:
        """Calculate portfolio returns"""
        if len(self.portfolio_values) < 2:
            return np.array([])
        
        return np.diff(self.portfolio_values) / self.portfolio_values[:-1]
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        returns = self.calculate_returns()
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        portfolio_array = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        return np.min(drawdown) 

    def get_position(self, symbol: str) -> int:
        """Get number of shares for specific position (compatibility)"""
        return self.get_position_shares(symbol) 