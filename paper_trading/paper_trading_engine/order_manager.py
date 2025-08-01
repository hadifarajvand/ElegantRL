"""
Order Management System for Paper Trading
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Order Management System
    
    Features:
    - Order execution simulation
    - Transaction costs calculation
    - Slippage modeling
    - Order tracking
    - Execution reporting
    """
    
    def __init__(self,
                 transaction_cost_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 min_order_size: float = 100.0,
                 max_order_size: float = 1000000.0):
        
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.order_history = []
    
    def execute_orders(self, action: np.ndarray, positions: Dict, symbols: List[str]) -> List[Dict]:
        """Execute trading orders based on action"""
        executed_orders = []
        
        for i, symbol in enumerate(symbols):
            if i >= len(action):
                continue
            
            order_amount = action[i]
            
            if abs(order_amount) < 0.01:  # Ignore very small orders
                continue
            
            # Get current price (placeholder - should come from market data)
            current_price = self._get_current_price(symbol)
            
            if order_amount > 0:  # Buy order
                order = self._execute_buy_order(symbol, order_amount, current_price)
                if order:
                    executed_orders.append(order)
            
            elif order_amount < 0:  # Sell order
                order = self._execute_sell_order(symbol, abs(order_amount), current_price, positions)
                if order:
                    executed_orders.append(order)
        
        # Log execution
        if executed_orders:
            self._log_order_execution(executed_orders)
        
        return executed_orders
    
    def _execute_buy_order(self, symbol: str, order_amount: float, current_price: float) -> Optional[Dict]:
        """Execute a buy order"""
        try:
            # Calculate order size
            order_value = order_amount * 10000  # Scale to reasonable order size
            
            if order_value < self.min_order_size:
                logger.debug(f"Order too small for {symbol}: ${order_value:.2f}")
                return None
            
            if order_value > self.max_order_size:
                order_value = self.max_order_size
                logger.warning(f"Order size capped for {symbol}")
            
            # Calculate shares to buy
            shares = int(order_value / current_price)
            
            if shares <= 0:
                return None
            
            # Apply slippage and transaction costs
            execution_price = current_price * (1 + self.slippage_pct)
            total_cost = shares * execution_price * (1 + self.transaction_cost_pct)
            
            order = {
                'order_id': str(uuid.uuid4()),
                'symbol': symbol,
                'type': 'buy',
                'shares': shares,
                'price': execution_price,
                'value': total_cost,
                'transaction_cost': shares * execution_price * self.transaction_cost_pct,
                'slippage': shares * execution_price * self.slippage_pct,
                'timestamp': datetime.now().isoformat(),
                'status': 'executed'
            }
            
            return order
            
        except Exception as e:
            logger.error(f"Error executing buy order for {symbol}: {e}")
            return None
    
    def _execute_sell_order(self, symbol: str, order_amount: float, current_price: float, positions: Dict) -> Optional[Dict]:
        """Execute a sell order"""
        try:
            # Check if we have shares to sell
            if symbol not in positions or positions[symbol]['shares'] <= 0:
                logger.debug(f"No shares to sell for {symbol}")
                return None
            
            available_shares = positions[symbol]['shares']
            
            # Calculate order size
            order_value = order_amount * 10000  # Scale to reasonable order size
            
            if order_value < self.min_order_size:
                logger.debug(f"Order too small for {symbol}: ${order_value:.2f}")
                return None
            
            # Calculate shares to sell
            shares_to_sell = int(order_value / current_price)
            shares_to_sell = min(shares_to_sell, available_shares)
            
            if shares_to_sell <= 0:
                return None
            
            # Apply slippage and transaction costs
            execution_price = current_price * (1 - self.slippage_pct)
            total_proceeds = shares_to_sell * execution_price * (1 - self.transaction_cost_pct)
            
            order = {
                'order_id': str(uuid.uuid4()),
                'symbol': symbol,
                'type': 'sell',
                'shares': shares_to_sell,
                'price': execution_price,
                'value': total_proceeds,
                'transaction_cost': shares_to_sell * execution_price * self.transaction_cost_pct,
                'slippage': shares_to_sell * execution_price * self.slippage_pct,
                'timestamp': datetime.now().isoformat(),
                'status': 'executed'
            }
            
            return order
            
        except Exception as e:
            logger.error(f"Error executing sell order for {symbol}: {e}")
            return None
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (placeholder)"""
        # This should be replaced with actual market data
        # For now, return a placeholder price
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'AMZN': 3300.0,
            'TSLA': 700.0,
            'NVDA': 500.0,
            'META': 300.0,
            'NFLX': 500.0,
            'PYPL': 200.0,
            'ADBE': 500.0
        }
        
        return base_prices.get(symbol, 100.0)
    
    def _log_order_execution(self, orders: List[Dict]):
        """Log order execution details"""
        total_value = sum(order['value'] for order in orders)
        total_transaction_cost = sum(order['transaction_cost'] for order in orders)
        total_slippage = sum(order['slippage'] for order in orders)
        
        logger.info(f"Executed {len(orders)} orders:")
        logger.info(f"  Total value: ${total_value:,.2f}")
        logger.info(f"  Transaction costs: ${total_transaction_cost:,.2f}")
        logger.info(f"  Slippage: ${total_slippage:,.2f}")
        
        for order in orders:
            logger.info(f"  {order['type'].upper()} {order['shares']} shares of {order['symbol']} @ ${order['price']:.2f}")
    
    def get_order_statistics(self) -> Dict:
        """Get order execution statistics"""
        if not self.order_history:
            return {}
        
        total_orders = len(self.order_history)
        buy_orders = [order for order in self.order_history if order['type'] == 'buy']
        sell_orders = [order for order in self.order_history if order['type'] == 'sell']
        
        total_value = sum(order['value'] for order in self.order_history)
        total_transaction_cost = sum(order['transaction_cost'] for order in self.order_history)
        total_slippage = sum(order['slippage'] for order in self.order_history)
        
        stats = {
            'total_orders': total_orders,
            'buy_orders': len(buy_orders),
            'sell_orders': len(sell_orders),
            'total_value': total_value,
            'total_transaction_cost': total_transaction_cost,
            'total_slippage': total_slippage,
            'avg_order_size': total_value / total_orders if total_orders > 0 else 0,
            'transaction_cost_ratio': total_transaction_cost / total_value if total_value > 0 else 0,
            'slippage_ratio': total_slippage / total_value if total_value > 0 else 0
        }
        
        return stats
    
    def calculate_execution_quality(self, orders: List[Dict]) -> Dict:
        """Calculate execution quality metrics"""
        if not orders:
            return {}
        
        # Calculate execution quality metrics
        total_slippage = sum(order['slippage'] for order in orders)
        total_transaction_cost = sum(order['transaction_cost'] for order in orders)
        total_value = sum(order['value'] for order in orders)
        
        # Calculate average execution price vs market price
        execution_prices = []
        market_prices = []
        
        for order in orders:
            execution_price = order['price']
            # Market price would be the price without slippage
            market_price = execution_price / (1 + self.slippage_pct) if order['type'] == 'buy' else execution_price / (1 - self.slippage_pct)
            
            execution_prices.append(execution_price)
            market_prices.append(market_price)
        
        avg_execution_price = np.mean(execution_prices)
        avg_market_price = np.mean(market_prices)
        price_improvement = (avg_market_price - avg_execution_price) / avg_market_price
        
        quality_metrics = {
            'total_orders': len(orders),
            'total_value': total_value,
            'total_transaction_cost': total_transaction_cost,
            'total_slippage': total_slippage,
            'avg_execution_price': avg_execution_price,
            'avg_market_price': avg_market_price,
            'price_improvement': price_improvement,
            'transaction_cost_ratio': total_transaction_cost / total_value if total_value > 0 else 0,
            'slippage_ratio': total_slippage / total_value if total_value > 0 else 0,
            'total_cost_ratio': (total_transaction_cost + total_slippage) / total_value if total_value > 0 else 0
        }
        
        return quality_metrics
    
    def validate_order(self, symbol: str, order_type: str, shares: int, price: float) -> Tuple[bool, str]:
        """Validate order parameters"""
        # Check order size
        order_value = shares * price
        
        if order_value < self.min_order_size:
            return False, f"Order value ${order_value:.2f} below minimum ${self.min_order_size}"
        
        if order_value > self.max_order_size:
            return False, f"Order value ${order_value:.2f} above maximum ${self.max_order_size}"
        
        # Check shares
        if shares <= 0:
            return False, "Number of shares must be positive"
        
        # Check price
        if price <= 0:
            return False, "Price must be positive"
        
        return True, "Order is valid"
    
    def estimate_order_cost(self, symbol: str, order_type: str, shares: int, price: float) -> Dict:
        """Estimate order execution costs"""
        # Calculate base value
        base_value = shares * price
        
        # Calculate transaction costs
        transaction_cost = base_value * self.transaction_cost_pct
        
        # Calculate slippage
        slippage = base_value * self.slippage_pct
        
        # Calculate total cost/proceeds
        if order_type == 'buy':
            total_cost = base_value + transaction_cost + slippage
        else:  # sell
            total_cost = base_value - transaction_cost - slippage
        
        return {
            'base_value': base_value,
            'transaction_cost': transaction_cost,
            'slippage': slippage,
            'total_cost': total_cost if order_type == 'buy' else total_cost,
            'total_proceeds': total_cost if order_type == 'sell' else None
        } 