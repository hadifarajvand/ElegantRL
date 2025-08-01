"""
Risk Management System for Paper Trading
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk Management System
    
    Features:
    - Position size limits
    - Stop loss and take profit
    - Leverage limits
    - Cash reserve requirements
    - Portfolio concentration limits
    """
    
    def __init__(self,
                 max_position_size: float = 0.2,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 max_leverage: float = 1.5,
                 min_cash_reserve: float = 0.1,
                 max_portfolio_concentration: float = 0.3):
        
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_leverage = max_leverage
        self.min_cash_reserve = min_cash_reserve
        self.max_portfolio_concentration = max_portfolio_concentration
    
    def validate_action(self, action: np.ndarray, positions: Dict, portfolio_value: float) -> np.ndarray:
        """Validate and adjust trading action based on risk rules"""
        validated_action = action.copy()
        
        # Check position size limits
        validated_action = self._check_position_size_limits(validated_action, positions, portfolio_value)
        
        # Check cash reserve requirements
        validated_action = self._check_cash_reserve(validated_action, positions, portfolio_value)
        
        # Check leverage limits
        validated_action = self._check_leverage_limits(validated_action, positions, portfolio_value)
        
        # Check portfolio concentration
        validated_action = self._check_portfolio_concentration(validated_action, positions, portfolio_value)
        
        return validated_action
    
    def check_positions(self, positions: Dict, portfolio_value: float) -> List[Dict]:
        """Check current positions for risk violations"""
        violations = []
        
        for symbol, position in positions.items():
            # Check stop loss
            if position['unrealized_pnl'] < 0:
                loss_pct = abs(position['unrealized_pnl']) / position['cost_basis']
                if loss_pct >= self.stop_loss_pct:
                    violations.append({
                        'symbol': symbol,
                        'type': 'stop_loss',
                        'severity': 'high',
                        'message': f'Stop loss triggered for {symbol}: {loss_pct:.2%} loss'
                    })
            
            # Check take profit
            if position['unrealized_pnl'] > 0:
                profit_pct = position['unrealized_pnl'] / position['cost_basis']
                if profit_pct >= self.take_profit_pct:
                    violations.append({
                        'symbol': symbol,
                        'type': 'take_profit',
                        'severity': 'medium',
                        'message': f'Take profit target reached for {symbol}: {profit_pct:.2%} profit'
                    })
            
            # Check position size
            position_size_pct = position['current_value'] / portfolio_value
            if position_size_pct > self.max_position_size:
                violations.append({
                    'symbol': symbol,
                    'type': 'position_size',
                    'severity': 'medium',
                    'message': f'Position size limit exceeded for {symbol}: {position_size_pct:.2%}'
                })
        
        # Check portfolio concentration
        total_stock_value = sum(pos['current_value'] for pos in positions.values())
        concentration_pct = total_stock_value / portfolio_value
        if concentration_pct > self.max_portfolio_concentration:
            violations.append({
                'symbol': 'PORTFOLIO',
                'type': 'concentration',
                'severity': 'high',
                'message': f'Portfolio concentration limit exceeded: {concentration_pct:.2%}'
            })
        
        return violations
    
    def _check_position_size_limits(self, action: np.ndarray, positions: Dict, portfolio_value: float) -> np.ndarray:
        """Check and adjust action based on position size limits"""
        adjusted_action = action.copy()
        
        for i, symbol in enumerate(positions.keys()):
            if i < len(action):
                current_position_value = positions[symbol]['current_value']
                current_position_pct = current_position_value / portfolio_value
                
                # Calculate potential new position size
                if action[i] > 0:  # Buying
                    potential_increase = action[i] * portfolio_value * self.max_position_size
                    max_allowed_increase = (self.max_position_size - current_position_pct) * portfolio_value
                    
                    if potential_increase > max_allowed_increase:
                        adjusted_action[i] = max_allowed_increase / portfolio_value
                        logger.warning(f"Position size limit applied to {symbol}")
        
        return adjusted_action
    
    def _check_cash_reserve(self, action: np.ndarray, positions: Dict, portfolio_value: float) -> np.ndarray:
        """Check and adjust action based on cash reserve requirements"""
        adjusted_action = action.copy()
        
        # Calculate current cash
        total_stock_value = sum(pos['current_value'] for pos in positions.values())
        current_cash = portfolio_value - total_stock_value
        
        # Calculate required cash reserve
        required_cash_reserve = portfolio_value * self.min_cash_reserve
        
        # Calculate available cash for trading
        available_cash = current_cash - required_cash_reserve
        
        if available_cash < 0:
            # Need to reduce positions to meet cash reserve
            logger.warning("Cash reserve requirement not met - reducing positions")
            return np.zeros_like(action)
        
        # Limit buy actions based on available cash
        for i in range(len(action)):
            if action[i] > 0:  # Buying
                max_buy_value = available_cash / (1 + 0.001)  # Account for transaction costs
                if action[i] * portfolio_value > max_buy_value:
                    adjusted_action[i] = max_buy_value / portfolio_value
                    logger.warning(f"Cash reserve limit applied to action {i}")
        
        return adjusted_action
    
    def _check_leverage_limits(self, action: np.ndarray, positions: Dict, portfolio_value: float) -> np.ndarray:
        """Check and adjust action based on leverage limits"""
        adjusted_action = action.copy()
        
        # Calculate current leverage
        total_stock_value = sum(pos['current_value'] for pos in positions.values())
        current_leverage = total_stock_value / portfolio_value
        
        if current_leverage >= self.max_leverage:
            # At maximum leverage - only allow selling
            adjusted_action = np.minimum(adjusted_action, 0)
            logger.warning("Maximum leverage reached - only selling allowed")
        
        return adjusted_action
    
    def _check_portfolio_concentration(self, action: np.ndarray, positions: Dict, portfolio_value: float) -> np.ndarray:
        """Check and adjust action based on portfolio concentration limits"""
        adjusted_action = action.copy()
        
        # Calculate current concentration
        total_stock_value = sum(pos['current_value'] for pos in positions.values())
        current_concentration = total_stock_value / portfolio_value
        
        if current_concentration >= self.max_portfolio_concentration:
            # At maximum concentration - only allow selling
            adjusted_action = np.minimum(adjusted_action, 0)
            logger.warning("Maximum portfolio concentration reached - only selling allowed")
        
        return adjusted_action
    
    def calculate_var(self, positions: Dict, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        if not positions:
            return 0.0
        
        # Calculate position returns
        returns = []
        for position in positions.values():
            if position['cost_basis'] > 0:
                return_pct = position['unrealized_pnl'] / position['cost_basis']
                returns.append(return_pct)
        
        if not returns:
            return 0.0
        
        # Calculate VaR
        returns_array = np.array(returns)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_array, var_percentile)
        
        return var
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        portfolio_array = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    def get_risk_metrics(self, positions: Dict, portfolio_value: float) -> Dict:
        """Get comprehensive risk metrics"""
        total_stock_value = sum(pos['current_value'] for pos in positions.values())
        current_cash = portfolio_value - total_stock_value
        
        metrics = {
            'total_portfolio_value': portfolio_value,
            'cash': current_cash,
            'cash_ratio': current_cash / portfolio_value,
            'stock_ratio': total_stock_value / portfolio_value,
            'leverage': total_stock_value / portfolio_value,
            'num_positions': len(positions),
            'var_95': self.calculate_var(positions, 0.95),
            'var_99': self.calculate_var(positions, 0.99),
            'largest_position_pct': 0.0,
            'average_position_pct': 0.0
        }
        
        if positions:
            position_sizes = [pos['current_value'] / portfolio_value for pos in positions.values()]
            metrics['largest_position_pct'] = max(position_sizes)
            metrics['average_position_pct'] = np.mean(position_sizes)
        
        return metrics 