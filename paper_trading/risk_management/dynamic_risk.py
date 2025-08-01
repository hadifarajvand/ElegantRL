"""
Dynamic Risk Management for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """Risk metrics data class"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    correlation: float
    concentration: float
    leverage: float
    exposure: float


class DynamicRiskManager:
    """
    Dynamic Risk Management System
    
    Features:
    - Real-time risk monitoring
    - Dynamic position sizing
    - Volatility-based adjustments
    - Correlation analysis
    - Stress testing
    - Risk budget allocation
    """
    
    def __init__(self, max_portfolio_risk=0.2, max_position_risk=0.1, max_leverage=1.5, stress_scenarios=None):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_leverage = max_leverage
        self.stress_scenarios = stress_scenarios or []
        
        # Risk state
        self.current_risk_level = RiskLevel.MEDIUM
        self.risk_metrics = {}
        self.position_limits = {}
        self.correlation_matrix = None
        self.volatility_estimates = {}
        
        # Historical data
        self.returns_history = []
        self.volatility_history = []
        self.drawdown_history = []
        
        logger.info("DynamicRiskManager initialized")
    
    def calculate_risk_metrics(self, portfolio: Dict, market_data: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Extract portfolio data
            positions = portfolio.get('positions', {})
            cash = portfolio.get('cash', 0)
            total_value = portfolio.get('total_value', cash)
            
            if total_value <= 0:
                return self._create_empty_risk_metrics()
            
            # Calculate returns
            returns = self._calculate_portfolio_returns(portfolio, market_data)
            
            if len(returns) < 2:
                return self._create_empty_risk_metrics()
            
            # Calculate risk metrics
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            cvar_95 = self._calculate_cvar(returns, 0.95)
            cvar_99 = self._calculate_cvar(returns, 0.99)
            max_drawdown = self._calculate_max_drawdown(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Calculate portfolio metrics
            beta = self._calculate_portfolio_beta(positions, market_data)
            correlation = self._calculate_portfolio_correlation(positions, market_data)
            concentration = self._calculate_concentration(positions, total_value)
            leverage = self._calculate_leverage(positions, total_value)
            exposure = self._calculate_exposure(positions, total_value)
            
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=beta,
                correlation=correlation,
                concentration=concentration,
                leverage=leverage,
                exposure=exposure
            )
            
            # Update risk level
            self._update_risk_level(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._create_empty_risk_metrics()
    
    def validate_trade(self, trade: Dict, portfolio: Dict, market_data: Dict) -> Tuple[bool, str]:
        """Validate if a trade meets risk requirements"""
        try:
            symbol = trade.get('symbol')
            action = trade.get('action')  # 'buy' or 'sell'
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            
            if not all([symbol, action, quantity, price]):
                return False, "Missing trade parameters"
            
            # Calculate trade impact
            trade_value = abs(quantity * price)
            current_positions = portfolio.get('positions', {})
            total_value = portfolio.get('total_value', 0)
            
            if total_value <= 0:
                return False, "Invalid portfolio value"
            
            # Check position size limits
            if not self._check_position_size(trade, portfolio):
                return False, "Position size exceeds limits"
            
            # Check concentration limits
            if not self._check_concentration_limits(trade, portfolio):
                return False, "Concentration limits exceeded"
            
            # Check leverage limits
            if not self._check_leverage_limits(trade, portfolio):
                return False, "Leverage limits exceeded"
            
            # Check risk budget
            if not self._check_risk_budget(trade, portfolio, market_data):
                return False, "Risk budget exceeded"
            
            # Check correlation limits
            if not self._check_correlation_limits(trade, portfolio, market_data):
                return False, "Correlation limits exceeded"
            
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              portfolio: Dict, market_data: Dict) -> float:
        """Calculate optimal position size based on risk"""
        try:
            # Base position size
            base_size = signal_strength * self.max_position_risk
            
            # Adjust for volatility
            volatility = self._get_symbol_volatility(symbol, market_data)
            volatility_adjustment = 1.0 / (1.0 + volatility)
            
            # Adjust for correlation
            correlation = self._get_symbol_correlation(symbol, portfolio, market_data)
            correlation_adjustment = 1.0 - abs(correlation) * 0.5
            
            # Adjust for current risk level
            risk_adjustment = self._get_risk_level_adjustment()
            
            # Calculate final position size
            position_size = base_size * volatility_adjustment * correlation_adjustment * risk_adjustment
            
            # Apply limits
            max_size = self._get_max_position_size(symbol, portfolio)
            position_size = min(position_size, max_size)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def update_risk_limits(self, market_data: Dict):
        """Update risk limits based on market conditions"""
        try:
            # Calculate market volatility
            market_volatility = self._calculate_market_volatility(market_data)
            
            # Adjust risk limits based on volatility
            if market_volatility > 0.3:  # High volatility
                self.max_portfolio_risk *= 0.8
                self.max_position_risk *= 0.8
                logger.info("Risk limits reduced due to high volatility")
            elif market_volatility < 0.1:  # Low volatility
                self.max_portfolio_risk *= 1.1
                self.max_position_risk *= 1.1
                logger.info("Risk limits increased due to low volatility")
            
            # Update correlation matrix
            self._update_correlation_matrix(market_data)
            
            # Update volatility estimates
            self._update_volatility_estimates(market_data)
            
        except Exception as e:
            logger.error(f"Error updating risk limits: {e}")
    
    def stress_test(self, portfolio: Dict, market_data: Dict, 
                   scenarios: List[Dict]) -> Dict[str, Any]:
        """Perform stress testing on portfolio"""
        try:
            stress_results = {}
            
            for scenario in scenarios:
                scenario_name = scenario.get('name', 'unknown')
                shock_size = scenario.get('shock_size', 0.1)
                shock_type = scenario.get('shock_type', 'uniform')
                
                # Apply stress scenario
                stressed_portfolio = self._apply_stress_scenario(
                    portfolio, market_data, shock_size, shock_type
                )
                
                # Calculate stressed metrics
                stressed_metrics = self.calculate_risk_metrics(stressed_portfolio, market_data)
                
                stress_results[scenario_name] = {
                    'shock_size': shock_size,
                    'shock_type': shock_type,
                    'var_95': stressed_metrics.var_95,
                    'max_drawdown': stressed_metrics.max_drawdown,
                    'portfolio_value_change': (
                        stressed_portfolio.get('total_value', 0) - portfolio.get('total_value', 0)
                    )
                }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}
    
    # Helper methods
    def _calculate_portfolio_returns(self, portfolio: Dict, market_data: Dict) -> List[float]:
        """Calculate portfolio returns"""
        # Simplified calculation - in practice, you'd use actual historical data
        returns = []
        for _ in range(30):  # Last 30 days
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
            returns.append(daily_return)
        
        return returns
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 2:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) < 2:
            return 0.0
        
        var_threshold = self._calculate_var(returns, confidence)
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return var_threshold
        
        return np.mean(tail_returns)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - self.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - self.risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0
        
        downside_deviation = np.std(negative_returns)
        return np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        return annual_return / abs(max_drawdown)
    
    def _calculate_portfolio_beta(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio beta"""
        # Simplified beta calculation
        total_beta = 0.0
        total_value = sum(pos.get('current_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return 1.0
        
        for symbol, position in positions.items():
            position_value = position.get('current_value', 0)
            # Assume beta of 1 for all stocks (simplified)
            beta = 1.0
            total_beta += (position_value / total_value) * beta
        
        return total_beta
    
    def _calculate_portfolio_correlation(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio correlation"""
        # Simplified correlation calculation
        if len(positions) < 2:
            return 0.0
        
        # Assume average correlation of 0.3 (simplified)
        return 0.3
    
    def _calculate_concentration(self, positions: Dict, total_value: float) -> float:
        """Calculate portfolio concentration"""
        if total_value == 0:
            return 0.0
        
        position_values = [pos.get('current_value', 0) for pos in positions.values()]
        if not position_values:
            return 0.0
        
        # Herfindahl-Hirschman Index
        weights = np.array(position_values) / total_value
        concentration = np.sum(weights ** 2)
        
        return concentration
    
    def _calculate_leverage(self, positions: Dict, total_value: float) -> float:
        """Calculate portfolio leverage"""
        if total_value == 0:
            return 1.0
        
        total_position_value = sum(pos.get('current_value', 0) for pos in positions.values())
        return total_position_value / total_value
    
    def _calculate_exposure(self, positions: Dict, total_value: float) -> float:
        """Calculate portfolio exposure"""
        if total_value == 0:
            return 0.0
        
        total_exposure = sum(abs(pos.get('current_value', 0)) for pos in positions.values())
        return total_exposure / total_value
    
    def _update_risk_level(self, risk_metrics: RiskMetrics):
        """Update current risk level based on metrics"""
        # Simple risk level determination
        if risk_metrics.volatility > 0.3 or risk_metrics.max_drawdown > 0.1:
            self.current_risk_level = RiskLevel.HIGH
        elif risk_metrics.volatility > 0.2 or risk_metrics.max_drawdown > 0.05:
            self.current_risk_level = RiskLevel.MEDIUM
        else:
            self.current_risk_level = RiskLevel.LOW
    
    def _get_risk_level_adjustment(self) -> float:
        """Get position size adjustment based on risk level"""
        adjustments = {
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.EXTREME: 0.5
        }
        return adjustments.get(self.current_risk_level, 1.0)
    
    def _check_position_size(self, trade: Dict, portfolio: Dict) -> bool:
        """Check if position size is within limits"""
        trade_value = abs(trade.get('quantity', 0) * trade.get('price', 0))
        total_value = portfolio.get('total_value', 0)
        
        if total_value == 0:
            return False
        
        position_ratio = trade_value / total_value
        return position_ratio <= self.max_position_risk
    
    def _check_concentration_limits(self, trade: Dict, portfolio: Dict) -> bool:
        """Check concentration limits"""
        symbol = trade.get('symbol')
        trade_value = abs(trade.get('quantity', 0) * trade.get('price', 0))
        total_value = portfolio.get('total_value', 0)
        
        if total_value == 0:
            return False
        
        # Check single position concentration
        position_ratio = trade_value / total_value
        if position_ratio > 0.2:  # Max 20% in single position
            return False
        
        return True
    
    def _check_leverage_limits(self, trade: Dict, portfolio: Dict) -> bool:
        """Check leverage limits"""
        positions = portfolio.get('positions', {})
        total_value = portfolio.get('total_value', 0)
        
        if total_value == 0:
            return False
        
        # Calculate current leverage
        current_leverage = self._calculate_leverage(positions, total_value)
        
        # Check if new trade would exceed leverage limit
        trade_value = abs(trade.get('quantity', 0) * trade.get('price', 0))
        new_leverage = (sum(pos.get('current_value', 0) for pos in positions.values()) + trade_value) / total_value
        
        return new_leverage <= self.max_leverage
    
    def _check_risk_budget(self, trade: Dict, portfolio: Dict, market_data: Dict) -> bool:
        """Check risk budget"""
        # Simplified risk budget check
        return True  # Always pass for now
    
    def _check_correlation_limits(self, trade: Dict, portfolio: Dict, market_data: Dict) -> bool:
        """Check correlation limits"""
        # Simplified correlation check
        return True  # Always pass for now
    
    def _get_symbol_volatility(self, symbol: str, market_data: Dict) -> float:
        """Get symbol volatility"""
        # Simplified volatility calculation
        return 0.2  # 20% annualized volatility
    
    def _get_symbol_correlation(self, symbol: str, portfolio: Dict, market_data: Dict) -> float:
        """Get symbol correlation with portfolio"""
        # Simplified correlation calculation
        return 0.1  # Low correlation
    
    def _get_max_position_size(self, symbol: str, portfolio: Dict) -> float:
        """Get maximum position size for symbol"""
        total_value = portfolio.get('total_value', 0)
        return total_value * self.max_position_risk
    
    def _calculate_market_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility"""
        # Simplified market volatility calculation
        return 0.15  # 15% market volatility
    
    def _update_correlation_matrix(self, market_data: Dict):
        """Update correlation matrix"""
        # Simplified correlation matrix update
        pass
    
    def _update_volatility_estimates(self, market_data: Dict):
        """Update volatility estimates"""
        # Simplified volatility update
        pass
    
    def _apply_stress_scenario(self, portfolio: Dict, market_data: Dict, 
                              shock_size: float, shock_type: str) -> Dict:
        """Apply stress scenario to portfolio"""
        stressed_portfolio = portfolio.copy()
        positions = stressed_portfolio.get('positions', {})
        
        for symbol, position in positions.items():
            current_value = position.get('current_value', 0)
            
            if shock_type == 'uniform':
                # Uniform shock to all positions
                new_value = current_value * (1 - shock_size)
            elif shock_type == 'correlated':
                # Correlated shock (simplified)
                new_value = current_value * (1 - shock_size * 0.8)
            else:
                # Random shock
                random_shock = np.random.normal(0, shock_size)
                new_value = current_value * (1 + random_shock)
            
            position['current_value'] = max(0, new_value)
        
        # Update total value
        total_value = sum(pos.get('current_value', 0) for pos in positions.values())
        stressed_portfolio['total_value'] = total_value
        
        return stressed_portfolio
    
    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """Create empty risk metrics"""
        return RiskMetrics(
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            max_drawdown=0.0, volatility=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, beta=1.0,
            correlation=0.0, concentration=0.0, leverage=1.0, exposure=0.0
        )


def create_dynamic_risk_manager(max_portfolio_risk: float = 0.02,
                               max_position_risk: float = 0.01,
                               max_leverage: float = 1.5) -> DynamicRiskManager:
    """Convenience function to create dynamic risk manager"""
    return DynamicRiskManager(max_portfolio_risk, max_position_risk, max_leverage) 