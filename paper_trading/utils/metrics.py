"""
Performance metrics calculation for paper trading system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, portfolio_values: np.ndarray, 
                            returns: np.ndarray = None) -> Dict[str, float]:
        """Calculate all performance metrics"""
        if returns is None:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = self._calculate_total_return(portfolio_values)
        metrics['annualized_return'] = self._calculate_annualized_return(portfolio_values, returns)
        metrics['volatility'] = self._calculate_volatility(returns)
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_values)
        
        # Risk metrics
        metrics['var_95'] = self._calculate_var(returns, 0.95)
        metrics['var_99'] = self._calculate_var(returns, 0.99)
        metrics['cvar_95'] = self._calculate_cvar(returns, 0.95)
        
        # Additional metrics
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(portfolio_values, returns)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        metrics['information_ratio'] = self._calculate_information_ratio(returns)
        
        return metrics
    
    def _calculate_total_return(self, portfolio_values: np.ndarray) -> float:
        """Calculate total return"""
        if len(portfolio_values) < 2:
            return 0.0
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    def _calculate_annualized_return(self, portfolio_values: np.ndarray, 
                                   returns: np.ndarray) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        
        total_return = self._calculate_total_return(portfolio_values)
        days = len(returns)
        return (1 + total_return) ** (252 / days) - 1
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        var_percentile = (1 - confidence_level) * 100
        return np.percentile(returns, var_percentile)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def _calculate_calmar_ratio(self, portfolio_values: np.ndarray, 
                               returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annualized_return = self._calculate_annualized_return(portfolio_values, returns)
        max_drawdown = abs(self._calculate_max_drawdown(portfolio_values))
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / (downside_deviation + 1e-8) * np.sqrt(252)
    
    def _calculate_information_ratio(self, returns: np.ndarray, 
                                   benchmark_returns: np.ndarray = None) -> float:
        """Calculate Information ratio"""
        if benchmark_returns is None:
            # Use zero as benchmark (excess return over cash)
            benchmark_returns = np.zeros_like(returns)
        
        if len(returns) != len(benchmark_returns):
            logger.warning("Returns and benchmark returns have different lengths")
            return 0.0
        
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(active_returns) / tracking_error * np.sqrt(252)


class TradingMetrics:
    """Calculate trading-specific metrics"""
    
    def __init__(self):
        pass
    
    def calculate_trading_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate trading metrics from trade history"""
        if not trades:
            return {}
        
        metrics = {}
        
        # Basic trading metrics
        metrics['total_trades'] = len(trades)
        metrics['winning_trades'] = len([t for t in trades if t.get('pnl', 0) > 0])
        metrics['losing_trades'] = len([t for t in trades if t.get('pnl', 0) < 0])
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        
        # PnL metrics
        pnls = [t.get('pnl', 0) for t in trades]
        metrics['total_pnl'] = sum(pnls)
        metrics['avg_trade_pnl'] = np.mean(pnls)
        metrics['max_profit'] = max(pnls) if pnls else 0
        metrics['max_loss'] = min(pnls) if pnls else 0
        
        # Risk metrics
        if pnls:
            metrics['pnl_std'] = np.std(pnls)
            metrics['profit_factor'] = self._calculate_profit_factor(pnls)
            metrics['avg_win'] = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
            metrics['avg_loss'] = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        
        return metrics
    
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)


class RiskMetrics:
    """Calculate risk metrics"""
    
    def __init__(self):
        pass
    
    def calculate_risk_metrics(self, positions: Dict, portfolio_value: float) -> Dict[str, float]:
        """Calculate risk metrics from current positions"""
        metrics = {}
        
        if not positions:
            return metrics
        
        # Position concentration
        position_values = [pos['current_value'] for pos in positions.values()]
        total_stock_value = sum(position_values)
        
        metrics['num_positions'] = len(positions)
        metrics['largest_position_pct'] = max(position_values) / portfolio_value if portfolio_value > 0 else 0
        metrics['avg_position_pct'] = np.mean(position_values) / portfolio_value if portfolio_value > 0 else 0
        metrics['position_concentration'] = total_stock_value / portfolio_value if portfolio_value > 0 else 0
        
        # Leverage
        metrics['leverage'] = total_stock_value / portfolio_value if portfolio_value > 0 else 0
        
        # Cash metrics
        cash = portfolio_value - total_stock_value
        metrics['cash_pct'] = cash / portfolio_value if portfolio_value > 0 else 0
        
        return metrics 