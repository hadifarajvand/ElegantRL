"""
Performance Analyzer for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies
    
    Features:
    - Performance metrics calculation
    - Risk analysis
    - Trade analysis
    - Portfolio attribution
    - Benchmark comparison
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def analyze_performance(self, portfolio_values: np.ndarray, 
                          trades: List[Dict] = None,
                          benchmark_values: np.ndarray = None) -> Dict:
        """Comprehensive performance analysis"""
        analysis = {}
        
        # Basic performance metrics
        analysis['basic_metrics'] = self._calculate_basic_metrics(portfolio_values)
        
        # Risk metrics
        analysis['risk_metrics'] = self._calculate_risk_metrics(portfolio_values)
        
        # Trading metrics (if trades available)
        if trades:
            analysis['trading_metrics'] = self._calculate_trading_metrics(trades)
        
        # Benchmark comparison (if benchmark available)
        if benchmark_values is not None:
            analysis['benchmark_comparison'] = self._compare_with_benchmark(
                portfolio_values, benchmark_values
            )
        
        # Drawdown analysis
        analysis['drawdown_analysis'] = self._analyze_drawdowns(portfolio_values)
        
        # Rolling metrics
        analysis['rolling_metrics'] = self._calculate_rolling_metrics(portfolio_values)
        
        return analysis
    
    def _calculate_basic_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """Calculate basic performance metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = {
            'initial_value': portfolio_values[0],
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            'total_return_pct': ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100,
            'annualized_return': self._calculate_annualized_return(portfolio_values, returns),
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_values, returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'max_drawdown_pct': self._calculate_max_drawdown(portfolio_values) * 100
        }
        
        return metrics
    
    def _calculate_risk_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """Calculate risk metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        metrics = {
            'var_95': self._calculate_var(returns, 0.95),
            'var_99': self._calculate_var(returns, 0.99),
            'cvar_95': self._calculate_cvar(returns, 0.95),
            'cvar_99': self._calculate_cvar(returns, 0.99),
            'downside_deviation': self._calculate_downside_deviation(returns),
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns),
            'var_ratio': self._calculate_var_ratio(returns),
            'gain_loss_ratio': self._calculate_gain_loss_ratio(returns)
        }
        
        return metrics
    
    def _calculate_trading_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trading-specific metrics"""
        if not trades:
            return {}
        
        # Extract PnL from trades
        pnls = [trade.get('pnl', 0) for trade in trades]
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len([p for p in pnls if p > 0]),
            'losing_trades': len([p for p in pnls if p < 0]),
            'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'avg_trade_pnl': np.mean(pnls) if pnls else 0,
            'max_profit': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'profit_factor': self._calculate_profit_factor(pnls),
            'avg_win': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            'largest_win': max(pnls) if pnls else 0,
            'largest_loss': min(pnls) if pnls else 0,
            'pnl_std': np.std(pnls) if pnls else 0
        }
        
        return metrics
    
    def _compare_with_benchmark(self, portfolio_values: np.ndarray, 
                               benchmark_values: np.ndarray) -> Dict:
        """Compare portfolio performance with benchmark"""
        if len(portfolio_values) != len(benchmark_values):
            logger.warning("Portfolio and benchmark have different lengths")
            return {}
        
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        metrics = {
            'information_ratio': self._calculate_information_ratio(excess_returns),
            'tracking_error': np.std(excess_returns) * np.sqrt(252),
            'beta': self._calculate_beta(portfolio_returns, benchmark_returns),
            'alpha': self._calculate_alpha(portfolio_returns, benchmark_returns),
            'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0, 1],
            'excess_return': np.mean(excess_returns) * 252,
            'excess_return_pct': np.mean(excess_returns) * 252 * 100
        }
        
        return metrics
    
    def _analyze_drawdowns(self, portfolio_values: np.ndarray) -> Dict:
        """Analyze drawdown periods"""
        if len(portfolio_values) < 2:
            return {}
        
        # Calculate drawdown series
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'duration': i - start_idx,
                    'max_drawdown': np.min(drawdown[start_idx:i])
                })
        
        # If still in drawdown at the end
        if in_drawdown:
            drawdown_periods.append({
                'start_idx': start_idx,
                'end_idx': len(drawdown) - 1,
                'duration': len(drawdown) - 1 - start_idx,
                'max_drawdown': np.min(drawdown[start_idx:])
            })
        
        analysis = {
            'max_drawdown': np.min(drawdown),
            'max_drawdown_pct': np.min(drawdown) * 100,
            'num_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': np.mean([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'max_drawdown_duration': max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'drawdown_periods': drawdown_periods
        }
        
        return analysis
    
    def _calculate_rolling_metrics(self, portfolio_values: np.ndarray, 
                                 window: int = 252) -> Dict:
        """Calculate rolling performance metrics"""
        if len(portfolio_values) < window:
            return {}
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Rolling metrics
        rolling_returns = pd.Series(returns).rolling(window=window).mean() * 252
        rolling_vol = pd.Series(returns).rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        
        # Rolling drawdown
        rolling_max = pd.Series(portfolio_values).rolling(window=window).max()
        rolling_drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
        
        metrics = {
            'rolling_returns': rolling_returns.tolist(),
            'rolling_volatility': rolling_vol.tolist(),
            'rolling_sharpe': rolling_sharpe.tolist(),
            'rolling_drawdown': rolling_drawdown.tolist(),
            'avg_rolling_return': rolling_returns.mean(),
            'avg_rolling_volatility': rolling_vol.mean(),
            'avg_rolling_sharpe': rolling_sharpe.mean(),
            'min_rolling_sharpe': rolling_sharpe.min(),
            'max_rolling_sharpe': rolling_sharpe.max()
        }
        
        return metrics
    
    # Helper methods for metric calculations
    def _calculate_annualized_return(self, portfolio_values: np.ndarray, returns: np.ndarray) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        days = len(returns)
        return (1 + total_return) ** (252 / days) - 1
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
    
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
    
    def _calculate_calmar_ratio(self, portfolio_values: np.ndarray, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annualized_return = self._calculate_annualized_return(portfolio_values, returns)
        max_drawdown = abs(self._calculate_max_drawdown(portfolio_values))
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
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
        """Calculate Conditional Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def _calculate_downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation"""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        return np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness"""
        if len(returns) == 0:
            return 0.0
        
        return ((returns - np.mean(returns)) ** 3).mean() / (np.std(returns) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis"""
        if len(returns) == 0:
            return 0.0
        
        return ((returns - np.mean(returns)) ** 4).mean() / (np.std(returns) ** 4) - 3
    
    def _calculate_var_ratio(self, returns: np.ndarray) -> float:
        """Calculate VaR ratio"""
        if len(returns) == 0:
            return 0.0
        
        var_95 = self._calculate_var(returns, 0.95)
        return np.mean(returns) / abs(var_95) if var_95 != 0 else 0.0
    
    def _calculate_gain_loss_ratio(self, returns: np.ndarray) -> float:
        """Calculate gain/loss ratio"""
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
        
        return avg_gain / avg_loss if avg_loss != 0 else 0.0
    
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor"""
        if not pnls:
            return 0.0
        
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    
    def _calculate_information_ratio(self, excess_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        if len(excess_returns) == 0:
            return 0.0
        
        tracking_error = np.std(excess_returns)
        return np.mean(excess_returns) / (tracking_error + 1e-8) * np.sqrt(252)
    
    def _calculate_beta(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta"""
        if len(portfolio_returns) != len(benchmark_returns):
            return 0.0
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 0.0
    
    def _calculate_alpha(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate alpha"""
        if len(portfolio_returns) != len(benchmark_returns):
            return 0.0
        
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        return np.mean(portfolio_returns) - beta * np.mean(benchmark_returns) 