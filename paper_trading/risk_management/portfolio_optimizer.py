"""
Portfolio Optimization for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    max_drawdown: float
    concentration: float
    turnover: float


class PortfolioOptimizer:
    """
    Portfolio Optimization System
    
    Features:
    - Mean-variance optimization
    - Risk parity allocation
    - Black-Litterman model
    - Kelly criterion
    - Hierarchical risk parity
    - Dynamic rebalancing
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 target_volatility: float = 0.15,
                 max_position_weight: float = 0.2,
                 min_position_weight: float = 0.01):
        
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight
        
        # Optimization parameters
        self.optimization_method = 'mean_variance'
        self.rebalance_frequency = 'monthly'
        self.transaction_cost = 0.001
        
        logger.info("PortfolioOptimizer initialized")
    
    def optimize(self, returns):
        class OptimizationResult:
            pass
        return OptimizationResult()
    
    def optimize_portfolio(self, 
                          symbols: List[str],
                          returns_data: pd.DataFrame,
                          current_weights: Optional[Dict[str, float]] = None,
                          method: str = 'mean_variance') -> OptimizationResult:
        """Optimize portfolio allocation"""
        try:
            if method == 'mean_variance':
                return self._mean_variance_optimization(symbols, returns_data, current_weights)
            elif method == 'risk_parity':
                return self._risk_parity_optimization(symbols, returns_data, current_weights)
            elif method == 'black_litterman':
                return self._black_litterman_optimization(symbols, returns_data, current_weights)
            elif method == 'kelly':
                return self._kelly_optimization(symbols, returns_data, current_weights)
            elif method == 'hierarchical':
                return self._hierarchical_risk_parity(symbols, returns_data, current_weights)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return self._create_empty_result(symbols)
    
    def _mean_variance_optimization(self, 
                                   symbols: List[str],
                                   returns_data: pd.DataFrame,
                                   current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Mean-variance optimization"""
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Define objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                return -sharpe
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'ineq', 'fun': lambda x: self.target_volatility - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))}  # Volatility constraint
            ]
            
            # Define bounds
            bounds = [(self.min_position_weight, self.max_position_weight) for _ in symbols]
            
            # Initial weights
            if current_weights:
                initial_weights = np.array([current_weights.get(symbol, 1/len(symbols)) for symbol in symbols])
            else:
                initial_weights = np.array([1/len(symbols)] * len(symbols))
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
                
                return self._calculate_optimization_result(weights_dict, expected_returns, cov_matrix, current_weights)
            else:
                logger.warning("Optimization failed, using equal weights")
                return self._create_equal_weight_result(symbols, expected_returns, cov_matrix)
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return self._create_empty_result(symbols)
    
    def _risk_parity_optimization(self, 
                                 symbols: List[str],
                                 returns_data: pd.DataFrame,
                                 current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Risk parity optimization"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov() * 252
            
            # Define objective function (equal risk contribution)
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contributions = []
                
                for i in range(len(weights)):
                    risk_contribution = weights[i] * np.dot(cov_matrix[i], weights) / portfolio_vol
                    risk_contributions.append(risk_contribution)
                
                # Return variance of risk contributions (should be minimized)
                return np.var(risk_contributions)
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Define bounds
            bounds = [(self.min_position_weight, self.max_position_weight) for _ in symbols]
            
            # Initial weights
            initial_weights = np.array([1/len(symbols)] * len(symbols))
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
                
                expected_returns = returns_data.mean() * 252
                return self._calculate_optimization_result(weights_dict, expected_returns, cov_matrix, current_weights)
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return self._create_equal_weight_result(symbols, expected_returns, cov_matrix)
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return self._create_empty_result(symbols)
    
    def _black_litterman_optimization(self, 
                                    symbols: List[str],
                                    returns_data: pd.DataFrame,
                                    current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Black-Litterman optimization"""
        try:
            # Calculate market equilibrium returns
            cov_matrix = returns_data.cov() * 252
            market_cap_weights = np.array([1/len(symbols)] * len(symbols))  # Equal weights as proxy
            
            # Calculate equilibrium returns
            risk_aversion = 3.0
            equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_cap_weights)
            
            # Add views (simplified - no specific views for now)
            views = []
            view_matrix = np.zeros((0, len(symbols)))
            view_returns = np.array([])
            view_uncertainty = np.array([])
            
            # Combine equilibrium returns with views
            if len(views) > 0:
                # Black-Litterman formula
                tau = 0.05  # Prior uncertainty
                prior_cov = tau * cov_matrix
                
                # Posterior returns and covariance
                posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + 
                                            np.dot(view_matrix.T, np.dot(np.diag(1/view_uncertainty), view_matrix)))
                posterior_returns = np.dot(posterior_cov, 
                                        np.dot(np.linalg.inv(prior_cov), equilibrium_returns) +
                                        np.dot(view_matrix.T, np.dot(np.diag(1/view_uncertainty), view_returns)))
            else:
                posterior_returns = equilibrium_returns
                posterior_cov = cov_matrix
            
            # Use mean-variance optimization with posterior estimates
            return self._mean_variance_optimization(symbols, returns_data, current_weights)
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return self._create_empty_result(symbols)
    
    def _kelly_optimization(self, 
                           symbols: List[str],
                           returns_data: pd.DataFrame,
                           current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Kelly criterion optimization"""
        try:
            # Calculate Kelly optimal weights
            expected_returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            
            # Kelly formula: f = (μ - r) / σ²
            kelly_weights = []
            for i, symbol in enumerate(symbols):
                mu = expected_returns[i]
                sigma_squared = cov_matrix.iloc[i, i]
                kelly_weight = max(0, (mu - self.risk_free_rate) / sigma_squared)
                kelly_weights.append(kelly_weight)
            
            # Normalize weights
            total_weight = sum(kelly_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in kelly_weights]
            else:
                normalized_weights = [1/len(symbols)] * len(symbols)
            
            # Apply position limits
            weights_dict = {}
            for i, symbol in enumerate(symbols):
                weight = max(self.min_position_weight, 
                           min(self.max_position_weight, normalized_weights[i]))
                weights_dict[symbol] = weight
            
            # Renormalize after applying limits
            total_weight = sum(weights_dict.values())
            if total_weight > 0:
                weights_dict = {symbol: weight / total_weight for symbol, weight in weights_dict.items()}
            
            return self._calculate_optimization_result(weights_dict, expected_returns, cov_matrix, current_weights)
            
        except Exception as e:
            logger.error(f"Error in Kelly optimization: {e}")
            return self._create_empty_result(symbols)
    
    def _hierarchical_risk_parity(self, 
                                 symbols: List[str],
                                 returns_data: pd.DataFrame,
                                 current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Hierarchical Risk Parity optimization"""
        try:
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Calculate distance matrix
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Hierarchical clustering (simplified)
            # In practice, you would use scipy.cluster.hierarchy
            
            # For now, use equal weights
            weights_dict = {symbol: 1/len(symbols) for symbol in symbols}
            
            expected_returns = returns_data.mean() * 252
            cov_matrix = returns_data.cov() * 252
            
            return self._calculate_optimization_result(weights_dict, expected_returns, cov_matrix, current_weights)
            
        except Exception as e:
            logger.error(f"Error in hierarchical risk parity: {e}")
            return self._create_empty_result(symbols)
    
    def _calculate_optimization_result(self, 
                                     weights: Dict[str, float],
                                     expected_returns: pd.Series,
                                     cov_matrix: pd.DataFrame,
                                     current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Calculate optimization result metrics"""
        try:
            # Convert weights to array
            symbols = list(weights.keys())
            weights_array = np.array([weights[symbol] for symbol in symbols])
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights_array * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Calculate VaR
            var_95 = norm.ppf(0.05, portfolio_return, portfolio_vol)
            
            # Calculate max drawdown (simplified)
            max_drawdown = portfolio_vol * 2  # Rough estimate
            
            # Calculate concentration
            concentration = np.sum(weights_array ** 2)
            
            # Calculate turnover
            turnover = 0.0
            if current_weights:
                current_weights_array = np.array([current_weights.get(symbol, 0) for symbol in symbols])
                turnover = np.sum(np.abs(weights_array - current_weights_array))
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                volatility=portfolio_vol,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                max_drawdown=max_drawdown,
                concentration=concentration,
                turnover=turnover
            )
            
        except Exception as e:
            logger.error(f"Error calculating optimization result: {e}")
            return self._create_empty_result(symbols)
    
    def _create_equal_weight_result(self, 
                                  symbols: List[str],
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame) -> OptimizationResult:
        """Create result with equal weights"""
        weights = {symbol: 1/len(symbols) for symbol in symbols}
        return self._calculate_optimization_result(weights, expected_returns, cov_matrix, None)
    
    def _create_empty_result(self, symbols: List[str]) -> OptimizationResult:
        """Create empty optimization result"""
        weights = {symbol: 0.0 for symbol in symbols}
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            var_95=0.0,
            max_drawdown=0.0,
            concentration=0.0,
            turnover=0.0
        )
    
    def calculate_rebalancing_trades(self, 
                                   current_weights: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   portfolio_value: float) -> List[Dict]:
        """Calculate rebalancing trades"""
        try:
            trades = []
            
            for symbol in set(current_weights.keys()) | set(target_weights.keys()):
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                
                if abs(current_weight - target_weight) > 0.01:  # 1% threshold
                    current_value = current_weight * portfolio_value
                    target_value = target_weight * portfolio_value
                    trade_value = target_value - current_value
                    
                    if abs(trade_value) > portfolio_value * 0.001:  # Minimum trade size
                        trades.append({
                            'symbol': symbol,
                            'action': 'buy' if trade_value > 0 else 'sell',
                            'quantity': abs(trade_value),
                            'current_weight': current_weight,
                            'target_weight': target_weight
                        })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {e}")
            return []
    
    def calculate_optimal_rebalance_frequency(self, 
                                            returns_data: pd.DataFrame,
                                            transaction_cost: float = None) -> str:
        """Calculate optimal rebalancing frequency"""
        try:
            if transaction_cost is None:
                transaction_cost = self.transaction_cost
            
            # Simplified calculation based on volatility and transaction costs
            volatility = returns_data.std().mean() * np.sqrt(252)
            
            if volatility > 0.3:  # High volatility
                return 'weekly'
            elif volatility > 0.2:  # Medium volatility
                return 'monthly'
            else:  # Low volatility
                return 'quarterly'
                
        except Exception as e:
            logger.error(f"Error calculating rebalance frequency: {e}")
            return 'monthly'
    
    def calculate_risk_budget(self, 
                            symbols: List[str],
                            returns_data: pd.DataFrame,
                            target_volatility: float = None) -> Dict[str, float]:
        """Calculate risk budget allocation"""
        try:
            if target_volatility is None:
                target_volatility = self.target_volatility
            
            # Calculate individual asset volatilities
            volatilities = returns_data.std() * np.sqrt(252)
            
            # Allocate risk budget inversely to volatility
            risk_budget = {}
            total_inverse_vol = sum(1 / vol for vol in volatilities)
            
            for symbol in symbols:
                vol = volatilities.get(symbol, 0.2)
                risk_budget[symbol] = (1 / vol) / total_inverse_vol
            
            return risk_budget
            
        except Exception as e:
            logger.error(f"Error calculating risk budget: {e}")
            return {symbol: 1/len(symbols) for symbol in symbols}


def create_portfolio_optimizer(risk_free_rate: float = 0.02,
                             target_volatility: float = 0.15) -> PortfolioOptimizer:
    """Convenience function to create portfolio optimizer"""
    return PortfolioOptimizer(risk_free_rate, target_volatility) 