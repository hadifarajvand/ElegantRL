"""
Backtesting Engine for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from ..models.trading_env import EnhancedStockTradingEnv
from ..utils.metrics import PerformanceMetrics, TradingMetrics, RiskMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting Engine for evaluating trading strategies
    
    Features:
    - Historical data simulation
    - Performance analysis
    - Risk metrics calculation
    - Trade analysis
    - Portfolio tracking
    """
    
    def __init__(self, 
                 data: Dict[str, pd.DataFrame],
                 initial_capital: float = 1000000.0,
                 trading_config: Dict = None):
        
        self.data = data
        self.initial_capital = initial_capital
        self.trading_config = trading_config or {}
        
        # Initialize metrics calculators
        self.performance_metrics = PerformanceMetrics()
        self.trading_metrics = TradingMetrics()
        self.risk_metrics = RiskMetrics()
        
        # Results storage
        self.portfolio_values = []
        self.trades = []
        self.positions_history = []
        self.performance_summary = {}
    
    def run_backtest(self, model=None, start_date: str = None, end_date: str = None) -> Dict:
        """Run backtest with given model"""
        logger.info("Starting backtest...")
        
        # Create environment
        env = EnhancedStockTradingEnv(
            data=self.data,
            initial_capital=self.initial_capital,
            **self.trading_config
        )
        
        # Run simulation
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # Get action from model or random
            if model is not None:
                action = self._get_model_action(model, state)
            else:
                action = env.action_space.sample()
            
            # Take step
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Store portfolio value
            self.portfolio_values.append(info['total_asset'])
            
            # Log progress
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}: Portfolio Value = ${info['total_asset']:,.2f}")
            
            if done or truncated:
                break
        
        # Calculate performance metrics
        self._calculate_performance_metrics(env)
        
        logger.info("Backtest completed!")
        return self.performance_summary
    
    def _get_model_action(self, model, state: np.ndarray) -> np.ndarray:
        """Get action from model (placeholder implementation)"""
        # This should be implemented based on your specific model format
        # For now, return random action
        return np.random.randn(model.action_dim) if hasattr(model, 'action_dim') else np.random.randn(10)
    
    def _calculate_performance_metrics(self, env):
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_values:
            return
        
        portfolio_array = np.array(self.portfolio_values)
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        # Performance metrics
        perf_metrics = self.performance_metrics.calculate_all_metrics(portfolio_array, returns)
        
        # Trading metrics (if available)
        trade_metrics = self.trading_metrics.calculate_trading_metrics(self.trades)
        
        # Risk metrics
        final_positions = env.current_positions if hasattr(env, 'current_positions') else {}
        risk_metrics = self.risk_metrics.calculate_risk_metrics(final_positions, portfolio_array[-1])
        
        # Combine all metrics
        self.performance_summary = {
            'performance': perf_metrics,
            'trading': trade_metrics,
            'risk': risk_metrics,
            'portfolio_final_value': portfolio_array[-1],
            'total_return_pct': perf_metrics.get('total_return', 0) * 100,
            'sharpe_ratio': perf_metrics.get('sharpe_ratio', 0),
            'max_drawdown_pct': perf_metrics.get('max_drawdown', 0) * 100
        }
    
    def get_results_summary(self) -> Dict:
        """Get formatted results summary"""
        if not self.performance_summary:
            return {}
        
        summary = {
            'Backtest Results': {
                'Initial Capital': f"${self.initial_capital:,.2f}",
                'Final Portfolio Value': f"${self.performance_summary.get('portfolio_final_value', 0):,.2f}",
                'Total Return': f"{self.performance_summary.get('total_return_pct', 0):.2f}%",
                'Sharpe Ratio': f"{self.performance_summary.get('sharpe_ratio', 0):.3f}",
                'Max Drawdown': f"{self.performance_summary.get('max_drawdown_pct', 0):.2f}%"
            }
        }
        
        # Add detailed metrics if available
        if 'performance' in self.performance_summary:
            perf = self.performance_summary['performance']
            summary['Performance Metrics'] = {
                'Annualized Return': f"{perf.get('annualized_return', 0):.2%}",
                'Volatility': f"{perf.get('volatility', 0):.2%}",
                'Sortino Ratio': f"{perf.get('sortino_ratio', 0):.3f}",
                'Calmar Ratio': f"{perf.get('calmar_ratio', 0):.3f}",
                'VaR (95%)': f"{perf.get('var_95', 0):.2%}",
                'CVaR (95%)': f"{perf.get('cvar_95', 0):.2%}"
            }
        
        if 'trading' in self.performance_summary:
            trading = self.performance_summary['trading']
            summary['Trading Metrics'] = {
                'Total Trades': trading.get('total_trades', 0),
                'Win Rate': f"{trading.get('win_rate', 0):.2%}",
                'Profit Factor': f"{trading.get('profit_factor', 0):.2f}",
                'Average Trade PnL': f"${trading.get('avg_trade_pnl', 0):,.2f}"
            }
        
        return summary
    
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.portfolio_values:
                logger.warning("No portfolio values to plot")
                return
            
            portfolio_array = np.array(self.portfolio_values)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Portfolio value over time
            ax1.plot(portfolio_array, label='Portfolio Value', linewidth=2)
            ax1.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('Trading Steps')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Returns distribution
            if len(portfolio_array) > 1:
                returns = np.diff(portfolio_array) / portfolio_array[:-1]
                ax2.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                ax2.set_title('Returns Distribution')
                ax2.set_xlabel('Returns')
                ax2.set_ylabel('Frequency')
                ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    def save_results(self, file_path: str):
        """Save backtest results to file"""
        import json
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'portfolio_values': self.portfolio_values,
            'performance_summary': self.performance_summary,
            'trades': self.trades,
            'positions_history': self.positions_history
        }
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {file_path}")
    
    def load_results(self, file_path: str):
        """Load backtest results from file"""
        import json
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        self.portfolio_values = results.get('portfolio_values', [])
        self.performance_summary = results.get('performance_summary', {})
        self.trades = results.get('trades', [])
        self.positions_history = results.get('positions_history', [])
        
        logger.info(f"Backtest results loaded from {file_path}") 