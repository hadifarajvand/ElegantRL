"""
Comprehensive Statistics Manager for Paper Trading System
Handles collection, analysis, and reporting of trading statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import sqlite3
from pathlib import Path
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TradeStatistics:
    """Trade statistics data class"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    total_profit: float
    total_loss: float
    net_profit: float


@dataclass
class PerformanceStatistics:
    """Performance statistics data class"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float
    information_ratio: float
    beta: float
    alpha: float


@dataclass
class RiskStatistics:
    """Risk statistics data class"""
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    downside_deviation: float
    ulcer_index: float
    gain_to_pain_ratio: float
    risk_of_ruin: float


class StatisticsManager:
    """
    Comprehensive statistics manager for paper trading system
    Handles:
    - Trade analysis and statistics
    - Performance metrics calculation
    - Risk metrics calculation
    - Portfolio analysis
    - Strategy comparison
    - Report generation
    """
    
    def __init__(self, data_dir: str = "./paper_trading_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "trading_database.db"
        
        # Statistics storage
        self.trade_stats = {}
        self.performance_stats = {}
        self.risk_stats = {}
        self.portfolio_stats = {}
        
        logger.info(f"Statistics manager initialized at {self.data_dir}")
    
    def calculate_trade_statistics(self, trade_logs: List[Dict[str, Any]]) -> TradeStatistics:
        """Calculate comprehensive trade statistics"""
        if not trade_logs:
            return TradeStatistics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trade_logs)
        
        # Calculate trade profits/losses
        profits = []
        losses = []
        
        for i in range(1, len(df)):
            current_value = df.iloc[i]['portfolio_value']
            previous_value = df.iloc[i-1]['portfolio_value']
            change = current_value - previous_value
            
            if change > 0:
                profits.append(change)
            else:
                losses.append(abs(change))
        
        # Calculate statistics
        total_trades = len(profits) + len(losses)
        winning_trades = len(profits)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        average_win = np.mean(profits) if profits else 0.0
        average_loss = np.mean(losses) if losses else 0.0
        largest_win = max(profits) if profits else 0.0
        largest_loss = max(losses) if losses else 0.0
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        net_profit = total_profit - total_loss
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return TradeStatistics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit
        )
    
    def calculate_performance_statistics(self, portfolio_values: List[float], 
                                      risk_free_rate: float = 0.02) -> PerformanceStatistics:
        """Calculate comprehensive performance statistics"""
        if len(portfolio_values) < 2:
            return PerformanceStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic statistics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Annualized return (assuming daily data)
        days = len(portfolio_values)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Information ratio (assuming benchmark return of 0)
        information_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        
        # Beta and Alpha (simplified - assuming market return of 0)
        beta = 1.0  # Placeholder
        alpha = np.mean(returns) * 252  # Annualized alpha
        
        return PerformanceStatistics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha
        )
    
    def calculate_risk_statistics(self, portfolio_values: List[float], 
                                confidence_level: float = 0.95) -> RiskStatistics:
        """Calculate comprehensive risk statistics"""
        if len(portfolio_values) < 2:
            return RiskStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, (1 - 0.95) * 100)
        var_99 = np.percentile(returns, (1 - 0.99) * 100)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95]) if var_95 in returns else var_95
        cvar_99 = np.mean(returns[returns <= var_99]) if var_99 in returns else var_99
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(portfolio_values)
        
        # Gain to Pain ratio
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        gain_to_pain_ratio = np.sum(gains) / abs(np.sum(losses)) if np.sum(losses) != 0 else float('inf')
        
        # Risk of Ruin (simplified)
        risk_of_ruin = self._calculate_risk_of_ruin(returns)
        
        return RiskStatistics(
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            downside_deviation=downside_deviation,
            ulcer_index=ulcer_index,
            gain_to_pain_ratio=gain_to_pain_ratio,
            risk_of_ruin=risk_of_ruin
        )
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_ulcer_index(self, portfolio_values: List[float]) -> float:
        """Calculate Ulcer Index"""
        if len(portfolio_values) < 2:
            return 0.0
        
        # Calculate drawdowns
        peak = portfolio_values[0]
        drawdowns = []
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            drawdowns.append(dd)
        
        # Calculate Ulcer Index
        squared_drawdowns = [dd ** 2 for dd in drawdowns]
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        return ulcer_index
    
    def _calculate_risk_of_ruin(self, returns: np.ndarray) -> float:
        """Calculate simplified risk of ruin"""
        if len(returns) == 0:
            return 0.0
        
        # Simplified calculation
        win_rate = np.sum(returns > 0) / len(returns)
        avg_win = np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0.0
        avg_loss = np.mean(returns[returns < 0]) if np.sum(returns < 0) > 0 else 0.0
        
        if avg_loss == 0:
            return 0.0
        
        # Kelly Criterion for risk of ruin
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win != 0 else 0.0
        
        # Simplified risk of ruin
        risk_of_ruin = max(0.0, 1 - kelly_fraction) if kelly_fraction > 0 else 1.0
        
        return risk_of_ruin
    
    def analyze_portfolio(self, portfolio_values: List[float], 
                         trade_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive portfolio analysis"""
        analysis = {}
        
        # Calculate all statistics
        trade_stats = self.calculate_trade_statistics(trade_logs)
        performance_stats = self.calculate_performance_statistics(portfolio_values)
        risk_stats = self.calculate_risk_statistics(portfolio_values)
        
        # Store results
        analysis['trade_statistics'] = asdict(trade_stats)
        analysis['performance_statistics'] = asdict(performance_stats)
        analysis['risk_statistics'] = asdict(risk_stats)
        
        # Additional analysis
        analysis['portfolio_summary'] = {
            'initial_value': portfolio_values[0] if portfolio_values else 0.0,
            'final_value': portfolio_values[-1] if portfolio_values else 0.0,
            'total_trades': len(trade_logs),
            'trading_days': len(portfolio_values),
            'average_daily_return': np.mean(np.diff(portfolio_values) / portfolio_values[:-1]) if len(portfolio_values) > 1 else 0.0
        }
        
        return analysis
    
    def compare_strategies(self, strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple trading strategies"""
        comparison = {
            'strategies': {},
            'rankings': {},
            'summary': {}
        }
        
        for strategy_name, results in strategy_results.items():
            if 'portfolio_values' in results and 'trade_logs' in results:
                analysis = self.analyze_portfolio(
                    results['portfolio_values'],
                    results['trade_logs']
                )
                comparison['strategies'][strategy_name] = analysis
        
        # Create rankings
        if comparison['strategies']:
            # Rank by total return
            returns = {name: data['performance_statistics']['total_return'] 
                      for name, data in comparison['strategies'].items()}
            comparison['rankings']['by_return'] = sorted(returns.items(), key=lambda x: x[1], reverse=True)
            
            # Rank by Sharpe ratio
            sharpe_ratios = {name: data['performance_statistics']['sharpe_ratio'] 
                            for name, data in comparison['strategies'].items()}
            comparison['rankings']['by_sharpe'] = sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True)
            
            # Rank by max drawdown (lower is better)
            drawdowns = {name: data['performance_statistics']['max_drawdown'] 
                        for name, data in comparison['strategies'].items()}
            comparison['rankings']['by_drawdown'] = sorted(drawdowns.items(), key=lambda x: x[1])
            
            # Summary statistics
            all_returns = list(returns.values())
            comparison['summary'] = {
                'best_return': max(all_returns),
                'worst_return': min(all_returns),
                'average_return': np.mean(all_returns),
                'return_std': np.std(all_returns),
                'best_strategy': comparison['rankings']['by_return'][0][0],
                'most_consistent': comparison['rankings']['by_sharpe'][0][0],
                'safest': comparison['rankings']['by_drawdown'][0][0]
            }
        
        return comparison
    
    def generate_report(self, analysis: Dict[str, Any], 
                       output_dir: str = "./paper_trading_data/reports") -> str:
        """Generate comprehensive analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f"trading_analysis_{timestamp}.json"
        
        # Add metadata
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'comprehensive_trading_analysis',
                'version': '1.0'
            },
            'analysis': analysis
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Analysis report generated: {report_file}")
        return str(report_file)
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get statistics from the database"""
        if not self.db_path.exists():
            return {'error': 'Database not found'}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get basic statistics
                stats = {}
                
                # Trade statistics
                trade_stats = conn.execute("""
                    SELECT COUNT(*) as total_trades,
                           COUNT(DISTINCT symbol) as unique_symbols,
                           COUNT(DISTINCT DATE(timestamp)) as trading_days,
                           MIN(timestamp) as first_trade,
                           MAX(timestamp) as last_trade,
                           AVG(portfolio_value) as avg_portfolio_value,
                           MAX(portfolio_value) as max_portfolio_value,
                           MIN(portfolio_value) as min_portfolio_value
                    FROM trades
                """).fetchone()
                
                stats['trades'] = dict(trade_stats)
                
                # Performance statistics
                perf_stats = conn.execute("""
                    SELECT COUNT(*) as total_records,
                           COUNT(DISTINCT strategy_name) as unique_strategies,
                           AVG(total_return) as avg_return,
                           AVG(sharpe_ratio) as avg_sharpe,
                           AVG(max_drawdown) as avg_drawdown,
                           MAX(total_return) as best_return,
                           MIN(total_return) as worst_return
                    FROM performance_metrics
                """).fetchone()
                
                stats['performance'] = dict(perf_stats)
                
                # Model statistics
                model_stats = conn.execute("""
                    SELECT COUNT(*) as total_models,
                           COUNT(DISTINCT agent_type) as unique_agent_types,
                           AVG(final_reward) as avg_final_reward,
                           MAX(final_reward) as best_final_reward
                    FROM model_checkpoints
                """).fetchone()
                
                stats['models'] = dict(model_stats)
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}


def create_statistics_manager(data_dir: str = "./paper_trading_data") -> StatisticsManager:
    """Create and return a statistics manager instance"""
    return StatisticsManager(data_dir) 