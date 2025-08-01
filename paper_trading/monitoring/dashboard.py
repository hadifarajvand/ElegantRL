"""
Dashboard and Reporting System for Paper Trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_interval: int = 30  # seconds
    chart_theme: str = 'dark_background'
    max_data_points: int = 1000
    export_format: str = 'html'
    enable_real_time: bool = True


class TradingDashboard:
    """
    Trading Dashboard System
    
    Features:
    - Real-time portfolio overview
    - Performance charts
    - Risk metrics display
    - Trade history
    - System status
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.dashboard_data = {}
        self.charts = {}
        self.reports = {}
        
        # Set up plotting style
        plt.style.use(self.config.chart_theme)
        
        logger.info("TradingDashboard initialized")
    
    def update_dashboard_data(self, data: Dict[str, Any]):
        """Update dashboard data"""
        self.dashboard_data.update(data)
        self.dashboard_data['last_updated'] = datetime.now()
    
    def create_portfolio_overview(self) -> Dict[str, Any]:
        """Create portfolio overview section"""
        try:
            portfolio_data = self.dashboard_data.get('portfolio', {})
            
            overview = {
                'total_value': portfolio_data.get('total_value', 0),
                'cash': portfolio_data.get('cash', 0),
                'positions_value': portfolio_data.get('positions_value', 0),
                'total_pnl': portfolio_data.get('total_pnl', 0),
                'daily_pnl': portfolio_data.get('daily_pnl', 0),
                'active_positions': portfolio_data.get('active_positions', 0),
                'total_trades': portfolio_data.get('total_trades', 0),
                'win_rate': portfolio_data.get('win_rate', 0),
                'last_updated': datetime.now()
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error creating portfolio overview: {e}")
            return {}
    
    def create_performance_chart(self, data_type: str = 'returns') -> str:
        """Create performance chart as base64 image"""
        try:
            # Get performance data
            performance_data = self.dashboard_data.get('performance', {})
            
            if not performance_data:
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if data_type == 'returns':
                # Plot cumulative returns
                dates = performance_data.get('dates', [])
                returns = performance_data.get('returns', [])
                
                if dates and returns:
                    cumulative_returns = np.cumprod(1 + np.array(returns))
                    ax.plot(dates, cumulative_returns, linewidth=2, color='green')
                    ax.set_title('Cumulative Returns')
                    ax.set_ylabel('Portfolio Value')
                    ax.grid(True, alpha=0.3)
            
            elif data_type == 'drawdown':
                # Plot drawdown
                dates = performance_data.get('dates', [])
                drawdown = performance_data.get('drawdown', [])
                
                if dates and drawdown:
                    ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
                    ax.plot(dates, drawdown, linewidth=2, color='red')
                    ax.set_title('Portfolio Drawdown')
                    ax.set_ylabel('Drawdown %')
                    ax.grid(True, alpha=0.3)
            
            elif data_type == 'volatility':
                # Plot rolling volatility
                dates = performance_data.get('dates', [])
                volatility = performance_data.get('volatility', [])
                
                if dates and volatility:
                    ax.plot(dates, volatility, linewidth=2, color='blue')
                    ax.set_title('Rolling Volatility (30-day)')
                    ax.set_ylabel('Volatility %')
                    ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return ""
    
    def create_risk_metrics_display(self) -> Dict[str, Any]:
        """Create risk metrics display"""
        try:
            risk_data = self.dashboard_data.get('risk_metrics', {})
            
            risk_display = {
                'sharpe_ratio': risk_data.get('sharpe_ratio', 0),
                'sortino_ratio': risk_data.get('sortino_ratio', 0),
                'calmar_ratio': risk_data.get('calmar_ratio', 0),
                'max_drawdown': risk_data.get('max_drawdown', 0),
                'var_95': risk_data.get('var_95', 0),
                'cvar_95': risk_data.get('cvar_95', 0),
                'volatility': risk_data.get('volatility', 0),
                'beta': risk_data.get('beta', 0),
                'alpha': risk_data.get('alpha', 0),
                'information_ratio': risk_data.get('information_ratio', 0),
                'last_updated': datetime.now()
            }
            
            return risk_display
            
        except Exception as e:
            logger.error(f"Error creating risk metrics display: {e}")
            return {}
    
    def create_trade_history_table(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Create trade history table"""
        try:
            trade_history = self.dashboard_data.get('trade_history', [])
            
            # Sort by timestamp (most recent first)
            sorted_trades = sorted(trade_history, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            # Limit results
            recent_trades = sorted_trades[:limit]
            
            # Format trades for display
            formatted_trades = []
            for trade in recent_trades:
                formatted_trade = {
                    'timestamp': trade.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': trade.get('symbol', ''),
                    'action': trade.get('action', ''),
                    'quantity': trade.get('quantity', 0),
                    'price': trade.get('price', 0),
                    'value': trade.get('value', 0),
                    'pnl': trade.get('pnl', 0),
                    'status': trade.get('status', '')
                }
                formatted_trades.append(formatted_trade)
            
            return formatted_trades
            
        except Exception as e:
            logger.error(f"Error creating trade history table: {e}")
            return []
    
    def create_system_status_display(self) -> Dict[str, Any]:
        """Create system status display"""
        try:
            system_data = self.dashboard_data.get('system_status', {})
            
            status_display = {
                'system_health': system_data.get('system_health', 'unknown'),
                'trading_health': system_data.get('trading_health', 'unknown'),
                'cpu_usage': system_data.get('cpu_usage', 0),
                'memory_usage': system_data.get('memory_usage', 0),
                'disk_usage': system_data.get('disk_usage', 0),
                'active_alerts': system_data.get('active_alerts', 0),
                'uptime': system_data.get('uptime', 0),
                'last_updated': datetime.now()
            }
            
            return status_display
            
        except Exception as e:
            logger.error(f"Error creating system status display: {e}")
            return {}
    
    def generate_html_report(self) -> str:
        """Generate HTML dashboard report"""
        try:
            # Get dashboard sections
            portfolio_overview = self.create_portfolio_overview()
            risk_metrics = self.create_risk_metrics_display()
            trade_history = self.create_trade_history_table()
            system_status = self.create_system_status_display()
            
            # Generate charts
            returns_chart = self.create_performance_chart('returns')
            drawdown_chart = self.create_performance_chart('drawdown')
            
            # Create HTML template
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; min-width: 120px; text-align: center; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                    .metric-label {{ font-size: 12px; color: #666; }}
                    .positive {{ color: #28a745; }}
                    .negative {{ color: #dc3545; }}
                    .warning {{ color: #ffc107; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f8f9fa; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Trading Dashboard</h1>
                    <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="section">
                        <h2>Portfolio Overview</h2>
                        <div class="metric">
                            <div class="metric-value">${portfolio_overview.get('total_value', 0):,.2f}</div>
                            <div class="metric-label">Total Value</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value {'positive' if portfolio_overview.get('total_pnl', 0) >= 0 else 'negative'}">
                                ${portfolio_overview.get('total_pnl', 0):,.2f}
                            </div>
                            <div class="metric-label">Total P&L</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{portfolio_overview.get('active_positions', 0)}</div>
                            <div class="metric-label">Active Positions</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{portfolio_overview.get('win_rate', 0):.1%}</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Risk Metrics</h2>
                        <div class="metric">
                            <div class="metric-value">{risk_metrics.get('sharpe_ratio', 0):.2f}</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{risk_metrics.get('max_drawdown', 0):.2%}</div>
                            <div class="metric-label">Max Drawdown</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{risk_metrics.get('volatility', 0):.2%}</div>
                            <div class="metric-label">Volatility</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{risk_metrics.get('var_95', 0):.2%}</div>
                            <div class="metric-label">VaR (95%)</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Performance Charts</h2>
                        <div class="chart">
                            <h3>Cumulative Returns</h3>
                            <img src="data:image/png;base64,{returns_chart}" alt="Cumulative Returns">
                        </div>
                        <div class="chart">
                            <h3>Portfolio Drawdown</h3>
                            <img src="data:image/png;base64,{drawdown_chart}" alt="Portfolio Drawdown">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Recent Trades</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Symbol</th>
                                    <th>Action</th>
                                    <th>Quantity</th>
                                    <th>Price</th>
                                    <th>Value</th>
                                    <th>P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join([f'''
                                <tr>
                                    <td>{trade['timestamp']}</td>
                                    <td>{trade['symbol']}</td>
                                    <td>{trade['action']}</td>
                                    <td>{trade['quantity']}</td>
                                    <td>${trade['price']:.2f}</td>
                                    <td>${trade['value']:.2f}</td>
                                    <td class="{'positive' if trade['pnl'] >= 0 else 'negative'}">${trade['pnl']:.2f}</td>
                                </tr>
                                ''' for trade in trade_history])}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>System Status</h2>
                        <div class="metric">
                            <div class="metric-value">{system_status.get('system_health', 'unknown')}</div>
                            <div class="metric-label">System Health</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{system_status.get('trading_health', 'unknown')}</div>
                            <div class="metric-label">Trading Health</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{system_status.get('cpu_usage', 0):.1f}%</div>
                            <div class="metric-label">CPU Usage</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{system_status.get('memory_usage', 0):.1f}%</div>
                            <div class="metric-label">Memory Usage</div>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error generating dashboard</h1><p>{e}</p></body></html>"
    
    def save_dashboard_report(self, filename: str = None):
        """Save dashboard report to file"""
        try:
            if filename is None:
                filename = f"trading_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            html_content = self.generate_html_report()
            
            with open(filename, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Dashboard report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving dashboard report: {e}")
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary"""
        try:
            portfolio_overview = self.create_portfolio_overview()
            risk_metrics = self.create_risk_metrics_display()
            system_status = self.create_system_status_display()
            
            return {
                'portfolio': portfolio_overview,
                'risk_metrics': risk_metrics,
                'system_status': system_status,
                'last_updated': datetime.now(),
                'data_points': len(self.dashboard_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {}


class ReportGenerator:
    """
    Report Generator System
    
    Features:
    - Performance reports
    - Risk reports
    - Trade analysis reports
    - Custom report templates
    """
    
    def __init__(self):
        self.report_templates = {}
        self.report_data = {}
        
        logger.info("ReportGenerator initialized")
    
    def generate_performance_report(self, performance_data: Dict[str, Any]) -> str:
        """Generate performance report"""
        try:
            report = f"""
            PERFORMANCE REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            SUMMARY METRICS:
            - Total Return: {performance_data.get('total_return', 0):.2%}
            - Annualized Return: {performance_data.get('annualized_return', 0):.2%}
            - Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}
            - Sortino Ratio: {performance_data.get('sortino_ratio', 0):.2f}
            - Maximum Drawdown: {performance_data.get('max_drawdown', 0):.2%}
            - Volatility: {performance_data.get('volatility', 0):.2%}
            
            RISK METRICS:
            - VaR (95%): {performance_data.get('var_95', 0):.2%}
            - CVaR (95%): {performance_data.get('cvar_95', 0):.2%}
            - Beta: {performance_data.get('beta', 0):.2f}
            - Alpha: {performance_data.get('alpha', 0):.2%}
            
            TRADING METRICS:
            - Total Trades: {performance_data.get('total_trades', 0)}
            - Win Rate: {performance_data.get('win_rate', 0):.2%}
            - Profit Factor: {performance_data.get('profit_factor', 0):.2f}
            - Average Trade Duration: {performance_data.get('avg_trade_duration', 0):.1f} days
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating performance report: {e}"
    
    def generate_risk_report(self, risk_data: Dict[str, Any]) -> str:
        """Generate risk report"""
        try:
            report = f"""
            RISK ANALYSIS REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            PORTFOLIO RISK:
            - Portfolio Value: ${risk_data.get('portfolio_value', 0):,.2f}
            - Total Risk: {risk_data.get('total_risk', 0):.2%}
            - Systematic Risk: {risk_data.get('systematic_risk', 0):.2%}
            - Idiosyncratic Risk: {risk_data.get('idiosyncratic_risk', 0):.2%}
            
            RISK METRICS:
            - VaR (95%): {risk_data.get('var_95', 0):.2%}
            - VaR (99%): {risk_data.get('var_99', 0):.2%}
            - CVaR (95%): {risk_data.get('cvar_95', 0):.2%}
            - CVaR (99%): {risk_data.get('cvar_99', 0):.2%}
            
            CONCENTRATION RISK:
            - Largest Position: {risk_data.get('largest_position', 0):.2%}
            - Top 5 Positions: {risk_data.get('top_5_concentration', 0):.2%}
            - Sector Concentration: {risk_data.get('sector_concentration', 0):.2%}
            
            STRESS TEST RESULTS:
            - Market Crash Scenario: {risk_data.get('stress_market_crash', 0):.2%}
            - Volatility Spike: {risk_data.get('stress_volatility', 0):.2%}
            - Interest Rate Shock: {risk_data.get('stress_rates', 0):.2%}
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return f"Error generating risk report: {e}"
    
    def generate_trade_analysis_report(self, trade_data: Dict[str, Any]) -> str:
        """Generate trade analysis report"""
        try:
            report = f"""
            TRADE ANALYSIS REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            TRADE SUMMARY:
            - Total Trades: {trade_data.get('total_trades', 0)}
            - Winning Trades: {trade_data.get('winning_trades', 0)}
            - Losing Trades: {trade_data.get('losing_trades', 0)}
            - Win Rate: {trade_data.get('win_rate', 0):.2%}
            
            PROFITABILITY:
            - Gross Profit: ${trade_data.get('gross_profit', 0):,.2f}
            - Gross Loss: ${trade_data.get('gross_loss', 0):,.2f}
            - Net Profit: ${trade_data.get('net_profit', 0):,.2f}
            - Profit Factor: {trade_data.get('profit_factor', 0):.2f}
            
            TRADE CHARACTERISTICS:
            - Average Win: ${trade_data.get('avg_win', 0):,.2f}
            - Average Loss: ${trade_data.get('avg_loss', 0):,.2f}
            - Largest Win: ${trade_data.get('largest_win', 0):,.2f}
            - Largest Loss: ${trade_data.get('largest_loss', 0):,.2f}
            
            TIMING ANALYSIS:
            - Average Hold Time: {trade_data.get('avg_hold_time', 0):.1f} days
            - Best Trading Day: {trade_data.get('best_day', 'N/A')}
            - Worst Trading Day: {trade_data.get('worst_day', 'N/A')}
            """
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating trade analysis report: {e}")
            return f"Error generating trade analysis report: {e}"
    
    def save_report(self, report_content: str, filename: str):
        """Save report to file"""
        try:
            with open(filename, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def create_trading_dashboard(config: DashboardConfig = None) -> TradingDashboard:
    """Convenience function to create trading dashboard"""
    return TradingDashboard(config)


def create_report_generator() -> ReportGenerator:
    """Convenience function to create report generator"""
    return ReportGenerator() 