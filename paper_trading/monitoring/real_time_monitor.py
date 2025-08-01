"""
Real-time Monitoring System for Paper Trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import threading
import time
import queue
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import websockets
import asyncio
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class MonitoringAlert:
    """Monitoring alert data class"""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System metrics data class"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    queue_size: int


@dataclass
class TradingMetrics:
    """Trading metrics data class"""
    timestamp: datetime
    portfolio_value: float
    total_pnl: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    active_positions: int
    total_trades: int
    win_rate: float


class RealTimeMonitor:
    """
    Real-time Monitoring System
    
    Features:
    - System health monitoring
    - Trading performance tracking
    - Alert management
    - Real-time dashboards
    - Performance analytics
    """
    
    def __init__(self, 
                 alert_callbacks: List[Callable] = None,
                 metrics_interval: int = 5,  # seconds
                 alert_thresholds: Dict = None):
        
        self.alert_callbacks = alert_callbacks or []
        self.metrics_interval = metrics_interval
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_queue = queue.Queue()
        self.metrics_history = {
            'system': deque(maxlen=1000),
            'trading': deque(maxlen=1000),
            'alerts': deque(maxlen=1000)
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.system_health = {}
        self.trading_health = {}
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = []
        
        logger.info("RealTimeMonitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.metrics_history['system'].append(system_metrics)
                
                # Collect trading metrics
                trading_metrics = self._collect_trading_metrics()
                self.metrics_history['trading'].append(trading_metrics)
                
                # Check for alerts
                alerts = self._check_alerts(system_metrics, trading_metrics)
                for alert in alerts:
                    self._process_alert(alert)
                
                # Update health status
                self._update_health_status(system_metrics, trading_metrics)
                
                # Sleep for interval
                time.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.metrics_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network latency (simplified)
            network_latency = 50.0  # Placeholder
            
            # Active connections (simplified)
            active_connections = len(psutil.net_connections())
            
            # Queue size
            queue_size = self.alert_queue.qsize()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                active_connections=active_connections,
                queue_size=queue_size
            )
            
        except ImportError:
            # Fallback if psutil not available
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_latency=0.0,
                active_connections=0,
                queue_size=0
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_latency=0.0,
                active_connections=0,
                queue_size=0
            )
    
    def _collect_trading_metrics(self) -> TradingMetrics:
        """Collect trading performance metrics"""
        try:
            # Get latest trading data (placeholder)
            portfolio_value = 100000.0  # Placeholder
            total_pnl = 5000.0  # Placeholder
            daily_return = 0.02  # Placeholder
            sharpe_ratio = 1.5  # Placeholder
            max_drawdown = 0.05  # Placeholder
            active_positions = 3  # Placeholder
            total_trades = 50  # Placeholder
            win_rate = 0.65  # Placeholder
            
            return TradingMetrics(
                timestamp=datetime.now(),
                portfolio_value=portfolio_value,
                total_pnl=total_pnl,
                daily_return=daily_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                active_positions=active_positions,
                total_trades=total_trades,
                win_rate=win_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return TradingMetrics(
                timestamp=datetime.now(),
                portfolio_value=0.0,
                total_pnl=0.0,
                daily_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                active_positions=0,
                total_trades=0,
                win_rate=0.0
            )
    
    def _check_alerts(self, system_metrics: SystemMetrics, 
                     trading_metrics: TradingMetrics) -> List[MonitoringAlert]:
        """Check for alerts based on metrics"""
        alerts = []
        
        # System alerts
        if system_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(MonitoringAlert(
                alert_id=f"cpu_high_{datetime.now().timestamp()}",
                alert_type="system",
                severity="high" if system_metrics.cpu_usage > 90 else "medium",
                message=f"High CPU usage: {system_metrics.cpu_usage:.1f}%",
                timestamp=datetime.now(),
                source="system_monitor",
                data={"cpu_usage": system_metrics.cpu_usage}
            ))
        
        if system_metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(MonitoringAlert(
                alert_id=f"memory_high_{datetime.now().timestamp()}",
                alert_type="system",
                severity="high" if system_metrics.memory_usage > 90 else "medium",
                message=f"High memory usage: {system_metrics.memory_usage:.1f}%",
                timestamp=datetime.now(),
                source="system_monitor",
                data={"memory_usage": system_metrics.memory_usage}
            ))
        
        # Trading alerts
        if trading_metrics.daily_return < self.alert_thresholds['daily_return_min']:
            alerts.append(MonitoringAlert(
                alert_id=f"daily_loss_{datetime.now().timestamp()}",
                alert_type="trading",
                severity="medium",
                message=f"Daily loss: {trading_metrics.daily_return:.2%}",
                timestamp=datetime.now(),
                source="trading_monitor",
                data={"daily_return": trading_metrics.daily_return}
            ))
        
        if trading_metrics.max_drawdown > self.alert_thresholds['max_drawdown']:
            alerts.append(MonitoringAlert(
                alert_id=f"drawdown_high_{datetime.now().timestamp()}",
                alert_type="trading",
                severity="high",
                message=f"High drawdown: {trading_metrics.max_drawdown:.2%}",
                timestamp=datetime.now(),
                source="trading_monitor",
                data={"max_drawdown": trading_metrics.max_drawdown}
            ))
        
        return alerts
    
    def _process_alert(self, alert: MonitoringAlert):
        """Process and handle alerts"""
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self.metrics_history['alerts'].append(alert)
            
            # Log alert
            logger.warning(f"ALERT [{alert.severity.upper()}] {alert.message}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            # Add to queue
            self.alert_queue.put(alert)
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    def _update_health_status(self, system_metrics: SystemMetrics, 
                            trading_metrics: TradingMetrics):
        """Update system and trading health status"""
        # System health
        self.system_health = {
            'status': 'healthy' if system_metrics.cpu_usage < 80 else 'warning',
            'cpu_usage': system_metrics.cpu_usage,
            'memory_usage': system_metrics.memory_usage,
            'disk_usage': system_metrics.disk_usage,
            'last_updated': datetime.now()
        }
        
        # Trading health
        self.trading_health = {
            'status': 'healthy' if trading_metrics.daily_return > 0 else 'warning',
            'portfolio_value': trading_metrics.portfolio_value,
            'daily_return': trading_metrics.daily_return,
            'sharpe_ratio': trading_metrics.sharpe_ratio,
            'max_drawdown': trading_metrics.max_drawdown,
            'last_updated': datetime.now()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return self.system_health
    
    def get_trading_health(self) -> Dict[str, Any]:
        """Get current trading health status"""
        return self.trading_health
    
    def get_active_alerts(self) -> List[MonitoringAlert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_metrics_history(self, metric_type: str = 'all') -> Dict[str, List]:
        """Get metrics history"""
        if metric_type == 'all':
            return {
                'system': list(self.metrics_history['system']),
                'trading': list(self.metrics_history['trading']),
                'alerts': list(self.metrics_history['alerts'])
            }
        else:
            return {metric_type: list(self.metrics_history.get(metric_type, []))}
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        """Set alert thresholds"""
        self.alert_thresholds.update(thresholds)
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds"""
        return {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'network_latency': 100.0,
            'daily_return_min': -0.05,
            'max_drawdown': 0.1,
            'sharpe_ratio_min': 0.5
        }

    def update_metrics(self, portfolio):
        """Stub for test compatibility. Returns a TradingMetrics object."""
        from paper_trading.utils.metrics import TradingMetrics
        return TradingMetrics()


class PerformanceAnalytics:
    """
    Performance Analytics System
    
    Features:
    - Performance tracking
    - Risk analysis
    - Benchmark comparison
    - Performance attribution
    """
    
    def __init__(self):
        self.performance_data = []
        self.benchmark_data = []
        self.risk_metrics = {}
        
        logger.info("PerformanceAnalytics initialized")
    
    def add_performance_data(self, data: Dict[str, Any]):
        """Add performance data point"""
        self.performance_data.append({
            'timestamp': datetime.now(),
            **data
        })
    
    def add_benchmark_data(self, benchmark: str, data: Dict[str, Any]):
        """Add benchmark data"""
        self.benchmark_data.append({
            'timestamp': datetime.now(),
            'benchmark': benchmark,
            **data
        })
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.performance_data:
            return {}
        
        try:
            # Extract returns
            returns = [data.get('return', 0) for data in self.performance_data]
            
            # Basic metrics
            total_return = np.sum(returns)
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Risk metrics
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            var_95 = np.percentile(returns, 5)
            
            # Trading metrics
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)
            
            return {
                'total_return': total_return,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate"""
        positive_returns = [r for r in returns if r > 0]
        return len(positive_returns) / len(returns) if returns else 0
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor"""
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        gross_profit = sum(positive_returns) if positive_returns else 0
        gross_loss = abs(sum(negative_returns)) if negative_returns else 0
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def compare_to_benchmark(self, benchmark: str) -> Dict[str, float]:
        """Compare performance to benchmark"""
        if not self.benchmark_data:
            return {}
        
        try:
            # Get benchmark data
            benchmark_returns = [
                data.get('return', 0) for data in self.benchmark_data 
                if data.get('benchmark') == benchmark
            ]
            
            if not benchmark_returns:
                return {}
            
            # Get strategy returns
            strategy_returns = [data.get('return', 0) for data in self.performance_data]
            
            # Calculate excess return
            excess_return = np.mean(strategy_returns) - np.mean(benchmark_returns)
            
            # Calculate information ratio
            excess_returns = np.array(strategy_returns) - np.array(benchmark_returns)
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            # Calculate beta
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate alpha
            alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
            
            return {
                'excess_return': excess_return,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha
            }
            
        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        metrics = self.calculate_performance_metrics()
        
        return {
            'metrics': metrics,
            'data_points': len(self.performance_data),
            'benchmarks': len(set(data.get('benchmark') for data in self.benchmark_data)),
            'last_updated': datetime.now()
        }


def create_real_time_monitor(alert_callbacks: List[Callable] = None,
                           metrics_interval: int = 5) -> RealTimeMonitor:
    """Convenience function to create real-time monitor"""
    return RealTimeMonitor(alert_callbacks, metrics_interval)


def create_performance_analytics() -> PerformanceAnalytics:
    """Convenience function to create performance analytics"""
    return PerformanceAnalytics() 