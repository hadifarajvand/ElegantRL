"""
Data Persistence System for Paper Trading
Handles storage of trading data, logs, statistics, and performance metrics
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import sqlite3
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class DataPersistenceManager:
    """
    Comprehensive data persistence manager for paper trading system
    Handles storage of:
    - Trading data and market data
    - Trade logs and execution records
    - Performance statistics and metrics
    - Configuration files
    - Model checkpoints and training data
    """
    
    def __init__(self, base_dir: str = "./paper_trading_data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_database()
        
        logger.info(f"Data persistence manager initialized at {self.base_dir}")
    
    def setup_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            "market_data",
            "trade_logs", 
            "performance_stats",
            "model_checkpoints",
            "configs",
            "backtest_results",
            "real_time_data",
            "logs",
            "reports"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def setup_database(self):
        """Setup SQLite database for structured data storage"""
        db_path = self.base_dir / "trading_database.db"
        self.db_path = db_path
        
        with sqlite3.connect(db_path) as conn:
            # Create tables for structured data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL,
                    price REAL,
                    portfolio_value REAL,
                    cash REAL,
                    positions TEXT,
                    reward REAL,
                    step INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy_name TEXT,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    volatility REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    final_value REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    data_hash TEXT,
                    file_path TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    agent_type TEXT,
                    performance_metrics TEXT,
                    file_path TEXT,
                    created_at TEXT NOT NULL,
                    training_episodes INTEGER,
                    final_reward REAL
                )
            """)
            
            conn.commit()
    
    def save_market_data(self, symbol: str, data: pd.DataFrame, 
                        start_date: str, end_date: str) -> str:
        """Save market data to file and database"""
        try:
            # Create filename with CSV as primary format
            filename = f"{symbol.replace('/', '_')}_{start_date}_{end_date}"
            csv_path = self.base_dir / "market_data" / f"{filename}.csv"
            
            # Save as CSV (primary format)
            data.to_csv(csv_path, index=True)
            file_path = csv_path
            logger.info(f"Market data saved as CSV: {file_path}")
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO market_data_cache 
                    (symbol, start_date, end_date, data_hash, file_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol, start_date, end_date, 
                    str(hash(data.to_string())),
                    str(file_path), datetime.now().isoformat()
                ))
                conn.commit()
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            return ""
    
    def load_market_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load market data from file"""
        try:
            filename = f"{symbol.replace('/', '_')}_{start_date}_{end_date}"
            csv_path = self.base_dir / "market_data" / f"{filename}.csv"
            
            # Load CSV (primary format)
            if csv_path.exists():
                try:
                    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    logger.info(f"Market data loaded from CSV: {csv_path}")
                    return data
                except Exception as csv_error:
                    logger.warning(f"CSV load failed: {csv_error}")
            
            logger.warning(f"Market data not found: {csv_path}")
            return None
                
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return None
    
    def save_trade_log(self, trade_data: Dict[str, Any]) -> bool:
        """Save individual trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades 
                    (timestamp, symbol, action, quantity, price, portfolio_value, 
                     cash, positions, reward, step)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('timestamp', datetime.now().isoformat()),
                    trade_data.get('symbol', ''),
                    trade_data.get('action', ''),
                    trade_data.get('quantity', 0.0),
                    trade_data.get('price', 0.0),
                    trade_data.get('portfolio_value', 0.0),
                    trade_data.get('cash', 0.0),
                    json.dumps(trade_data.get('positions', [])),
                    trade_data.get('reward', 0.0),
                    trade_data.get('step', 0)
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")
            return False
    
    def save_trade_logs_batch(self, trade_logs: List[Dict[str, Any]]) -> bool:
        """Save multiple trade logs at once"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for trade in trade_logs:
                    conn.execute("""
                        INSERT INTO trades 
                        (timestamp, symbol, action, quantity, price, portfolio_value, 
                         cash, positions, reward, step)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade.get('timestamp', datetime.now().isoformat()),
                        trade.get('symbol', ''),
                        trade.get('action', ''),
                        trade.get('quantity', 0.0),
                        trade.get('price', 0.0),
                        trade.get('portfolio_value', 0.0),
                        trade.get('cash', 0.0),
                        json.dumps(trade.get('positions', [])),
                        trade.get('reward', 0.0),
                        trade.get('step', 0)
                    ))
                conn.commit()
            
            logger.info(f"Saved {len(trade_logs)} trade logs")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade logs batch: {e}")
            return False
    
    def get_trade_logs(self, symbol: str = None, start_date: str = None, 
                       end_date: str = None, limit: int = 1000) -> pd.DataFrame:
        """Retrieve trade logs from database"""
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving trade logs: {e}")
            return pd.DataFrame()
    
    def save_performance_metrics(self, metrics: Dict[str, Any], 
                                strategy_name: str = "default") -> bool:
        """Save performance metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, strategy_name, total_return, sharpe_ratio, 
                     max_drawdown, volatility, win_rate, total_trades, final_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    strategy_name,
                    metrics.get('total_return', 0.0),
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('max_drawdown', 0.0),
                    metrics.get('volatility', 0.0),
                    metrics.get('win_rate', 0.0),
                    metrics.get('total_trades', 0),
                    metrics.get('final_value', 0.0)
                ))
                conn.commit()
            
            # Also save to JSON file for easy access
            metrics_file = self.base_dir / "performance_stats" / f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Performance metrics saved for {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            return False
    
    def get_performance_metrics(self, strategy_name: str = None, 
                               limit: int = 100) -> pd.DataFrame:
        """Retrieve performance metrics from database"""
        try:
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving performance metrics: {e}")
            return pd.DataFrame()
    
    def save_model_checkpoint(self, model_data: Dict[str, Any], 
                             model_name: str, agent_type: str = "PPO") -> str:
        """Save model checkpoint and metadata"""
        try:
            # Create checkpoint directory
            checkpoint_dir = self.base_dir / "model_checkpoints" / model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_file = checkpoint_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            
            if 'model_state' in model_data:
                import torch
                torch.save(model_data['model_state'], model_file)
            
            # Save metadata
            metadata_file = checkpoint_dir / f"{model_name}_metadata.json"
            metadata = {
                'model_name': model_name,
                'agent_type': agent_type,
                'file_path': str(model_file),
                'created_at': datetime.now().isoformat(),
                'training_episodes': model_data.get('training_episodes', 0),
                'final_reward': model_data.get('final_reward', 0.0),
                'performance_metrics': model_data.get('performance_metrics', {})
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_checkpoints 
                    (model_name, agent_type, performance_metrics, file_path, 
                     created_at, training_episodes, final_reward)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    agent_type,
                    json.dumps(model_data.get('performance_metrics', {})),
                    str(model_file),
                    datetime.now().isoformat(),
                    model_data.get('training_episodes', 0),
                    model_data.get('final_reward', 0.0)
                ))
                conn.commit()
            
            logger.info(f"Model checkpoint saved: {model_file}")
            return str(model_file)
            
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {e}")
            return ""
    
    def load_model_checkpoint(self, model_name: str, checkpoint_name: str = None) -> Optional[Dict[str, Any]]:
        """Load model checkpoint and metadata"""
        try:
            checkpoint_dir = self.base_dir / "model_checkpoints" / model_name
            
            if not checkpoint_dir.exists():
                logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
                return None
            
            # Find the latest checkpoint if none specified
            if checkpoint_name is None:
                model_files = list(checkpoint_dir.glob("*.pt"))
                if not model_files:
                    logger.warning(f"No model files found in {checkpoint_dir}")
                    return None
                model_file = max(model_files, key=lambda x: x.stat().st_mtime)
            else:
                model_file = checkpoint_dir / checkpoint_name
            
            # Load metadata
            metadata_file = model_file.parent / f"{model_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Load model state
            model_data = {'metadata': metadata}
            if model_file.exists():
                import torch
                model_data['model_state'] = torch.load(model_file)
            
            logger.info(f"Model checkpoint loaded: {model_file}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model checkpoint: {e}")
            return None
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> str:
        """Save configuration to file"""
        try:
            config_file = self.base_dir / "configs" / f"{config_name}.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved: {config_file}")
            return str(config_file)
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return ""
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        try:
            config_file = self.base_dir / "configs" / f"{config_name}.yaml"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                logger.info(f"Configuration loaded: {config_file}")
                return config
            else:
                logger.warning(f"Configuration file not found: {config_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None
    
    def save_backtest_results(self, results: Dict[str, Any], 
                            backtest_name: str) -> str:
        """Save backtest results to files"""
        try:
            backtest_dir = self.base_dir / "backtest_results" / backtest_name
            backtest_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            summary_file = backtest_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(results.get('summary', {}), f, indent=2, default=str)
            
            # Save detailed results
            detailed_file = backtest_dir / "detailed_results.json"
            with open(detailed_file, 'w') as f:
                json.dump(results.get('detailed_results', {}), f, indent=2, default=str)
            
            # Save trade logs
            if 'trade_logs' in results:
                trade_logs_file = backtest_dir / "trade_logs.json"
                with open(trade_logs_file, 'w') as f:
                    json.dump(results['trade_logs'], f, indent=2, default=str)
            
            # Save portfolio values
            if 'portfolio_values' in results:
                portfolio_file = backtest_dir / "portfolio_values.csv"
                pd.DataFrame(results['portfolio_values']).to_csv(portfolio_file, index=False)
            
            logger.info(f"Backtest results saved: {backtest_dir}")
            return str(backtest_dir)
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return ""
    
    def save_real_time_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Save real-time market data"""
        try:
            real_time_dir = self.base_dir / "real_time_data"
            real_time_dir.mkdir(exist_ok=True)
            
            # Save to JSON file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol.replace('/', '_')}_{timestamp}.json"
            file_path = real_time_dir / filename
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Real-time data saved: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving real-time data: {e}")
            return False
    
    def generate_report(self, report_name: str, data: Dict[str, Any]) -> str:
        """Generate and save comprehensive report"""
        try:
            reports_dir = self.base_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            report_file = reports_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get trade statistics
                trade_stats = conn.execute("""
                    SELECT COUNT(*) as total_trades,
                           COUNT(DISTINCT symbol) as unique_symbols,
                           MIN(timestamp) as first_trade,
                           MAX(timestamp) as last_trade
                    FROM trades
                """).fetchone()
                
                # Get performance statistics
                perf_stats = conn.execute("""
                    SELECT COUNT(*) as total_records,
                           COUNT(DISTINCT strategy_name) as unique_strategies,
                           AVG(total_return) as avg_return,
                           AVG(sharpe_ratio) as avg_sharpe
                    FROM performance_metrics
                """).fetchone()
                
                # Get model statistics
                model_stats = conn.execute("""
                    SELECT COUNT(*) as total_models,
                           COUNT(DISTINCT agent_type) as unique_agent_types
                    FROM model_checkpoints
                """).fetchone()
            
            return {
                'trades': dict(trade_stats),
                'performance': dict(perf_stats),
                'models': dict(model_stats),
                'database_path': str(self.db_path),
                'base_directory': str(self.base_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


def create_data_persistence_manager(base_dir: str = "./paper_trading_data") -> DataPersistenceManager:
    """Create and return a data persistence manager instance"""
    return DataPersistenceManager(base_dir) 