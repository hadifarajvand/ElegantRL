"""
Comprehensive Logging Configuration for Paper Trading System
Handles structured logging, file rotation, and different log levels
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import traceback


class PaperTradingLogger:
    """
    Comprehensive logging system for paper trading
    Handles:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - File rotation and archiving
    - Structured logging with JSON format
    - Performance metrics logging
    - Trade execution logging
    - Error tracking and reporting
    """
    
    def __init__(self, 
                 log_dir: str = "./paper_trading_data/logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        
        # Initialize loggers
        self._setup_loggers()
        
        # Performance tracking
        self.performance_logger = self._get_logger("performance")
        self.trade_logger = self._get_logger("trades")
        self.error_logger = self._get_logger("errors")
        self.system_logger = self._get_logger("system")
        
        self.system_logger.info("Paper Trading Logger initialized", extra={
            'log_dir': str(self.log_dir),
            'log_level': log_level,
            'enable_console': enable_console,
            'enable_file': enable_file
        })
    
    def _setup_loggers(self):
        """Setup all loggers with appropriate handlers"""
        # Create formatters
        self._create_formatters()
        
        # Setup root logger
        self._setup_root_logger()
        
        # Setup specific loggers
        self._setup_specific_loggers()
    
    def _create_formatters(self):
        """Create log formatters"""
        # Standard formatter
        self.standard_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # JSON formatter
        self.json_formatter = JSONFormatter()
        
        # Performance formatter
        self.performance_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _setup_root_logger(self):
        """Setup root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(self.standard_formatter)
            root_logger.addHandler(console_handler)
    
    def _setup_specific_loggers(self):
        """Setup specific loggers for different components"""
        loggers = {
            'performance': self._create_file_handler('performance.log'),
            'trades': self._create_file_handler('trades.log'),
            'errors': self._create_file_handler('errors.log'),
            'system': self._create_file_handler('system.log'),
            'data': self._create_file_handler('data.log'),
            'backtest': self._create_file_handler('backtest.log'),
            'training': self._create_file_handler('training.log'),
            'risk': self._create_file_handler('risk.log')
        }
        
        for logger_name, handlers in loggers.items():
            logger = logging.getLogger(f"paper_trading.{logger_name}")
            logger.setLevel(self.log_level)
            logger.propagate = False  # Don't propagate to root logger
            
            for handler in handlers:
                logger.addHandler(handler)
    
    def _create_file_handler(self, filename: str) -> list:
        """Create file handler with rotation"""
        handlers = []
        
        if self.enable_file:
            # Standard file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / filename,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(self.standard_formatter)
            handlers.append(file_handler)
            
            # JSON file handler
            if self.enable_json:
                json_filename = filename.replace('.log', '_json.log')
                json_handler = logging.handlers.RotatingFileHandler(
                    self.log_dir / json_filename,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count
                )
                json_handler.setLevel(self.log_level)
                json_handler.setFormatter(self.json_formatter)
                handlers.append(json_handler)
        
        return handlers
    
    def _get_logger(self, name: str) -> logging.Logger:
        """Get a specific logger"""
        return logging.getLogger(f"paper_trading.{name}")
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution"""
        self.trade_logger.info("Trade executed", extra=trade_data)
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics"""
        self.performance_logger.info("Performance update", extra=performance_data)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        self.error_logger.error("Error occurred", extra=error_data)
    
    def log_system_event(self, event: str, data: Dict[str, Any] = None):
        """Log system events"""
        self.system_logger.info(f"System event: {event}", extra=data or {})
    
    def log_data_event(self, event: str, data: Dict[str, Any] = None):
        """Log data-related events"""
        data_logger = self._get_logger("data")
        data_logger.info(f"Data event: {event}", extra=data or {})
    
    def log_backtest_event(self, event: str, data: Dict[str, Any] = None):
        """Log backtesting events"""
        backtest_logger = self._get_logger("backtest")
        backtest_logger.info(f"Backtest event: {event}", extra=data or {})
    
    def log_training_event(self, event: str, data: Dict[str, Any] = None):
        """Log training events"""
        training_logger = self._get_logger("training")
        training_logger.info(f"Training event: {event}", extra=data or {})
    
    def log_risk_event(self, event: str, data: Dict[str, Any] = None):
        """Log risk management events"""
        risk_logger = self._get_logger("risk")
        risk_logger.info(f"Risk event: {event}", extra=data or {})
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'log_directory': str(self.log_dir),
            'log_files': [],
            'total_size': 0
        }
        
        for log_file in self.log_dir.glob("*.log*"):
            stats['log_files'].append({
                'name': log_file.name,
                'size': log_file.stat().st_size,
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            })
            stats['total_size'] += log_file.stat().st_size
        
        return stats


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        if hasattr(record, 'action'):
            log_entry['action'] = record.action
        if hasattr(record, 'quantity'):
            log_entry['quantity'] = record.quantity
        if hasattr(record, 'price'):
            log_entry['price'] = record.price
        if hasattr(record, 'portfolio_value'):
            log_entry['portfolio_value'] = record.portfolio_value
        if hasattr(record, 'total_return'):
            log_entry['total_return'] = record.total_return
        if hasattr(record, 'sharpe_ratio'):
            log_entry['sharpe_ratio'] = record.sharpe_ratio
        if hasattr(record, 'max_drawdown'):
            log_entry['max_drawdown'] = record.max_drawdown
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'error_message'):
            log_entry['error_message'] = record.error_message
        if hasattr(record, 'traceback'):
            log_entry['traceback'] = record.traceback
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        return json.dumps(log_entry)


def create_paper_trading_logger(log_dir: str = "./paper_trading_data/logs",
                               log_level: str = "INFO",
                               enable_console: bool = True,
                               enable_file: bool = True) -> PaperTradingLogger:
    """Create and return a paper trading logger instance"""
    return PaperTradingLogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file
    )


# Global logger instance
_paper_trading_logger = None


def get_logger() -> PaperTradingLogger:
    """Get the global paper trading logger instance"""
    global _paper_trading_logger
    if _paper_trading_logger is None:
        _paper_trading_logger = create_paper_trading_logger()
    return _paper_trading_logger


def setup_logging(log_dir: str = "./paper_trading_data/logs",
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = True) -> PaperTradingLogger:
    """Setup global logging configuration"""
    global _paper_trading_logger
    _paper_trading_logger = create_paper_trading_logger(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file
    )
    return _paper_trading_logger 