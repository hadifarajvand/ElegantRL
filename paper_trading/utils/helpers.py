"""
Helper utility functions for paper trading system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(name: str, log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series"""
    return np.diff(prices) / prices[:-1]


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(portfolio_values) < 2:
        return 0.0
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return np.min(drawdown)


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk"""
    if len(returns) == 0:
        return 0.0
    
    var_percentile = (1 - confidence_level) * 100
    return np.percentile(returns, var_percentile)


def normalize_data(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize data using specified method"""
    if method == "zscore":
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    elif method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    elif method == "log":
        return np.log(data + 1e-8)
    else:
        return data


def validate_data_quality(data: pd.DataFrame, min_points: int = 100, max_missing_pct: float = 0.1) -> bool:
    """Validate data quality"""
    if len(data) < min_points:
        logger.warning(f"Data has only {len(data)} points, minimum required: {min_points}")
        return False
    
    missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
    if missing_pct > max_missing_pct:
        logger.warning(f"Data has {missing_pct:.2%} missing values, maximum allowed: {max_missing_pct:.2%}")
        return False
    
    return True


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.2%}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default 