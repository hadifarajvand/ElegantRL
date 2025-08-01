"""
Trading Configuration for Paper Trading System
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import yaml


@dataclass
class TradingConfig:
    """Configuration for paper trading system"""
    
    # Trading Parameters
    initial_capital: float = 1000000.0  # $1M initial capital
    max_stock_quantity: int = 100  # Maximum shares per stock
    transaction_cost_pct: float = 0.001  # 0.1% transaction cost
    slippage_pct: float = 0.0005  # 0.05% slippage
    
    # Risk Management
    max_position_size: float = 0.2  # Max 20% in single position
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    
    # Portfolio Constraints
    min_cash_reserve: float = 0.1  # Keep 10% cash reserve
    max_leverage: float = 1.5  # Max 1.5x leverage
    
    # Trading Schedule
    trading_hours: Tuple[str, str] = ("09:30", "16:00")  # Market hours
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    
    # Data Parameters
    data_source: str = "yahoo"  # yahoo, alpaca, polygon
    update_frequency: str = "1min"  # 1min, 5min, 15min, 1hour, daily
    
    # Technical Indicators
    technical_indicators: List[str] = None
    
    # API Configuration (for live trading)
    api_key: str = ""
    api_secret: str = ""
    api_base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "wss://data.alpaca.markets"
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
                'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower',
                'ATR_14', 'ADX_14', 'CCI_14'
            ]
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'TradingConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'initial_capital': self.initial_capital,
            'max_stock_quantity': self.max_stock_quantity,
            'transaction_cost_pct': self.transaction_cost_pct,
            'slippage_pct': self.slippage_pct,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'min_cash_reserve': self.min_cash_reserve,
            'max_leverage': self.max_leverage,
            'trading_hours': self.trading_hours,
            'rebalance_frequency': self.rebalance_frequency,
            'data_source': self.data_source,
            'update_frequency': self.update_frequency,
            'technical_indicators': self.technical_indicators,
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'api_base_url': self.api_base_url,
            'data_url': self.data_url
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configurations for different market types
DEFAULT_US_MARKET_CONFIG = TradingConfig(
    initial_capital=1000000.0,
    trading_hours=("09:30", "16:00"),
    data_source="alpaca"
)

DEFAULT_CHINA_MARKET_CONFIG = TradingConfig(
    initial_capital=1000000.0,
    trading_hours=("09:30", "15:00"),
    data_source="yahoo"
)

DEFAULT_CRYPTO_CONFIG = TradingConfig(
    initial_capital=100000.0,
    trading_hours=("00:00", "23:59"),  # 24/7 trading
    data_source="yahoo",
    update_frequency="5min"
) 