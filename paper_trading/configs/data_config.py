"""
Data Configuration for Paper Trading System
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml


@dataclass
class DataConfig:
    """Configuration for data sources and processing"""
    
    # Data Source
    data_source: str = "yahoo"  # yahoo, alpaca, polygon
    
    # Data Parameters
    update_frequency: str = "1min"  # 1min, 5min, 15min, 1hour, daily
    cache_enabled: bool = True
    cache_dir: str = "./data_cache"
    
    # Technical Indicators
    technical_indicators: List[str] = None
    
    # Data Quality
    min_data_points: int = 100
    max_missing_pct: float = 0.1  # 10% missing data threshold
    
    # API Configuration
    api_key: str = ""
    api_secret: str = ""
    api_base_url: str = ""
    data_url: str = ""
    
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
    def from_yaml(cls, file_path: str) -> 'DataConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data_source': self.data_source,
            'update_frequency': self.update_frequency,
            'cache_enabled': self.cache_enabled,
            'cache_dir': self.cache_dir,
            'technical_indicators': self.technical_indicators,
            'min_data_points': self.min_data_points,
            'max_missing_pct': self.max_missing_pct,
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'api_base_url': self.api_base_url,
            'data_url': self.data_url
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configurations for different data sources
DEFAULT_YAHOO_CONFIG = DataConfig(
    data_source="yahoo",
    update_frequency="1min",
    cache_enabled=True
)

DEFAULT_ALPACA_CONFIG = DataConfig(
    data_source="alpaca",
    update_frequency="1min",
    cache_enabled=True,
    api_base_url="https://paper-api.alpaca.markets",
    data_url="wss://data.alpaca.markets"
) 