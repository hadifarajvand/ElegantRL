"""
Data Manager for Paper Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import os
import pickle
from datetime import datetime, timedelta

from .market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data collection, processing, and storage
    
    Features:
    - Data fetching from multiple sources
    - Data caching and storage
    - Data quality validation
    - Technical indicator calculation
    - Data alignment and preprocessing
    """
    
    def __init__(self, provider: MarketDataProvider, cache_dir: str = "./data_cache"):
        self.provider = provider
        self.cache_dir = cache_dir
        self.cache = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self, symbols: List[str], start_date: str, end_date: str, 
                 interval: str = "1d", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols with caching"""
        data = {}
        
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            
            if use_cache and cache_key in self.cache:
                data[symbol] = self.cache[cache_key]
                logger.info(f"Using cached data for {symbol}")
                continue
            
            # Fetch data from provider
            df = self.provider.get_historical_data(symbol, start_date, end_date, interval)
            
            if not df.empty:
                # Add technical indicators
                df = self.provider.get_technical_indicators(df)
                
                # Validate data quality
                if self._validate_data_quality(df):
                    data[symbol] = df
                    
                    if use_cache:
                        self.cache[cache_key] = df
                        self._save_to_cache(cache_key, df)
                else:
                    logger.warning(f"Data quality validation failed for {symbol}")
            else:
                logger.warning(f"No data retrieved for {symbol}")
        
        return data
    
    def _validate_data_quality(self, df: pd.DataFrame, min_points: int = 100, 
                             max_missing_pct: float = 0.1) -> bool:
        """Validate data quality"""
        if len(df) < min_points:
            logger.warning(f"Data has only {len(df)} points, minimum required: {min_points}")
            return False
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > max_missing_pct:
            logger.warning(f"Data has {missing_pct:.2%} missing values, maximum allowed: {max_missing_pct:.2%}")
            return False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False
        
        return True
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache file"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Data cached to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache file"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Data loaded from cache: {cache_file}")
            return data
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            return None
    
    def align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align data from multiple symbols to common date range"""
        if not data:
            return data
        
        # Find common date range
        start_dates = []
        end_dates = []
        
        for symbol, df in data.items():
            if not df.empty:
                start_dates.append(df.index.min())
                end_dates.append(df.index.max())
        
        if not start_dates:
            logger.warning("No valid data found for alignment")
            return data
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Align data
        aligned_data = {}
        for symbol, df in data.items():
            if not df.empty:
                aligned_df = df[(df.index >= common_start) & (df.index <= common_end)]
                if not aligned_df.empty:
                    aligned_data[symbol] = aligned_df
                    logger.info(f"Aligned {symbol}: {len(aligned_df)} data points")
        
        return aligned_data
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess data for training"""
        processed_data = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            # Remove any remaining NaN values
            df_clean = df.dropna()
            
            # Ensure data is sorted by date
            df_clean = df_clean.sort_index()
            
            # Add basic features if not present
            if 'returns' not in df_clean.columns:
                df_clean['returns'] = df_clean['close'].pct_change()
            
            if 'log_returns' not in df_clean.columns:
                df_clean['log_returns'] = np.log(df_clean['close'] / df_clean['close'].shift(1))
            
            # Add volatility (rolling standard deviation of returns)
            if 'volatility' not in df_clean.columns:
                df_clean['volatility'] = df_clean['returns'].rolling(window=20).std()
            
            processed_data[symbol] = df_clean
            logger.info(f"Preprocessed {symbol}: {len(df_clean)} data points")
        
        return processed_data
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Get summary of data"""
        summary = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            summary[symbol] = {
                'data_points': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'price_range': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'mean': df['close'].mean()
                }
            }
        
        return summary
    
    def save_data(self, data: Dict[str, pd.DataFrame], filename: str):
        """Save data to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def load_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """Load data from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filename}")
            return data
        except FileNotFoundError:
            logger.warning(f"Data file not found: {filename}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        
        # Remove cache files
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
        
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        info = {
            'cache_dir': self.cache_dir,
            'cached_files': len(cache_files),
            'memory_cache_size': len(self.cache),
            'cache_files': cache_files
        }
        
        return info 