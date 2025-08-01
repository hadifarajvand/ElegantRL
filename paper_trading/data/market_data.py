"""
Market Data Interface for Paper Trading System
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import requests
import time
from datetime import datetime, timedelta
import logging

# Import CCXT provider
try:
    from .ccxt_provider import CCXTProvider, create_ccxt_provider
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: CCXT not available, cryptocurrency data will not be available")

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Base class for market data providers"""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical market data"""
        raise NotImplementedError
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get real-time market data"""
        raise NotImplementedError
    
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self):
        super().__init__()
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            data = data.reset_index()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get real-time prices"""
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'regularMarketPrice' in info and info['regularMarketPrice']:
                    prices[symbol] = info['regularMarketPrice']
            except Exception as e:
                logger.error(f"Error fetching real-time data for {symbol}: {e}")
        
        return prices
    
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df['close'].ewm(span=5).mean()
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR_14'] = true_range.rolling(window=14).mean()
        
        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr = true_range
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX_14'] = dx.rolling(window=14).mean()
        
        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI_14'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Stochastic Oscillator
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        # Price changes
        df['Price_Change'] = df['close'].pct_change()
        df['Price_Change_5'] = df['close'].pct_change(periods=5)
        df['Price_Change_10'] = df['close'].pct_change(periods=10)
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Clean up NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df


class CCXTDataProvider(MarketDataProvider):
    """CCXT data provider for cryptocurrency exchanges"""
    
    def __init__(self, exchange_name: str = 'binance', api_key: str = "", 
                 api_secret: str = "", sandbox: bool = False):
        super().__init__(api_key, api_secret)
        
        if not CCXT_AVAILABLE:
            raise ImportError("CCXT is not available. Please install it with: pip install ccxt")
        
        self.ccxt_provider = create_ccxt_provider(
            exchange_name=exchange_name,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox
        )
        self.exchange_name = exchange_name
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data from CCXT exchange"""
        try:
            # Convert interval format if needed
            timeframe = self._convert_interval(interval)
            
            data = self.ccxt_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching CCXT data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get real-time prices from CCXT exchange"""
        return self.ccxt_provider.get_realtime_data(symbols)
    
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using CCXT provider"""
        return self.ccxt_provider.get_technical_indicators(data)
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval format to CCXT timeframe"""
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        return interval_map.get(interval, '1d')
    
    def get_available_symbols(self) -> List[str]:
        """Get available trading symbols"""
        return self.ccxt_provider.get_available_symbols()
    
    def test_connection(self) -> bool:
        """Test connection to the exchange"""
        return self.ccxt_provider.test_connection()


class AlpacaProvider(MarketDataProvider):
    """Alpaca data provider"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        super().__init__(api_key, api_secret)
        self.base_url = base_url
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Alpaca"""
        try:
            # Implementation for Alpaca API
            # This is a placeholder - you would need to implement the actual API calls
            logger.warning("Alpaca provider not fully implemented")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get real-time prices from Alpaca"""
        # Placeholder implementation
        return {}


class DataManager:
    """Data manager for handling multiple data sources"""
    
    def __init__(self, provider: MarketDataProvider, cache_dir: str = "./data_cache"):
        self.provider = provider
        self.cache_dir = cache_dir
        self.cache = {}
        
        # Create cache directory if it doesn't exist
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self, symbols: List[str], start_date: str, end_date: str, 
                 interval: str = "1d", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols with caching"""
        data = {}
        
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            
            # Check cache first
            if use_cache and cache_key in self.cache:
                data[symbol] = self.cache[cache_key]
                logger.info(f"Using cached data for {symbol}")
                continue
            
            # Fetch data from provider
            symbol_data = self.provider.get_historical_data(symbol, start_date, end_date, interval)
            
            if not symbol_data.empty:
                # Add technical indicators
                symbol_data = self.provider.get_technical_indicators(symbol_data)
                
                # Cache the data
                if use_cache:
                    self.cache[cache_key] = symbol_data
                
                data[symbol] = symbol_data
                logger.info(f"Fetched {len(symbol_data)} records for {symbol}")
            else:
                logger.warning(f"No data retrieved for {symbol}")
        
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], filename: str):
        """Save data to cache file"""
        import pickle
        filepath = f"{self.cache_dir}/{filename}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """Load data from cache file"""
        import pickle
        filepath = f"{self.cache_dir}/{filename}.pkl"
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from {filepath}")
            return data
        except FileNotFoundError:
            logger.warning(f"Cache file not found: {filepath}")
            return {}


# Factory functions for creating data providers
def create_yahoo_provider() -> YahooFinanceProvider:
    """Create Yahoo Finance provider"""
    return YahooFinanceProvider()


def create_ccxt_provider(exchange_name: str = 'binance', api_key: str = "", 
                        api_secret: str = "", sandbox: bool = False) -> CCXTDataProvider:
    """Create CCXT provider"""
    if not CCXT_AVAILABLE:
        raise ImportError("CCXT is not available. Please install it with: pip install ccxt")
    return CCXTDataProvider(exchange_name, api_key, api_secret, sandbox)


def create_alpaca_provider(api_key: str, api_secret: str, 
                          base_url: str = "https://paper-api.alpaca.markets") -> AlpacaProvider:
    """Create Alpaca provider"""
    return AlpacaProvider(api_key, api_secret, base_url) 