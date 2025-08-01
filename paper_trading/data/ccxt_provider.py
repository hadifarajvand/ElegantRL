"""
CCXT Data Provider for Paper Trading System
Provides access to real cryptocurrency market data
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class CCXTProvider:
    """
    CCXT-based data provider for cryptocurrency market data.
    Supports multiple exchanges including MEXC and Binance.
    """
    
    def __init__(self, exchange_name: str = 'mexc', api_key: Optional[str] = None, 
                 secret: Optional[str] = None, sandbox: bool = False):
        """
        Initialize CCXT provider.
        
        Args:
            exchange_name: Name of the exchange (default: 'mexc')
            api_key: API key for authenticated requests
            secret: Secret key for authenticated requests
            sandbox: Whether to use sandbox/testnet
        """
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the exchange connection."""
        try:
            # Get exchange class from ccxt
            exchange_class = getattr(ccxt, self.exchange_name)
            
            # Initialize exchange
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            # Load markets
            self.exchange.load_markets()
            logger.info(f"Successfully initialized {self.exchange_name} exchange")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name} exchange: {e}")
            raise
    
    def get_historical_data(self, symbol: str, timeframe: str = '15m', 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None, 
                          limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for data (e.g., '15m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert dates to timestamps
            since = None
            if start_date:
                since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} on {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end_date if provided
            if end_date:
                end_timestamp = datetime.strptime(end_date, '%Y-%m-%d')
                df = df[df.index <= end_timestamp]
            
            logger.info(f"Fetched {len(df)} records for {symbol} on {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time ticker data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with current price and volume data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {e}")
            return {}
    
    def get_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        if df.empty:
            return df
        
        try:
            # Copy DataFrame to avoid modifying original
            df_indicators = df.copy()
            
            # Simple Moving Averages
            df_indicators['sma_20'] = df_indicators['close'].rolling(window=20).mean()
            df_indicators['sma_50'] = df_indicators['close'].rolling(window=50).mean()
            df_indicators['sma_200'] = df_indicators['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df_indicators['ema_12'] = df_indicators['close'].ewm(span=12).mean()
            df_indicators['ema_26'] = df_indicators['close'].ewm(span=26).mean()
            
            # RSI
            delta = df_indicators['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df_indicators['macd'] = df_indicators['ema_12'] - df_indicators['ema_26']
            df_indicators['macd_signal'] = df_indicators['macd'].ewm(span=9).mean()
            df_indicators['macd_histogram'] = df_indicators['macd'] - df_indicators['macd_signal']
            
            # Bollinger Bands
            df_indicators['bb_middle'] = df_indicators['close'].rolling(window=20).mean()
            bb_std = df_indicators['close'].rolling(window=20).std()
            df_indicators['bb_upper'] = df_indicators['bb_middle'] + (bb_std * 2)
            df_indicators['bb_lower'] = df_indicators['bb_middle'] - (bb_std * 2)
            
            # Average True Range (ATR)
            high_low = df_indicators['high'] - df_indicators['low']
            high_close = np.abs(df_indicators['high'] - df_indicators['close'].shift())
            low_close = np.abs(df_indicators['low'] - df_indicators['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df_indicators['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            df_indicators['volume_sma'] = df_indicators['volume'].rolling(window=20).mean()
            df_indicators['volume_ratio'] = df_indicators['volume'] / df_indicators['volume_sma']
            
            # Price change
            df_indicators['price_change'] = df_indicators['close'].pct_change()
            df_indicators['price_change_5'] = df_indicators['close'].pct_change(periods=5)
            df_indicators['price_change_20'] = df_indicators['close'].pct_change(periods=20)
            
            # Remove NaN values
            df_indicators = df_indicators.dropna()
            
            logger.info(f"Calculated technical indicators for {len(df_indicators)} data points")
            return df_indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information and available symbols.
        
        Returns:
            Dictionary with exchange information
        """
        try:
            markets = self.exchange.markets
            symbols = list(markets.keys())
            
            return {
                'exchange': self.exchange_name,
                'symbols': symbols,
                'total_symbols': len(symbols),
                'status': 'connected'
            }
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {}
    
    def test_connection(self) -> bool:
        """
        Test the connection to the exchange.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to fetch ticker for a common pair
            test_symbol = 'BTC/USDT'
            ticker = self.exchange.fetch_ticker(test_symbol)
            logger.info(f"Connection test successful for {self.exchange_name}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.exchange_name}: {e}")
            return False 