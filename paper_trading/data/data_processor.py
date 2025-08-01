"""
Data Processor for Paper Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing utilities for financial data
    
    Features:
    - Technical indicator calculation
    - Data normalization
    - Feature engineering
    - Data validation
    - Data transformation
    """
    
    def __init__(self):
        pass
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return df
        
        df = df.copy()
        
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
        df['RSI_14'] = self._calculate_rsi(df['close'], window=14)
        
        # MACD
        macd_data = self._calculate_macd(df['close'])
        df['MACD'] = macd_data['macd']
        df['MACD_signal'] = macd_data['signal']
        df['MACD_hist'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'])
        df['BB_upper'] = bb_data['upper']
        df['BB_middle'] = bb_data['middle']
        df['BB_lower'] = bb_data['lower']
        df['BB_width'] = bb_data['width']
        df['BB_position'] = bb_data['position']
        
        # ATR
        df['ATR_14'] = self._calculate_atr(df, window=14)
        
        # ADX
        adx_data = self._calculate_adx(df, window=14)
        df['ADX_14'] = adx_data['adx']
        df['DI_plus'] = adx_data['di_plus']
        df['DI_minus'] = adx_data['di_minus']
        
        # CCI
        df['CCI_14'] = self._calculate_cci(df, window=14)
        
        # Stochastic Oscillator
        stoch_data = self._calculate_stochastic(df, window=14)
        df['Stoch_K'] = stoch_data['k']
        df['Stoch_D'] = stoch_data['d']
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df, window=14)
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_10'] = df['close'].pct_change(periods=10)
        
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = upper - lower
        position = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width,
            'position': position
        }
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> Dict:
        """Calculate Average Directional Index"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / true_range.rolling(window=window).mean())
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / true_range.rolling(window=window).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'adx': adx,
            'di_plus': plus_di,
            'di_minus': minus_di
        }
    
    def _calculate_cci(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_stochastic(self, df: pd.DataFrame, window: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=window).min()
        highest_high = df['high'].rolling(window=window).max()
        
        k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        
        return {
            'k': k,
            'd': d
        }
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        return williams_r
    
    def normalize_features(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Normalize features using specified method"""
        df_norm = df.copy()
        
        # Select numeric columns for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "zscore":
                df_norm[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            elif method == "minmax":
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            elif method == "log":
                df_norm[col] = np.log(df[col] + 1e-8)
        
        return df_norm
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML models"""
        df_feat = df.copy()
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df_feat['day_of_week'] = df.index.dayofweek
            df_feat['month'] = df.index.month
            df_feat['quarter'] = df.index.quarter
            df_feat['year'] = df.index.year
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df_feat[f'close_lag_{lag}'] = df['close'].shift(lag)
            df_feat[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df_feat[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df_feat[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df_feat[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean() if 'volume' in df.columns else 0
        
        # Price ratios
        df_feat['price_to_sma_20'] = df['close'] / df['SMA_20']
        df_feat['price_to_sma_50'] = df['close'] / df['SMA_50']
        
        return df_feat
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                logger.error(f"Negative or zero prices found in {col}")
                return False
        
        # Check for logical price relationships
        if not ((df['low'] <= df['close']) & (df['close'] <= df['high'])).all():
            logger.error("Price relationships are not logical")
            return False
        
        # Check for excessive missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.1:
            logger.warning(f"High missing value percentage: {missing_pct:.2%}")
        
        return True 