"""
Advanced Technical Indicators for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    from scipy import stats
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some advanced features may be limited")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: talib not available, some technical indicators may be limited")

logger = logging.getLogger(__name__)


class AdvancedTechnicalIndicators:
    """
    Advanced Technical Indicators Calculator
    
    Features:
    - Comprehensive technical analysis
    - Multi-timeframe indicators
    - Statistical indicators
    - Machine learning features
    - Custom indicator combinations
    """
    
    def __init__(self):
        self.indicators = {}
        self.lookback_periods = [5, 10, 20, 50, 100]
        
        logger.info("AdvancedTechnicalIndicators initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available technical indicators"""
        result_df = df.copy()
        
        # Price-based indicators
        result_df = self._add_price_indicators(result_df)
        
        # Volume-based indicators
        result_df = self._add_volume_indicators(result_df)
        
        # Momentum indicators
        result_df = self._add_momentum_indicators(result_df)
        
        # Volatility indicators
        result_df = self._add_volatility_indicators(result_df)
        
        # Trend indicators
        result_df = self._add_trend_indicators(result_df)
        
        # Statistical indicators
        result_df = self._add_statistical_indicators(result_df)
        
        # Machine learning features
        result_df = self._add_ml_features(result_df)
        
        # Custom combinations
        result_df = self._add_custom_combinations(result_df)
        
        logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} additional indicators")
        return result_df
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical indicators"""
        # Moving averages
        for period in self.lookback_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'wma_{period}'] = self._weighted_moving_average(df['close'], period)
        
        # Price levels
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['support_1'] = 2 * df['pivot_point'] - df['high']
        df['resistance_1'] = 2 * df['pivot_point'] - df['low']
        
        # Price channels
        for period in [20, 50]:
            df[f'upper_channel_{period}'] = df['high'].rolling(window=period).max()
            df[f'lower_channel_{period}'] = df['low'].rolling(window=period).min()
            df[f'channel_width_{period}'] = df[f'upper_channel_{period}'] - df[f'lower_channel_{period}']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based technical indicators"""
        # Volume moving averages
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume price analysis
        df['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        df['volume_price_trend_cum'] = df['volume_price_trend'].cumsum()
        
        # On-balance volume
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        
        # Money flow index
        df['mfi'] = self._calculate_money_flow_index(df)
        
        # Volume weighted average price
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators"""
        # RSI
        for period in [14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)
        
        # CCI (Commodity Channel Index)
        df['cci'] = self._calculate_cci(df)
        
        # Rate of change
        for period in [10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based technical indicators"""
        # Bollinger Bands
        for period in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], period)
            df[f'bb_upper_{period}'] = bb_upper
            df[f'bb_middle_{period}'] = bb_middle
            df[f'bb_lower_{period}'] = bb_lower
            df[f'bb_width_{period}'] = bb_upper - bb_lower
            df[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range
        df['atr'] = self._calculate_atr(df)
        
        # Historical volatility
        for period in [20, 50]:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std() * np.sqrt(252)
        
        # Keltner Channels
        df['keltner_upper'], df['keltner_middle'], df['keltner_lower'] = self._calculate_keltner_channels(df)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical indicators"""
        # ADX (Average Directional Index)
        df['adx'] = self._calculate_adx(df)
        
        # Parabolic SAR
        df['psar'] = self._calculate_parabolic_sar(df)
        
        # Ichimoku Cloud
        ichimoku = self._calculate_ichimoku(df)
        for key, value in ichimoku.items():
            df[f'ichimoku_{key}'] = value
        
        # Supertrend
        df['supertrend'] = self._calculate_supertrend(df)
        
        # Price action patterns
        df['doji'] = self._detect_doji(df)
        df['hammer'] = self._detect_hammer(df)
        df['engulfing'] = self._detect_engulfing(df)
        
        return df
    
    def _add_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical indicators"""
        # Z-score
        for period in [20, 50]:
            df[f'zscore_{period}'] = (df['close'] - df['close'].rolling(window=period).mean()) / df['close'].rolling(window=period).std()
        
        # Percentile rank
        for period in [20, 50]:
            df[f'percentile_rank_{period}'] = df['close'].rolling(window=period).rank(pct=True)
        
        # Skewness and kurtosis
        for period in [20, 50]:
            df[f'skewness_{period}'] = df['close'].rolling(window=period).skew()
            df[f'kurtosis_{period}'] = df['close'].rolling(window=period).kurt()
        
        # Linear regression
        for period in [20, 50]:
            df[f'linear_regression_{period}'] = self._calculate_linear_regression(df['close'], period)
        
        return df
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add machine learning features"""
        # Price momentum features
        for period in [1, 3, 5, 10]:
            df[f'price_momentum_{period}'] = df['close'].pct_change(period)
            df[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
        
        # Rolling statistics
        for period in [20, 50]:
            df[f'price_std_{period}'] = df['close'].rolling(window=period).std()
            df[f'volume_std_{period}'] = df['volume'].rolling(window=period).std()
        
        # Cross-sectional features
        df['price_rank'] = df['close'].rank(pct=True)
        df['volume_rank'] = df['volume'].rank(pct=True)
        
        # Time-based features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['quarter'] = pd.to_datetime(df.index).quarter
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df
    
    def _add_custom_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicator combinations"""
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['sma_20']
        
        # Volume price confirmation
        df['volume_price_confirmation'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['volume'] > df['volume_sma_20']), 1,
            np.where((df['close'] < df['close'].shift(1)) & (df['volume'] > df['volume_sma_20']), -1, 0)
        )
        
        # Momentum divergence
        df['momentum_divergence'] = np.where(
            (df['close'] > df['close'].shift(1)) & (df['rsi_14'] < df['rsi_14'].shift(1)), -1,
            np.where((df['close'] < df['close'].shift(1)) & (df['rsi_14'] > df['rsi_14'].shift(1)), 1, 0)
        )
        
        # Volatility regime
        df['volatility_regime'] = np.where(
            df['atr'] > df['atr'].rolling(window=20).mean(), 'high',
            np.where(df['atr'] < df['atr'].rolling(window=20).mean() * 0.5, 'low', 'normal')
        )
        
        return df
    
    # Helper methods for individual indicators
    def _weighted_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate weighted moving average"""
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    def _calculate_bollinger_bands(self, close: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        atr = self._calculate_atr(df, period)
        
        upper_channel = typical_price + (multiplier * atr)
        lower_channel = typical_price - (multiplier * atr)
        
        return upper_channel, typical_price, lower_channel
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = self._calculate_atr(df, period)
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period).mean()
        
        return adx
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR"""
        # Simplified Parabolic SAR
        psar = pd.Series(index=df.index, dtype=float)
        psar.iloc[0] = df['low'].iloc[0]
        
        af = acceleration
        ep = df['high'].iloc[0]
        long = True
        
        for i in range(1, len(df)):
            if long:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af = min(af + acceleration, maximum)
                if df['low'].iloc[i] < psar.iloc[i]:
                    long = False
                    psar.iloc[i] = ep
                    ep = df['low'].iloc[i]
                    af = acceleration
            else:
                psar.iloc[i] = psar.iloc[i-1] - af * (psar.iloc[i-1] - ep)
                if df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af = min(af + acceleration, maximum)
                if df['high'].iloc[i] > psar.iloc[i]:
                    long = True
                    psar.iloc[i] = ep
                    ep = df['high'].iloc[i]
                    af = acceleration
        
        return psar
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud"""
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        chikou_span = df['close'].shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.Series:
        """Calculate Supertrend"""
        atr = self._calculate_atr(df, period)
        
        basic_upper = ((df['high'] + df['low']) / 2) + (multiplier * atr)
        basic_lower = ((df['high'] + df['low']) / 2) - (multiplier * atr)
        
        final_upper = pd.Series(index=df.index, dtype=float)
        final_lower = pd.Series(index=df.index, dtype=float)
        
        for i in range(period, len(df)):
            final_upper.iloc[i] = basic_upper.iloc[i] if (
                basic_upper.iloc[i] < final_upper.iloc[i-1] or df['close'].iloc[i-1] > final_upper.iloc[i-1]
            ) else final_upper.iloc[i-1]
            
            final_lower.iloc[i] = basic_lower.iloc[i] if (
                basic_lower.iloc[i] > final_lower.iloc[i-1] or df['close'].iloc[i-1] < final_lower.iloc[i-1]
            ) else final_lower.iloc[i-1]
        
        supertrend = pd.Series(index=df.index, dtype=float)
        for i in range(period, len(df)):
            supertrend.iloc[i] = final_upper.iloc[i] if (
                supertrend.iloc[i-1] == final_upper.iloc[i-1] and df['close'].iloc[i] <= final_upper.iloc[i]
            ) else final_lower.iloc[i]
        
        return supertrend
    
    def _detect_doji(self, df: pd.DataFrame, tolerance: float = 0.1) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        doji = (body_size <= tolerance * total_range)
        return doji.astype(int)
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick pattern"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        
        hammer = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
        return hammer.astype(int)
    
    def _detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect Engulfing candlestick pattern"""
        bullish_engulfing = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous bearish
            (df['close'] > df['open']) &  # Current bullish
            (df['open'] < df['close'].shift(1)) &  # Current open below previous close
            (df['close'] > df['open'].shift(1))  # Current close above previous open
        )
        
        bearish_engulfing = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous bullish
            (df['close'] < df['open']) &  # Current bearish
            (df['open'] > df['close'].shift(1)) &  # Current open above previous close
            (df['close'] < df['open'].shift(1))  # Current close below previous open
        )
        
        return (bullish_engulfing | bearish_engulfing).astype(int)
    
    def _calculate_linear_regression(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate linear regression slope"""
        def slope(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return series.rolling(window=period).apply(slope)


def create_advanced_indicators() -> AdvancedTechnicalIndicators:
    """Convenience function to create advanced technical indicators calculator"""
    return AdvancedTechnicalIndicators() 