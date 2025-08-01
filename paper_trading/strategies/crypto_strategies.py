"""
Advanced Cryptocurrency Trading Strategies
Implements sophisticated trading strategies for cryptocurrency markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CryptoStrategyBase:
    """Base class for cryptocurrency trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.positions = {}
        self.signals = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate trading signals for all symbols"""
        raise NotImplementedError
    
    def get_action(self, state: np.ndarray, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert signals to actions"""
        raise NotImplementedError


class CryptoMomentumStrategy(CryptoStrategyBase):
    """
    Momentum-based cryptocurrency trading strategy
    Uses RSI, MACD, and price momentum for signal generation
    """
    
    def __init__(self, 
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 macd_signal_threshold: float = 0.0,
                 momentum_period: int = 14,
                 position_size: float = 0.3):
        
        super().__init__("CryptoMomentum", {
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_signal_threshold': macd_signal_threshold,
            'momentum_period': momentum_period,
            'position_size': position_size
        })
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate momentum-based signals"""
        signals = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            # Initialize signal array
            signal = np.zeros(len(df))
            
            # RSI signals
            if 'RSI_14' in df.columns:
                rsi = df['RSI_14'].values
                rsi_buy = (rsi < self.parameters['rsi_oversold']).astype(float)
                rsi_sell = (rsi > self.parameters['rsi_overbought']).astype(float)
                signal += rsi_buy - rsi_sell
            
            # MACD signals
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].values
                macd_signal = df['MACD_signal'].values
                macd_cross = (macd > macd_signal).astype(float) - (macd < macd_signal).astype(float)
                signal += macd_cross
            
            # Price momentum
            if 'Price_Change' in df.columns:
                momentum = df['Price_Change'].rolling(
                    window=self.parameters['momentum_period']
                ).mean().values
                momentum_signal = np.where(momentum > 0, 0.5, -0.5)
                signal += np.nan_to_num(momentum_signal, 0)
            
            # Normalize signals to [-1, 1]
            signal = np.clip(signal, -1, 1)
            signals[symbol] = signal
        
        return signals
    
    def get_action(self, state: np.ndarray, signals: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert signals to actions"""
        # This would be implemented based on the specific environment
        # For now, return a simple conversion
        actions = []
        for symbol, signal in signals.items():
            # Convert signal to action (simplified)
            action = signal * self.parameters['position_size']
            actions.append(action)
        
        return np.array(actions)


class CryptoMeanReversionStrategy(CryptoStrategyBase):
    """
    Mean reversion strategy for cryptocurrency trading
    Uses Bollinger Bands and moving averages
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 ma_short: int = 10,
                 ma_long: int = 50,
                 position_size: float = 0.3):
        
        super().__init__("CryptoMeanReversion", {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'position_size': position_size
        })
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate mean reversion signals"""
        signals = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            signal = np.zeros(len(df))
            
            # Bollinger Bands signals
            if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'close']):
                close = df['close'].values
                bb_upper = df['BB_upper'].values
                bb_lower = df['BB_lower'].values
                
                # Buy when price touches lower band
                bb_buy = (close <= bb_lower).astype(float)
                # Sell when price touches upper band
                bb_sell = (close >= bb_upper).astype(float)
                
                signal += bb_buy - bb_sell
            
            # Moving average crossover
            if f'SMA_{self.parameters["ma_short"]}' in df.columns and f'SMA_{self.parameters["ma_long"]}' in df.columns:
                ma_short = df[f'SMA_{self.parameters["ma_short"]}'].values
                ma_long = df[f'SMA_{self.parameters["ma_long"]}'].values
                
                # Buy when short MA crosses above long MA
                ma_cross_buy = (ma_short > ma_long).astype(float)
                # Sell when short MA crosses below long MA
                ma_cross_sell = (ma_short < ma_long).astype(float)
                
                signal += ma_cross_buy - ma_cross_sell
            
            # Normalize signals
            signal = np.clip(signal, -1, 1)
            signals[symbol] = signal
        
        return signals


class CryptoVolatilityStrategy(CryptoStrategyBase):
    """
    Volatility-based cryptocurrency trading strategy
    Uses ATR and volatility indicators for position sizing
    """
    
    def __init__(self,
                 atr_period: int = 14,
                 volatility_threshold: float = 0.02,
                 max_position_size: float = 0.5):
        
        super().__init__("CryptoVolatility", {
            'atr_period': atr_period,
            'volatility_threshold': volatility_threshold,
            'max_position_size': max_position_size
        })
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate volatility-based signals"""
        signals = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
            
            signal = np.zeros(len(df))
            
            # ATR-based position sizing
            if 'ATR_14' in df.columns and 'close' in df.columns:
                atr = df['ATR_14'].values
                close = df['close'].values
                
                # Calculate volatility ratio
                volatility_ratio = atr / close
                
                # Adjust position size based on volatility
                position_adjustment = np.where(
                    volatility_ratio > self.parameters['volatility_threshold'],
                    -0.5,  # Reduce position in high volatility
                    0.5    # Increase position in low volatility
                )
                
                signal += position_adjustment
            
            # Volatility breakout signals
            if 'Volatility' in df.columns:
                volatility = df['Volatility'].values
                
                # Buy when volatility is low (stability)
                low_vol_buy = (volatility < self.parameters['volatility_threshold']).astype(float)
                # Sell when volatility is high (uncertainty)
                high_vol_sell = (volatility > self.parameters['volatility_threshold'] * 2).astype(float)
                
                signal += low_vol_buy - high_vol_sell
            
            # Normalize signals
            signal = np.clip(signal, -1, 1)
            signals[symbol] = signal
        
        return signals


class CryptoArbitrageStrategy(CryptoStrategyBase):
    """
    Arbitrage strategy for cryptocurrency trading
    Exploits price differences between correlated assets
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.7,
                 price_diff_threshold: float = 0.02,
                 position_size: float = 0.2):
        
        super().__init__("CryptoArbitrage", {
            'correlation_threshold': correlation_threshold,
            'price_diff_threshold': price_diff_threshold,
            'position_size': position_size
        })
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate arbitrage signals"""
        signals = {}
        
        if len(data) < 2:
            return signals
        
        # Calculate correlations between assets
        symbols = list(data.keys())
        correlations = {}
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                if 'close' in data[symbol1].columns and 'close' in data[symbol2].columns:
                    corr = data[symbol1]['close'].corr(data[symbol2]['close'])
                    correlations[(symbol1, symbol2)] = corr
        
        # Generate signals based on price divergences
        for symbol, df in data.items():
            if df.empty or 'close' not in df.columns:
                continue
            
            signal = np.zeros(len(df))
            
            # Find correlated pairs
            for (symbol1, symbol2), corr in correlations.items():
                if corr > self.parameters['correlation_threshold']:
                    if symbol in [symbol1, symbol2]:
                        other_symbol = symbol2 if symbol == symbol1 else symbol1
                        
                        if other_symbol in data and 'close' in data[other_symbol].columns:
                            # Calculate price ratio
                            price_ratio = df['close'] / data[other_symbol]['close']
                            
                            # Detect divergences
                            mean_ratio = price_ratio.rolling(window=20).mean()
                            std_ratio = price_ratio.rolling(window=20).std()
                            
                            # Buy when ratio is below mean (underpriced)
                            buy_signal = (price_ratio < mean_ratio - std_ratio).astype(float)
                            # Sell when ratio is above mean (overpriced)
                            sell_signal = (price_ratio > mean_ratio + std_ratio).astype(float)
                            
                            signal += buy_signal - sell_signal
            
            # Normalize signals
            signal = np.clip(signal, -1, 1)
            signals[symbol] = signal
        
        return signals


class CryptoMultiStrategy(CryptoStrategyBase):
    """
    Multi-strategy approach combining multiple strategies
    """
    
    def __init__(self, strategies: List[CryptoStrategyBase], weights: List[float] = None):
        super().__init__("CryptoMultiStrategy", {})
        
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Generate combined signals from multiple strategies"""
        all_signals = {}
        
        # Get signals from each strategy
        for strategy in self.strategies:
            strategy_signals = strategy.generate_signals(data)
            
            for symbol, signal in strategy_signals.items():
                if symbol not in all_signals:
                    all_signals[symbol] = []
                all_signals[symbol].append(signal)
        
        # Combine signals with weights
        combined_signals = {}
        for symbol, signal_list in all_signals.items():
            if signal_list:
                # Weighted average of signals
                weighted_signal = np.zeros_like(signal_list[0])
                for i, signal in enumerate(signal_list):
                    weighted_signal += signal * self.weights[i]
                
                # Normalize
                weighted_signal = np.clip(weighted_signal, -1, 1)
                combined_signals[symbol] = weighted_signal
        
        return combined_signals


class CryptoStrategyManager:
    """
    Manager for cryptocurrency trading strategies
    """
    
    def __init__(self):
        self.strategies = {}
        self.active_strategy = None
    
    def add_strategy(self, name: str, strategy: CryptoStrategyBase):
        """Add a strategy to the manager"""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
    
    def set_active_strategy(self, name: str):
        """Set the active strategy"""
        if name in self.strategies:
            self.active_strategy = self.strategies[name]
            logger.info(f"Set active strategy: {name}")
        else:
            logger.error(f"Strategy not found: {name}")
    
    def get_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Get signals from active strategy"""
        if self.active_strategy is None:
            logger.warning("No active strategy set")
            return {}
        
        return self.active_strategy.generate_signals(data)
    
    def create_momentum_strategy(self, **kwargs) -> CryptoMomentumStrategy:
        """Create a momentum strategy"""
        return CryptoMomentumStrategy(**kwargs)
    
    def create_mean_reversion_strategy(self, **kwargs) -> CryptoMeanReversionStrategy:
        """Create a mean reversion strategy"""
        return CryptoMeanReversionStrategy(**kwargs)
    
    def create_volatility_strategy(self, **kwargs) -> CryptoVolatilityStrategy:
        """Create a volatility strategy"""
        return CryptoVolatilityStrategy(**kwargs)
    
    def create_arbitrage_strategy(self, **kwargs) -> CryptoArbitrageStrategy:
        """Create an arbitrage strategy"""
        return CryptoArbitrageStrategy(**kwargs)
    
    def create_multi_strategy(self, strategies: List[CryptoStrategyBase], 
                            weights: List[float] = None) -> CryptoMultiStrategy:
        """Create a multi-strategy approach"""
        return CryptoMultiStrategy(strategies, weights)


def create_crypto_strategies() -> CryptoStrategyManager:
    """Create and configure cryptocurrency trading strategies"""
    manager = CryptoStrategyManager()
    
    # Add individual strategies
    momentum = manager.create_momentum_strategy(
        rsi_oversold=25,
        rsi_overbought=75,
        momentum_period=14,
        position_size=0.3
    )
    manager.add_strategy("Momentum", momentum)
    
    mean_reversion = manager.create_mean_reversion_strategy(
        bb_period=20,
        ma_short=10,
        ma_long=50,
        position_size=0.3
    )
    manager.add_strategy("MeanReversion", mean_reversion)
    
    volatility = manager.create_volatility_strategy(
        atr_period=14,
        volatility_threshold=0.03,
        max_position_size=0.4
    )
    manager.add_strategy("Volatility", volatility)
    
    arbitrage = manager.create_arbitrage_strategy(
        correlation_threshold=0.8,
        price_diff_threshold=0.01,
        position_size=0.2
    )
    manager.add_strategy("Arbitrage", arbitrage)
    
    # Create multi-strategy
    multi_strategy = manager.create_multi_strategy(
        strategies=[momentum, mean_reversion, volatility],
        weights=[0.4, 0.3, 0.3]
    )
    manager.add_strategy("MultiStrategy", multi_strategy)
    
    return manager


if __name__ == "__main__":
    # Test strategy creation
    manager = create_crypto_strategies()
    print("✅ Cryptocurrency trading strategies created successfully!")
    print(f"Available strategies: {list(manager.strategies.keys())}") 