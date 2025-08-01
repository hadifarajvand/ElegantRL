"""
Multi-Strategy Framework for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import threading
import time

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy type enumeration"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    QUANTITATIVE = "quantitative"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class StrategySignal:
    """Strategy signal data class"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    quantity: float
    price: float
    strategy_name: str
    timestamp: Any
    metadata: Dict[str, Any]


@dataclass
class StrategyPerformance:
    """Strategy performance data class"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float


class BaseStrategy(ABC):
    """Base strategy class"""
    
    def __init__(self, name: str, symbols: List[str], **kwargs):
        self.name = name
        self.symbols = symbols
        self.is_active = True
        self.performance_history = []
        self.signal_history = []
        
        # Strategy parameters
        self.min_confidence = kwargs.get('min_confidence', 0.6)
        self.max_position_size = kwargs.get('max_position_size', 0.1)
        self.stop_loss = kwargs.get('stop_loss', 0.05)
        self.take_profit = kwargs.get('take_profit', 0.1)
        
        logger.info(f"Strategy {name} initialized for {len(symbols)} symbols")
    
    @abstractmethod
    def generate_signals(self, market_data: Dict) -> List[StrategySignal]:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def update_parameters(self, market_data: Dict):
        """Update strategy parameters based on market conditions"""
        pass
    
    def calculate_performance(self, portfolio: Dict) -> StrategyPerformance:
        """Calculate strategy performance"""
        try:
            # Simplified performance calculation
            total_return = 0.1  # Placeholder
            sharpe_ratio = 0.5  # Placeholder
            max_drawdown = 0.05  # Placeholder
            win_rate = 0.6  # Placeholder
            profit_factor = 1.2  # Placeholder
            total_trades = len(self.signal_history)
            avg_trade_duration = 5.0  # Placeholder
            volatility = 0.15  # Placeholder
            
            return StrategyPerformance(
                strategy_name=self.name,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance for {self.name}: {e}")
            return StrategyPerformance(
                strategy_name=self.name,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                volatility=0.0
            )


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, symbols=["AAPL"], lookback_period=20, threshold=0.02, **kwargs):
        super().__init__("Momentum", symbols, **kwargs)
        
        # Momentum parameters
        self.lookback_period = lookback_period
        self.momentum_threshold = threshold
        self.volume_threshold = kwargs.get('volume_threshold', 1.5)
        
    def generate_signals(self, market_data: Dict) -> List[StrategySignal]:
        """Generate momentum-based signals"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if not isinstance(data, pd.DataFrame) or len(data) < self.lookback_period:
                continue
            
            # Calculate momentum indicators
            price_momentum = self._calculate_price_momentum(data)
            volume_momentum = self._calculate_volume_momentum(data)
            
            # Generate signal
            signal = self._generate_momentum_signal(symbol, data, price_momentum, volume_momentum)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_price_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum"""
        if len(data) < self.lookback_period:
            return 0.0
        
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-self.lookback_period]
        
        return (current_price - past_price) / past_price
    
    def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """Calculate volume momentum"""
        if len(data) < self.lookback_period:
            return 0.0
        
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=self.lookback_period).mean().iloc[-1]
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _generate_momentum_signal(self, symbol: str, data: pd.DataFrame, 
                                 price_momentum: float, volume_momentum: float) -> Optional[StrategySignal]:
        """Generate momentum signal"""
        current_price = data['close'].iloc[-1]
        
        # Determine action and confidence
        if price_momentum > self.momentum_threshold and volume_momentum > self.volume_threshold:
            action = 'buy'
            confidence = min(0.9, 0.5 + abs(price_momentum) * 2)
        elif price_momentum < -self.momentum_threshold and volume_momentum > self.volume_threshold:
            action = 'sell'
            confidence = min(0.9, 0.5 + abs(price_momentum) * 2)
        else:
            action = 'hold'
            confidence = 0.5
        
        if confidence < self.min_confidence:
            return None
        
        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            quantity=0.0,  # Will be calculated by position sizing
            price=current_price,
            strategy_name=self.name,
            timestamp=data.index[-1] if hasattr(data, 'index') else None,
            metadata={
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'lookback_period': self.lookback_period
            }
        )
    
    def update_parameters(self, market_data: Dict):
        """Update momentum parameters based on market conditions"""
        # Adjust momentum threshold based on market volatility
        volatility = self._calculate_market_volatility(market_data)
        if volatility > 0.3:  # High volatility
            self.momentum_threshold *= 1.2
        elif volatility < 0.1:  # Low volatility
            self.momentum_threshold *= 0.8
    
    def _calculate_market_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility"""
        volatilities = []
        for symbol, data in market_data.items():
            if isinstance(data, pd.DataFrame) and len(data) > 20:
                returns = data['close'].pct_change().dropna()
                volatilities.append(returns.std())
        
        return np.mean(volatilities) if volatilities else 0.2


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, symbols=["AAPL"], lookback_period=50, threshold=2.0, **kwargs):
        super().__init__("MeanReversion", symbols, **kwargs)
        
        # Mean reversion parameters
        self.lookback_period = lookback_period
        self.z_score_threshold = threshold
        self.reversion_strength = kwargs.get('reversion_strength', 0.5)
        
    def generate_signals(self, market_data: Dict) -> List[StrategySignal]:
        """Generate mean reversion signals"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if not isinstance(data, pd.DataFrame) or len(data) < self.lookback_period:
                continue
            
            # Calculate mean reversion indicators
            z_score = self._calculate_z_score(data)
            mean_reversion_signal = self._calculate_mean_reversion_signal(data)
            
            # Generate signal
            signal = self._generate_reversion_signal(symbol, data, z_score, mean_reversion_signal)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_z_score(self, data: pd.DataFrame) -> float:
        """Calculate z-score for mean reversion"""
        if len(data) < self.lookback_period:
            return 0.0
        
        prices = data['close'].tail(self.lookback_period)
        mean_price = prices.mean()
        std_price = prices.std()
        
        current_price = prices.iloc[-1]
        return (current_price - mean_price) / std_price if std_price > 0 else 0
    
    def _calculate_mean_reversion_signal(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion signal strength"""
        if len(data) < self.lookback_period:
            return 0.0
        
        # Calculate distance from moving average
        ma = data['close'].rolling(window=self.lookback_period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        return (ma - current_price) / ma if ma > 0 else 0
    
    def _generate_reversion_signal(self, symbol: str, data: pd.DataFrame, 
                                  z_score: float, reversion_signal: float) -> Optional[StrategySignal]:
        """Generate mean reversion signal"""
        current_price = data['close'].iloc[-1]
        
        # Determine action and confidence
        if z_score > self.z_score_threshold:
            action = 'sell'  # Price too high, expect reversion down
            confidence = min(0.9, 0.5 + abs(z_score) * 0.1)
        elif z_score < -self.z_score_threshold:
            action = 'buy'   # Price too low, expect reversion up
            confidence = min(0.9, 0.5 + abs(z_score) * 0.1)
        else:
            action = 'hold'
            confidence = 0.5
        
        if confidence < self.min_confidence:
            return None
        
        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            quantity=0.0,
            price=current_price,
            strategy_name=self.name,
            timestamp=data.index[-1] if hasattr(data, 'index') else None,
            metadata={
                'z_score': z_score,
                'reversion_signal': reversion_signal,
                'lookback_period': self.lookback_period
            }
        )
    
    def update_parameters(self, market_data: Dict):
        """Update mean reversion parameters"""
        # Adjust std dev threshold based on market conditions
        volatility = self._calculate_market_volatility(market_data)
        if volatility > 0.3:
            self.z_score_threshold *= 1.1
        elif volatility < 0.1:
            self.z_score_threshold *= 0.9
    
    def _calculate_market_volatility(self, market_data: Dict) -> float:
        """Calculate market volatility"""
        volatilities = []
        for symbol, data in market_data.items():
            if isinstance(data, pd.DataFrame) and len(data) > 20:
                returns = data['close'].pct_change().dropna()
                volatilities.append(returns.std())
        
        return np.mean(volatilities) if volatilities else 0.2


class MultiStrategyFramework:
    """
    Multi-Strategy Framework
    
    Features:
    - Multiple strategy execution
    - Signal aggregation
    - Risk management integration
    - Performance tracking
    - Dynamic allocation
    """
    
    def __init__(self, strategies: List[BaseStrategy], allocation_method: str = 'equal_weight', symbols: List[str] = ["AAPL"]):
        self.strategies = strategies
        self.allocation_method = allocation_method
        self.strategy_weights = {}
        self.performance_history = {}
        self.aggregated_signals = []
        self.symbols = symbols
        
        # Initialize equal weights
        if allocation_method == 'equal':
            weight = 1.0 / len(strategies)
            self.strategy_weights = {strategy.name: weight for strategy in strategies}
        
        logger.info(f"MultiStrategyFramework initialized with {len(strategies)} strategies")
    
    def generate_aggregated_signals(self, market_data: Dict) -> List[StrategySignal]:
        """Generate aggregated signals from all strategies"""
        all_signals = []
        
        for strategy in self.strategies:
            if not strategy.is_active:
                continue
            
            try:
                # Update strategy parameters
                strategy.update_parameters(market_data)
                
                # Generate signals
                signals = strategy.generate_signals(market_data)
                all_signals.extend(signals)
                
                # Record signals
                strategy.signal_history.extend(signals)
                
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {e}")
        
        # Aggregate signals
        aggregated = self._aggregate_signals(all_signals)
        self.aggregated_signals.extend(aggregated)
        
        return aggregated
    
    def _aggregate_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Aggregate signals from multiple strategies"""
        if not signals:
            return []
        
        # Group signals by symbol
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Aggregate signals for each symbol
        aggregated = []
        for symbol, symbol_signal_list in symbol_signals.items():
            aggregated_signal = self._aggregate_symbol_signals(symbol, symbol_signal_list)
            if aggregated_signal:
                aggregated.append(aggregated_signal)
        
        return aggregated
    
    def _aggregate_symbol_signals(self, symbol: str, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Aggregate signals for a single symbol"""
        if not signals:
            return None
        
        # Calculate weighted average
        total_weight = 0
        weighted_action = 0
        weighted_confidence = 0
        weighted_price = 0
        
        for signal in signals:
            weight = self.strategy_weights.get(signal.strategy_name, 1.0)
            total_weight += weight
            
            # Convert action to numeric
            action_value = 1 if signal.action == 'buy' else (-1 if signal.action == 'sell' else 0)
            
            weighted_action += action_value * weight * signal.confidence
            weighted_confidence += signal.confidence * weight
            weighted_price += signal.price * weight
        
        if total_weight == 0:
            return None
        
        # Determine final action
        if weighted_action > 0.1:
            action = 'buy'
        elif weighted_action < -0.1:
            action = 'sell'
        else:
            action = 'hold'
        
        # Calculate final metrics
        final_confidence = weighted_confidence / total_weight
        final_price = weighted_price / total_weight
        
        # Only return signal if confidence is high enough
        if final_confidence < 0.5:
            return None
        
        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=final_confidence,
            quantity=0.0,
            price=final_price,
            strategy_name='MultiStrategy',
            timestamp=signals[0].timestamp,
            metadata={
                'num_strategies': len(signals),
                'weighted_action': weighted_action,
                'strategy_weights': self.strategy_weights
            }
        )
    
    def update_strategy_weights(self, performance_data: Dict[str, StrategyPerformance]):
        """Update strategy weights based on performance"""
        if self.allocation_method == 'performance':
            total_performance = sum(perf.sharpe_ratio for perf in performance_data.values())
            
            if total_performance > 0:
                for strategy_name, performance in performance_data.items():
                    weight = performance.sharpe_ratio / total_performance
                    self.strategy_weights[strategy_name] = weight
            else:
                # Equal weights if no positive performance
                weight = 1.0 / len(self.strategies)
                self.strategy_weights = {strategy.name: weight for strategy in self.strategies}
    
    def calculate_framework_performance(self) -> StrategyPerformance:
        """Calculate overall framework performance"""
        try:
            # Aggregate performance metrics
            total_return = 0.0
            total_sharpe = 0.0
            total_trades = 0
            total_volatility = 0.0
            
            for strategy in self.strategies:
                performance = strategy.calculate_performance({})
                weight = self.strategy_weights.get(strategy.name, 1.0)
                
                total_return += performance.total_return * weight
                total_sharpe += performance.sharpe_ratio * weight
                total_trades += performance.total_trades
                total_volatility += performance.volatility * weight
            
            return StrategyPerformance(
                strategy_name='MultiStrategy',
                total_return=total_return,
                sharpe_ratio=total_sharpe,
                max_drawdown=0.05,  # Placeholder
                win_rate=0.6,  # Placeholder
                profit_factor=1.2,  # Placeholder
                total_trades=total_trades,
                avg_trade_duration=5.0,  # Placeholder
                volatility=total_volatility
            )
            
        except Exception as e:
            logger.error(f"Error calculating framework performance: {e}")
            return StrategyPerformance(
                strategy_name='MultiStrategy',
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                volatility=0.0
            )
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all strategies"""
        status = {}
        
        for strategy in self.strategies:
            status[strategy.name] = {
                'active': strategy.is_active,
                'weight': self.strategy_weights.get(strategy.name, 0.0),
                'signals_generated': len(strategy.signal_history),
                'performance': strategy.calculate_performance({})
            }
        
        return status


def create_momentum_strategy(symbols: List[str], **kwargs) -> MomentumStrategy:
    """Convenience function to create momentum strategy"""
    return MomentumStrategy(symbols, **kwargs)


def create_mean_reversion_strategy(symbols: List[str], **kwargs) -> MeanReversionStrategy:
    """Convenience function to create mean reversion strategy"""
    return MeanReversionStrategy(symbols, **kwargs)


def create_multi_strategy_framework(strategies: List[BaseStrategy], 
                                  allocation_method: str = 'equal', symbols: List[str] = ["AAPL"]) -> MultiStrategyFramework:
    """Convenience function to create multi-strategy framework"""
    return MultiStrategyFramework(strategies, allocation_method, symbols) 