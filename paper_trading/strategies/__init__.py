"""
Strategies package for paper trading system
"""

from .multi_strategy import (
    BaseStrategy, MomentumStrategy, MeanReversionStrategy, MultiStrategyFramework,
    StrategySignal, StrategyPerformance, StrategyType,
    create_momentum_strategy, create_mean_reversion_strategy, create_multi_strategy_framework
)
from .ml_strategies import (
    MLStrategy, EnsembleMLStrategy,
    create_ml_strategy, create_ensemble_ml_strategy
) 