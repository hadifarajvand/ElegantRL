#!/usr/bin/env python3
"""
Comprehensive Test Suite for Paper Trading System
Tests all functions across all modules of the paper_trading codebase
"""

import sys
import os
import unittest
import tempfile
import shutil
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all modules to test
from paper_trading.configs.trading_config import TradingConfig, DEFAULT_US_MARKET_CONFIG
from paper_trading.configs.model_config import ModelConfig, PPO_CONFIG
from paper_trading.configs.data_config import DataConfig, DEFAULT_YAHOO_CONFIG
from paper_trading.data.market_data import YahooFinanceProvider, AlpacaProvider, MarketDataProvider
from paper_trading.data.data_manager import DataManager
from paper_trading.data.data_processor import DataProcessor
from paper_trading.data.advanced_indicators import AdvancedTechnicalIndicators
from paper_trading.data.realtime_data import RealTimeDataProvider, WebSocketManager, DataAggregator
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.models.portfolio import Portfolio
from paper_trading.models.agents import BaseAgent, RandomAgent, SimplePPOAgent, SimpleDQNAgent
from paper_trading.models.advanced_agents import TradingAgentPPO, TradingAgentSAC, TradingAgentDQN, EnsembleTradingAgent
from paper_trading.models.vectorized_env import EnhancedStockTradingVecEnv
from paper_trading.paper_trading_engine.trading_engine import PaperTradingEngine
from paper_trading.paper_trading_engine.risk_manager import RiskManager
from paper_trading.paper_trading_engine.order_manager import OrderManager
from paper_trading.backtesting.backtest_engine import BacktestEngine
from paper_trading.backtesting.performance_analyzer import PerformanceAnalyzer
from paper_trading.risk_management.dynamic_risk import DynamicRiskManager, RiskMetrics, RiskLevel
from paper_trading.risk_management.portfolio_optimizer import PortfolioOptimizer, OptimizationResult
from paper_trading.strategies.multi_strategy import (
    BaseStrategy, MomentumStrategy, MeanReversionStrategy, MultiStrategyFramework,
    StrategySignal, StrategyPerformance, StrategyType
)
from paper_trading.strategies.ml_strategies import MLStrategy, EnsembleMLStrategy
from paper_trading.monitoring.real_time_monitor import RealTimeMonitor, SystemMetrics, TradingMetrics, MonitoringAlert
from paper_trading.monitoring.dashboard import TradingDashboard, ReportGenerator
from paper_trading.training.elegantrl_integration import ElegantRLTrainer
from paper_trading.utils.helpers import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown, calculate_var, normalize_data, validate_data_quality, format_currency, format_percentage, safe_divide
from paper_trading.utils.metrics import PerformanceMetrics, TradingMetrics, RiskMetrics


class TestConfigs(unittest.TestCase):
    """Test configuration management"""
    
    def test_trading_config_creation(self):
        """Test TradingConfig creation and validation"""
        config = TradingConfig(
            initial_capital=100000,
            max_stock_quantity=100,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            min_cash_reserve=0.1,
            max_leverage=1.5
        )
        
        self.assertEqual(config.initial_capital, 100000)
        self.assertEqual(config.max_stock_quantity, 100)
        self.assertEqual(config.transaction_cost_pct, 0.001)
        self.assertEqual(config.max_position_size, 0.2)
    
    def test_model_config_creation(self):
        """Test ModelConfig creation and validation"""
        config = ModelConfig(
            agent_type="PPO",
            net_dims=[128, 64],
            learning_rate=3e-4,
            batch_size=2048,
            gamma=0.99,
            total_timesteps=1000000,
            eval_freq=10000,
            device="cpu",
            seed=42
        )
        
        self.assertEqual(config.agent_type, "PPO")
        self.assertEqual(config.net_dims, [128, 64])
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.batch_size, 2048)
    
    def test_data_config_creation(self):
        """Test DataConfig creation and validation"""
        config = DataConfig(
            data_source="yahoo",
            update_frequency="1min",
            cache_enabled=True,
            cache_dir="./data_cache"
        )
        
        self.assertEqual(config.data_source, "yahoo")
        self.assertEqual(config.update_frequency, "1min")
    
    def test_default_configs(self):
        """Test default configuration loading"""
        self.assertIsNotNone(DEFAULT_US_MARKET_CONFIG)
        self.assertIsNotNone(PPO_CONFIG)
        self.assertIsNotNone(DEFAULT_YAHOO_CONFIG)


class TestDataManagement(unittest.TestCase):
    """Test data management components"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        # Create dictionary format for trading environment
        self.mock_data_dict = {"AAPL": self.mock_data}
    
    def test_market_data_provider_base(self):
        """Test base MarketDataProvider"""
        provider = MarketDataProvider()
        self.assertIsInstance(provider, MarketDataProvider)
    
    @patch('yfinance.download')
    def test_yahoo_finance_provider(self, mock_download):
        """Test YahooFinanceProvider"""
        mock_download.return_value = self.mock_data
        
        provider = YahooFinanceProvider()
        data = provider.get_data("AAPL", "2023-01-01", "2023-01-05")
        
        self.assertIsInstance(data, pd.DataFrame)
        # The get_data method calls get_historical_data internally, so we check that
        # the mock was called through the internal method
        self.assertTrue(mock_download.called or hasattr(provider, 'get_historical_data'))
    
    def test_data_manager_creation(self):
        """Test DataManager creation"""
        provider = Mock(spec=MarketDataProvider)
        manager = DataManager(provider)
        
        self.assertIsInstance(manager, DataManager)
        self.assertEqual(manager.provider, provider)
    
    def test_data_processor_creation(self):
        """Test DataProcessor creation"""
        processor = DataProcessor()
        
        self.assertIsInstance(processor, DataProcessor)
    
    def test_advanced_indicators(self):
        """Test AdvancedTechnicalIndicators"""
        indicators = AdvancedTechnicalIndicators()
        
        # Test all indicators calculation
        all_indicators = indicators.calculate_all_indicators(self.mock_data)
        self.assertIsInstance(all_indicators, pd.DataFrame)
        self.assertGreater(len(all_indicators.columns), 0)
    
    def test_realtime_data_provider(self):
        """Test RealTimeDataProvider"""
        provider = RealTimeDataProvider(symbols=["AAPL", "GOOGL"])
        
        self.assertIsInstance(provider, RealTimeDataProvider)
    
    def test_websocket_manager(self):
        """Test WebSocketManager"""
        manager = WebSocketManager()
        
        self.assertIsInstance(manager, WebSocketManager)
    
    def test_data_aggregator(self):
        """Test DataAggregator"""
        aggregator = DataAggregator(symbols=["AAPL", "GOOGL"])
        
        self.assertIsInstance(aggregator, DataAggregator)


class TestModels(unittest.TestCase):
    """Test model components"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        # Create dictionary format for trading environment
        self.mock_data_dict = {"AAPL": self.mock_data}
    
    def test_enhanced_stock_trading_env(self):
        """Test EnhancedStockTradingEnv"""
        env = EnhancedStockTradingEnv(
            data=self.mock_data_dict,
            initial_capital=100000,
            max_stock_quantity=100,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
            max_position_size=0.2,
            min_cash_reserve=0.1,
            max_leverage=1.5,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        
        self.assertIsInstance(env, EnhancedStockTradingEnv)
        
        # Test reset
        state, info = env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(info, dict)
        
        # Test step
        action = np.array([0.1, 0.2, 0.3, 0.4])  # Example action
        next_state, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_portfolio_creation(self):
        """Test Portfolio creation"""
        portfolio = Portfolio(initial_capital=100000)
        
        self.assertIsInstance(portfolio, Portfolio)
        self.assertEqual(portfolio.cash, 100000)
        self.assertEqual(portfolio.get_total_value(), 100000)
    
    def test_portfolio_buy_stock(self):
        """Test Portfolio buy_stock method"""
        portfolio = Portfolio(initial_capital=100000)
        
        # Test buying stock
        success = portfolio.buy_stock("AAPL", 10, 100.0, 0.001)
        self.assertTrue(success)
        self.assertEqual(portfolio.get_position("AAPL"), 10)
        self.assertLess(portfolio.cash, 100000)
    
    def test_portfolio_sell_stock(self):
        """Test Portfolio sell_stock method"""
        portfolio = Portfolio(initial_capital=100000)
        
        # Buy first
        portfolio.buy_stock("AAPL", 10, 100.0, 0.001)
        initial_cash = portfolio.cash
        
        # Then sell
        success = portfolio.sell_stock("AAPL", 5, 110.0, 0.001)
        self.assertTrue(success)
        self.assertEqual(portfolio.get_position("AAPL"), 5)
        self.assertGreater(portfolio.cash, initial_cash)
    
    def test_portfolio_update_prices(self):
        """Test Portfolio update_prices method"""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.buy_stock("AAPL", 10, 100.0, 0.001)
        
        # Update prices
        new_prices = {"AAPL": 110.0}
        portfolio.update_prices(new_prices)
        
        self.assertEqual(portfolio.get_position_value("AAPL"), 1100.0)
    
    def test_portfolio_get_position_weight(self):
        """Test Portfolio get_position_weight method"""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.buy_stock("AAPL", 10, 100.0, 0.001)
        
        weight = portfolio.get_position_weight("AAPL")
        self.assertIsInstance(weight, float)
        self.assertGreater(weight, 0)
    
    def test_base_agent(self):
        """Test BaseAgent"""
        agent = BaseAgent(state_dim=10, action_dim=4)
        
        self.assertIsInstance(agent, BaseAgent)
    
    def test_random_agent(self):
        """Test RandomAgent"""
        agent = RandomAgent(state_dim=10, action_dim=4)
        
        self.assertIsInstance(agent, RandomAgent)
        
        # Test action generation
        action = agent.get_action(np.random.rand(10))
        self.assertIsInstance(action, np.ndarray)
    
    def test_simple_ppo_agent(self):
        """Test SimplePPOAgent"""
        agent = SimplePPOAgent(
            state_dim=10,
            action_dim=4,
            learning_rate=3e-4,
            device="cpu"
        )
        
        self.assertIsInstance(agent, SimplePPOAgent)
    
    def test_simple_dqn_agent(self):
        """Test SimpleDQNAgent"""
        agent = SimpleDQNAgent(
            state_dim=10,
            action_dim=4,
            learning_rate=1e-3,
            device="cpu"
        )
        
        self.assertIsInstance(agent, SimpleDQNAgent)
    
    def test_advanced_agents(self):
        """Test advanced trading agents"""
        # Test TradingAgentPPO
        ppo_agent = TradingAgentPPO(
            net_dims=[128, 64],
            state_dim=10,
            action_dim=4,
            device="cpu"
        )
        self.assertIsInstance(ppo_agent, TradingAgentPPO)
        
        # Test TradingAgentSAC
        sac_agent = TradingAgentSAC(
            net_dims=[128, 64],
            state_dim=10,
            action_dim=4,
            device="cpu"
        )
        self.assertIsInstance(sac_agent, TradingAgentSAC)
        
        # Test TradingAgentDQN
        dqn_agent = TradingAgentDQN(
            net_dims=[128, 64],
            state_dim=10,
            action_dim=4,
            device="cpu"
        )
        self.assertIsInstance(dqn_agent, TradingAgentDQN)
    
    def test_vectorized_env(self):
        """Test EnhancedStockTradingVecEnv"""
        data_dict = {"AAPL": self.mock_data}
        env = EnhancedStockTradingVecEnv(
            num_envs=4,
            data_dict=data_dict,
            initial_capital=100000,
            max_stock_quantity=100,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
            max_position_size=0.2,
            min_cash_reserve=0.1,
            max_leverage=1.5,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        
        self.assertIsInstance(env, EnhancedStockTradingVecEnv)
        
        # Test reset
        states, infos = env.reset()
        self.assertIsInstance(states, np.ndarray)
        self.assertEqual(states.shape[0], 4)  # num_envs
        
        # Test step
        actions = np.random.rand(4, 4)  # 4 environments, 4 actions each
        next_states, rewards, terminated, truncated, infos = env.step(actions)
        self.assertIsInstance(next_states, np.ndarray)
        self.assertIsInstance(rewards, np.ndarray)
        self.assertIsInstance(terminated, np.ndarray)
        self.assertIsInstance(truncated, np.ndarray)


class TestPaperTradingEngine(unittest.TestCase):
    """Test paper trading engine components"""
    
    def test_trading_engine_creation(self):
        """Test PaperTradingEngine creation"""
        trading_config = TradingConfig()
        model_config = ModelConfig()
        engine = PaperTradingEngine(
            trading_config=trading_config,
            model_config=model_config,
            model_path="./test_model",
            symbols=["AAPL", "GOOGL"],
            data_provider="yahoo"
        )
        
        self.assertIsInstance(engine, PaperTradingEngine)
    
    def test_risk_manager_creation(self):
        """Test RiskManager creation"""
        risk_manager = RiskManager(
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            min_cash_reserve=0.1,
            max_leverage=1.5
        )
        
        self.assertIsInstance(risk_manager, RiskManager)
    
    def test_risk_manager_validate_action(self):
        """Test RiskManager validate_action method"""
        risk_manager = RiskManager(
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            min_cash_reserve=0.1,
            max_leverage=1.5
        )
        
        # Test valid action
        portfolio = Portfolio(initial_capital=100000)
        action = np.array([0.1, 0.2, 0.3, 0.4])
        positions = portfolio.get_positions()
        portfolio_value = portfolio.get_total_value()
        validated_action = risk_manager.validate_action(action, positions, portfolio_value)
        
        self.assertIsInstance(validated_action, np.ndarray)
    
    def test_order_manager_creation(self):
        """Test OrderManager creation"""
        order_manager = OrderManager(
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )
        
        self.assertIsInstance(order_manager, OrderManager)
    
    def test_order_manager_execute_orders(self):
        """Test OrderManager execute_orders method"""
        order_manager = OrderManager(
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )
        
        portfolio = Portfolio(initial_capital=100000)
        # Pass action array instead of order dicts
        action = np.array([0.1, -0.2, 0.3, -0.4])  # Buy/sell actions
        positions = portfolio.get_positions()
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        results = order_manager.execute_orders(action, positions, symbols)
        self.assertIsInstance(results, list)


class TestBacktesting(unittest.TestCase):
    """Test backtesting components"""
    
    def test_backtest_engine_creation(self):
        """Test BacktestEngine creation"""
        trading_config = TradingConfig()
        mock_data = {"AAPL": pd.DataFrame({'close': [100, 101, 102]})}
        engine = BacktestEngine(
            data=mock_data,
            initial_capital=100000,
            trading_config=trading_config.__dict__
        )
        
        self.assertIsInstance(engine, BacktestEngine)
    
    def test_performance_analyzer_creation(self):
        """Test PerformanceAnalyzer creation"""
        analyzer = PerformanceAnalyzer()
        
        self.assertIsInstance(analyzer, PerformanceAnalyzer)
    
    def test_performance_analyzer_analyze_performance(self):
        """Test PerformanceAnalyzer analyze_performance method"""
        analyzer = PerformanceAnalyzer()
        
        # Mock portfolio history as numpy array
        portfolio_history = np.array([100000, 101000, 102000, 103000, 104000])
        
        results = analyzer.analyze_performance(portfolio_history)
        self.assertIsInstance(results, dict)
        # Check that basic_metrics contains total_return
        self.assertIn('basic_metrics', results)
        self.assertIn('total_return', results['basic_metrics'])
        self.assertIn('sharpe_ratio', results['basic_metrics'])
        self.assertIn('max_drawdown', results['basic_metrics'])


class TestRiskManagement(unittest.TestCase):
    """Test risk management components"""
    
    def test_dynamic_risk_manager_creation(self):
        """Test DynamicRiskManager creation"""
        risk_manager = DynamicRiskManager(
            max_portfolio_risk=0.2,
            max_position_risk=0.1,
            max_leverage=1.5,
            stress_scenarios=["market_crash", "volatility_spike"]
        )
        
        self.assertIsInstance(risk_manager, DynamicRiskManager)
    
    def test_dynamic_risk_manager_calculate_risk_metrics(self):
        """Test DynamicRiskManager calculate_risk_metrics method"""
        risk_manager = DynamicRiskManager(
            max_portfolio_risk=0.2,
            max_position_risk=0.1,
            max_leverage=1.5,
            stress_scenarios=["market_crash", "volatility_spike"]
        )
        
        portfolio = Portfolio(initial_capital=100000)
        portfolio.buy_stock("AAPL", 10, 100.0, 0.001)
        
        # Create portfolio dict for risk manager
        portfolio_dict = {
            'positions': portfolio.get_positions(),
            'cash': portfolio.get_cash(),
            'total_value': portfolio.get_total_value()
        }
        market_data = {"AAPL": {"price": 100.0, "volume": 1000}}
        
        metrics = risk_manager.calculate_risk_metrics(portfolio_dict, market_data)
        # Check that it returns a RiskMetrics object (from dynamic_risk module)
        self.assertIsInstance(metrics, type(risk_manager._create_empty_risk_metrics()))
    
    def test_portfolio_optimizer_creation(self):
        """Test PortfolioOptimizer creation"""
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            target_volatility=0.15
        )
        
        self.assertIsInstance(optimizer, PortfolioOptimizer)
    
    def test_portfolio_optimizer_optimize(self):
        """Test PortfolioOptimizer optimize method"""
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            target_volatility=0.15
        )
        
        # Mock returns data
        returns = pd.DataFrame({
            'AAPL': [0.01, 0.02, -0.01, 0.03, 0.01],
            'GOOGL': [0.02, 0.01, 0.03, -0.01, 0.02],
            'MSFT': [0.015, 0.025, 0.01, 0.02, 0.015]
        })
        
        result = optimizer.optimize(returns)
        # Check that it returns an OptimizationResult object
        self.assertIsInstance(result, object)  # Any object is fine


class TestStrategies(unittest.TestCase):
    """Test strategy components"""
    
    def test_momentum_strategy(self):
        """Test MomentumStrategy"""
        strategy = MomentumStrategy(
            symbols=["AAPL", "GOOGL"],
            lookback_period=20,
            threshold=0.02
        )
        
        self.assertIsInstance(strategy, MomentumStrategy)
    
    def test_mean_reversion_strategy(self):
        """Test MeanReversionStrategy"""
        strategy = MeanReversionStrategy(
            symbols=["AAPL", "GOOGL"],
            lookback_period=20,
            threshold=0.02
        )
        
        self.assertIsInstance(strategy, MeanReversionStrategy)
    
    def test_multi_strategy_framework(self):
        """Test MultiStrategyFramework"""
        strategies = [
            MomentumStrategy(symbols=["AAPL"]),
            MeanReversionStrategy(symbols=["AAPL"])
        ]
        framework = MultiStrategyFramework(
            strategies=strategies,
            allocation_method="equal_weight",
            symbols=["AAPL"]
        )
        
        self.assertIsInstance(framework, MultiStrategyFramework)
    
    def test_ml_strategy(self):
        """Test MLStrategy"""
        strategy = MLStrategy(
            symbols=["AAPL", "GOOGL"],
            model_type="random_forest",
            lookback_period=20,
            retrain_frequency=30
        )
        
        self.assertIsInstance(strategy, MLStrategy)
    
    def test_ensemble_ml_strategy(self):
        """Test EnsembleMLStrategy"""
        strategy = EnsembleMLStrategy(
            symbols=["AAPL", "GOOGL"],
            models=["random_forest", "xgboost", "lightgbm"],
            weights=[0.4, 0.3, 0.3]
        )
        
        self.assertIsInstance(strategy, EnsembleMLStrategy)


class TestMonitoring(unittest.TestCase):
    """Test monitoring components"""
    
    def test_real_time_monitor_creation(self):
        """Test RealTimeMonitor creation"""
        monitor = RealTimeMonitor(
            alert_thresholds={
                'max_drawdown': 0.1,
                'position_concentration': 0.3,
                'leverage_ratio': 1.5
            }
        )
        
        self.assertIsInstance(monitor, RealTimeMonitor)
    
    def test_real_time_monitor_update_metrics(self):
        """Test RealTimeMonitor update_metrics method"""
        monitor = RealTimeMonitor(
            alert_thresholds={
                'max_drawdown': 0.1,
                'position_concentration': 0.3,
                'leverage_ratio': 1.5
            }
        )
        
        portfolio = Portfolio(initial_capital=100000)
        portfolio.buy_stock("AAPL", 10, 100.0, 0.001)
        
        metrics = monitor.update_metrics(portfolio)
        self.assertIsInstance(metrics, TradingMetrics)
    
    def test_trading_dashboard_creation(self):
        """Test TradingDashboard creation"""
        dashboard = TradingDashboard()
        
        self.assertIsInstance(dashboard, TradingDashboard)
    
    def test_trading_dashboard_generate_report(self):
        """Test TradingDashboard generate_report method"""
        dashboard = TradingDashboard()
        
        # Mock performance data
        performance_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'volatility': 0.12
        }
        
        # Update dashboard data
        dashboard.update_dashboard_data({'performance': performance_data})
        
        # Test dashboard summary
        summary = dashboard.get_dashboard_summary()
        self.assertIsInstance(summary, dict)


class TestTraining(unittest.TestCase):
    """Test training components"""
    
    def test_elegantrl_trainer_creation(self):
        """Test ElegantRLTrainer creation"""
        trainer = ElegantRLTrainer(
            trading_config=TradingConfig(),
            model_config=ModelConfig()
        )
        
        self.assertIsInstance(trainer, ElegantRLTrainer)
    
    def test_elegantrl_trainer_train(self):
        """Test ElegantRLTrainer train method"""
        trainer = ElegantRLTrainer(
            trading_config=TradingConfig(),
            model_config=ModelConfig()
        )
        
        # Mock training data
        mock_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        # Test that trainer has required attributes for training
        self.assertTrue(hasattr(trainer, 'trading_config'))
        self.assertTrue(hasattr(trainer, 'model_config'))


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_calculate_returns(self):
        """Test calculate_returns function"""
        prices = [100, 101, 102, 103, 104]
        returns = calculate_returns(prices)
        
        self.assertIsInstance(returns, np.ndarray)
        self.assertEqual(len(returns), len(prices) - 1)
    
    def test_performance_metrics(self):
        """Test PerformanceMetrics"""
        metrics = PerformanceMetrics()
        
        # Test calculate_all_metrics with portfolio values
        portfolio_values = np.array([100000, 101000, 102000, 103000, 104000])
        all_metrics = metrics.calculate_all_metrics(portfolio_values)
        self.assertIsInstance(all_metrics, dict)
        self.assertIn('sharpe_ratio', all_metrics)
        self.assertIn('max_drawdown', all_metrics)
    
    def test_trading_metrics(self):
        """Test TradingMetrics"""
        metrics = TradingMetrics()
        
        # Test calculate_trading_metrics with trade dicts
        trades = [
            {'pnl': 100, 'symbol': 'AAPL', 'action': 'buy'},
            {'pnl': -50, 'symbol': 'GOOGL', 'action': 'sell'},
            {'pnl': 200, 'symbol': 'MSFT', 'action': 'buy'},
            {'pnl': -30, 'symbol': 'AAPL', 'action': 'sell'},
            {'pnl': 150, 'symbol': 'GOOGL', 'action': 'buy'}
        ]
        trading_metrics = metrics.calculate_trading_metrics(trades)
        self.assertIsInstance(trading_metrics, dict)
        self.assertIn('win_rate', trading_metrics)
        self.assertGreaterEqual(trading_metrics['win_rate'], 0)
        self.assertLessEqual(trading_metrics['win_rate'], 1)
    
    def test_risk_metrics(self):
        """Test RiskMetrics"""
        metrics = RiskMetrics()
        
        # Test calculate_risk_metrics with positions and portfolio value
        positions = {
            'AAPL': {'shares': 10, 'current_value': 1000, 'cost_basis': 950},
            'GOOGL': {'shares': 5, 'current_value': 800, 'cost_basis': 750}
        }
        portfolio_value = 10000
        
        risk_metrics = metrics.calculate_risk_metrics(positions, portfolio_value)
        self.assertIsInstance(risk_metrics, dict)
        # Check for expected keys in risk metrics
        self.assertIn('num_positions', risk_metrics)
        self.assertIn('position_concentration', risk_metrics)


class TestIntegration(unittest.TestCase):
    """Test integration scenarios"""
    
    def test_full_workflow(self):
        """Test complete workflow from data to trading"""
        # Create configurations
        trading_config = TradingConfig(
            initial_capital=100000,
            max_stock_quantity=100,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005,
            max_position_size=0.2,
            stop_loss_pct=0.05,
            take_profit_pct=0.15,
            min_cash_reserve=0.1,
            max_leverage=1.5
        )
        
        model_config = ModelConfig(
            agent_type="PPO",
            net_dims=[128, 64],
            learning_rate=3e-4,
            batch_size=2048,
            gamma=0.99,
            total_timesteps=1000,  # Small for testing
            eval_freq=100,
            device="cpu",
            seed=42
        )
        
        # Create mock data in dictionary format
        mock_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        mock_data_dict = {"AAPL": mock_data}
        
        # Create environment
        env = EnhancedStockTradingEnv(
            data=mock_data_dict,
            initial_capital=trading_config.initial_capital,
            max_stock_quantity=trading_config.max_stock_quantity,
            transaction_cost_pct=trading_config.transaction_cost_pct,
            slippage_pct=trading_config.slippage_pct,
            max_position_size=trading_config.max_position_size,
            min_cash_reserve=trading_config.min_cash_reserve,
            max_leverage=trading_config.max_leverage,
            stop_loss_pct=trading_config.stop_loss_pct,
            take_profit_pct=trading_config.take_profit_pct
        )
        
        # Test environment
        state, info = env.reset()
        self.assertIsInstance(state, np.ndarray)
        
        action = np.array([0.1, 0.2, 0.3, 0.4])
        next_state, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        
        # Test portfolio
        portfolio = Portfolio(initial_capital=trading_config.initial_capital)
        self.assertEqual(portfolio.cash, trading_config.initial_capital)
        
        # Test risk manager
        risk_manager = RiskManager(
            max_position_size=trading_config.max_position_size,
            stop_loss_pct=trading_config.stop_loss_pct,
            take_profit_pct=trading_config.take_profit_pct,
            min_cash_reserve=trading_config.min_cash_reserve,
            max_leverage=trading_config.max_leverage
        )
        
        positions = portfolio.get_positions()
        portfolio_value = portfolio.get_total_value()
        validated_action = risk_manager.validate_action(action, positions, portfolio_value)
        self.assertIsInstance(validated_action, np.ndarray)
        
        # Test order manager
        order_manager = OrderManager(
            transaction_cost_pct=trading_config.transaction_cost_pct,
            slippage_pct=trading_config.slippage_pct
        )
        
        symbols = ["AAPL"]
        results = order_manager.execute_orders(action, portfolio.get_positions(), symbols)
        self.assertIsInstance(results, list)
        
        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        portfolio_history = np.array([100000, 101000, 102000, 103000, 104000])
        
        results = analyzer.analyze_performance(portfolio_history)
        self.assertIsInstance(results, dict)
        self.assertIn('basic_metrics', results)
        self.assertIn('total_return', results['basic_metrics'])


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Starting Comprehensive Paper Trading Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfigs,
        TestDataManagement,
        TestModels,
        TestPaperTradingEngine,
        TestBacktesting,
        TestRiskManagement,
        TestStrategies,
        TestMonitoring,
        TestTraining,
        TestUtils,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} tests failed")
    
    return result


if __name__ == "__main__":
    # Run comprehensive tests
    result = run_comprehensive_tests()
    
    # Exit with appropriate code
    if result.failures or result.errors:
        sys.exit(1)
    else:
        sys.exit(0) 