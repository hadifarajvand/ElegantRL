"""
Main Paper Trading System
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trading.configs.trading_config import TradingConfig, DEFAULT_US_MARKET_CONFIG
from paper_trading.configs.model_config import ModelConfig, PPO_CONFIG
from paper_trading.data.market_data import YahooFinanceProvider, DataManager
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.paper_trading_engine.trading_engine import PaperTradingEngine
from elegantrl import Config, train_agent
from elegantrl.agents import AgentPPO, AgentA2C


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('paper_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def train_model(symbols: list, start_date: str, end_date: str, config_path: str = None):
    """Train DRL model for paper trading"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Load configurations
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        trading_config = TradingConfig(**config_data.get('trading', {}))
        model_config = ModelConfig(**config_data.get('model', {}))
    else:
        trading_config = DEFAULT_US_MARKET_CONFIG
        model_config = PPO_CONFIG
    
    # Initialize data provider
    data_provider = YahooFinanceProvider()
    data_manager = DataManager(data_provider)
    
    # Get historical data
    logger.info(f"Fetching data for symbols: {symbols}")
    data = data_manager.get_data(symbols, start_date, end_date, interval="1d")
    
    if not data:
        logger.error("No data retrieved. Exiting.")
        return None
    
    # Create environment
    env = EnhancedStockTradingEnv(
        data=data,
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
    
    # Set up ElegantRL configuration
    elegantrl_config = Config(
        agent_class=AgentPPO if model_config.agent_type == "PPO" else AgentA2C,
        env_class=EnhancedStockTradingEnv,
        env_args={
            'data': data,
            'initial_capital': trading_config.initial_capital,
            'max_stock_quantity': trading_config.max_stock_quantity,
            'transaction_cost_pct': trading_config.transaction_cost_pct,
            'slippage_pct': trading_config.slippage_pct,
            'max_position_size': trading_config.max_position_size,
            'min_cash_reserve': trading_config.min_cash_reserve,
            'max_leverage': trading_config.max_leverage,
            'stop_loss_pct': trading_config.stop_loss_pct,
            'take_profit_pct': trading_config.take_profit_pct
        },
        net_dims=model_config.net_dims,
        learning_rate=model_config.learning_rate,
        batch_size=model_config.batch_size,
        gamma=model_config.gamma,
        target_step=model_config.total_timesteps,
        eval_gap=model_config.eval_freq,
        eval_times=8,
        cwd=f"./trained_models/{model_config.agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Train the model
    logger.info("Starting training...")
    train_agent(elegantrl_config)
    
    logger.info("Training completed!")
    return elegantrl_config.cwd


def run_paper_trading(symbols: list, model_path: str, config_path: str = None):
    """Run paper trading with trained model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting paper trading...")
    
    # Load configurations
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        trading_config = TradingConfig(**config_data.get('trading', {}))
        model_config = ModelConfig(**config_data.get('model', {}))
    else:
        trading_config = DEFAULT_US_MARKET_CONFIG
        model_config = PPO_CONFIG
    
    # Initialize trading engine
    engine = PaperTradingEngine(
        trading_config=trading_config,
        model_config=model_config,
        model_path=model_path,
        symbols=symbols,
        data_provider="yahoo"
    )
    
    try:
        # Start paper trading
        engine.start()
        
        # Keep running until interrupted
        logger.info("Paper trading is running. Press Ctrl+C to stop.")
        while True:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping paper trading...")
        engine.stop()
        
        # Print performance summary
        summary = engine.get_performance_summary()
        logger.info("Performance Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")


def backtest_model(symbols: list, start_date: str, end_date: str, model_path: str, config_path: str = None):
    """Backtest trained model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting backtest...")
    
    # Load configurations
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        trading_config = TradingConfig(**config_data.get('trading', {}))
        model_config = ModelConfig(**config_data.get('model', {}))
    else:
        trading_config = DEFAULT_US_MARKET_CONFIG
        model_config = PPO_CONFIG
    
    # Initialize data provider
    data_provider = YahooFinanceProvider()
    data_manager = DataManager(data_provider)
    
    # Get historical data
    data = data_manager.get_data(symbols, start_date, end_date, interval="1d")
    
    if not data:
        logger.error("No data retrieved. Exiting.")
        return
    
    # Create environment
    env = EnhancedStockTradingEnv(
        data=data,
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
    
    # Load trained model (placeholder - implement based on your model format)
    # model = load_trained_model(model_path)
    
    # Run backtest
    state, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        # Get action from model (placeholder)
        action = env.action_space.sample()  # Replace with model prediction
        
        # Take step
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            logger.info(f"Step {step_count}: Total Reward = {total_reward:.2f}, Portfolio Value = ${info['total_asset']:,.2f}")
        
        if done or truncated:
            break
    
    # Print final results
    stats = env.get_portfolio_stats()
    logger.info("Backtest Results:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def create_config_file(config_path: str):
    """Create a sample configuration file"""
    config = {
        'trading': {
            'initial_capital': 1000000.0,
            'max_stock_quantity': 100,
            'transaction_cost_pct': 0.001,
            'slippage_pct': 0.0005,
            'max_position_size': 0.2,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15,
            'min_cash_reserve': 0.1,
            'max_leverage': 1.5,
            'trading_hours': ["09:30", "16:00"],
            'rebalance_frequency': "daily",
            'data_source': "yahoo",
            'update_frequency': "1min"
        },
        'model': {
            'agent_type': "PPO",
            'net_dims': [128, 64],
            'learning_rate': 3e-4,
            'batch_size': 2048,
            'gamma': 0.99,
            'total_timesteps': 1000000,
            'eval_freq': 10000,
            'save_freq': 50000,
            'log_freq': 1000,
            'device': "auto",
            'seed': 42
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration file created: {config_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Paper Trading System")
    parser.add_argument('--mode', choices=['train', 'trade', 'backtest', 'create_config'], 
                       required=True, help='Mode to run')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT'], 
                       help='Trading symbols')
    parser.add_argument('--start_date', default='2023-01-01', help='Start date for data')
    parser.add_argument('--end_date', default='2023-12-31', help='End date for data')
    parser.add_argument('--model_path', help='Path to trained model')
    parser.add_argument('--config_path', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if args.mode == 'create_config':
        config_path = args.config_path or 'paper_trading_config.yaml'
        create_config_file(config_path)
        return
    
    if args.mode == 'train':
        model_path = train_model(args.symbols, args.start_date, args.end_date, args.config_path)
        if model_path:
            logger.info(f"Model saved to: {model_path}")
    
    elif args.mode == 'trade':
        if not args.model_path:
            logger.error("Model path is required for trading mode")
            return
        run_paper_trading(args.symbols, args.model_path, args.config_path)
    
    elif args.mode == 'backtest':
        if not args.model_path:
            logger.error("Model path is required for backtest mode")
            return
        backtest_model(args.symbols, args.start_date, args.end_date, args.model_path, args.config_path)


if __name__ == "__main__":
    main() 