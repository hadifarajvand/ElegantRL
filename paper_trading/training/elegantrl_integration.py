"""
ElegantRL Integration for Paper Trading System
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

# Add parent directory to path for ElegantRL imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elegantrl import Config, train_agent
from elegantrl import get_gym_env_args
from elegantrl.agents import AgentPPO, AgentA2C, AgentDQN, AgentSAC, AgentDDPG, AgentTD3

from ..configs.trading_config import TradingConfig
from ..configs.model_config import ModelConfig
from ..models.trading_env import EnhancedStockTradingEnv
from ..data.market_data import YahooFinanceProvider, DataManager

logger = logging.getLogger(__name__)


class ElegantRLTrainer:
    """
    ElegantRL Integration for Paper Trading Training
    
    Features:
    - Direct integration with ElegantRL training framework
    - Support for all ElegantRL agents (PPO, A2C, DQN, SAC, DDPG, TD3)
    - Automatic environment configuration
    - Training progress monitoring
    - Model checkpointing and saving
    """
    
    def __init__(self, trading_config: TradingConfig, model_config: ModelConfig):
        self.trading_config = trading_config
        self.model_config = model_config
        self.agent_map = {
            'PPO': AgentPPO,
            'A2C': AgentA2C,
            'DQN': AgentDQN,
            'SAC': AgentSAC,
            'DDPG': AgentDDPG,
            'TD3': AgentTD3
        }
    
    def prepare_environment(self, data: Dict[str, pd.DataFrame]) -> EnhancedStockTradingEnv:
        """Prepare trading environment for ElegantRL"""
        logger.info("Preparing trading environment for ElegantRL training...")
        
        env = EnhancedStockTradingEnv(
            data=data,
            initial_capital=self.trading_config.initial_capital,
            max_stock_quantity=self.trading_config.max_stock_quantity,
            transaction_cost_pct=self.trading_config.transaction_cost_pct,
            slippage_pct=self.trading_config.slippage_pct,
            max_position_size=self.trading_config.max_position_size,
            min_cash_reserve=self.trading_config.min_cash_reserve,
            max_leverage=self.trading_config.max_leverage,
            stop_loss_pct=self.trading_config.stop_loss_pct,
            take_profit_pct=self.trading_config.take_profit_pct
        )
        
        logger.info(f"Environment prepared - State dim: {env.state_dim}, Action dim: {env.action_dim}")
        return env
    
    def create_elegantrl_config(self, env: EnhancedStockTradingEnv, 
                               agent_type: str = None) -> Config:
        """Create ElegantRL configuration for training"""
        agent_type = agent_type or self.model_config.agent_type
        
        if agent_type not in self.agent_map:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        agent_class = self.agent_map[agent_type]
        
        # Create environment arguments manually
        env_args = {
            'env_name': 'EnhancedStockTradingEnv-v1',
            'state_dim': env.state_dim,
            'action_dim': env.action_dim,
            'if_discrete': False,
            'target_return': +np.inf,
            'max_step': env.max_step,
            'env_num': 1
        }
        
        # Create ElegantRL configuration
        config = Config(
            agent_class=agent_class,
            env_class=EnhancedStockTradingEnv,
            env_args=env_args
        )
        
        logger.info(f"ElegantRL config created for {agent_type} agent")
        logger.info(f"Training directory: {config.cwd}")
        
        return config
    
    def train_model(self, data: Dict[str, pd.DataFrame], 
                   agent_type: str = None,
                   gpu_id: int = 0) -> str:
        """Train model using ElegantRL framework"""
        logger.info(f"Starting ElegantRL training for {agent_type or self.model_config.agent_type}")
        
        # Prepare environment
        env = self.prepare_environment(data)
        
        # Create configuration
        config = self.create_elegantrl_config(env, agent_type)
        
        # Train agent
        try:
            train_agent(config, gpu_id=gpu_id)
            logger.info(f"Training completed successfully. Model saved to: {config.cwd}")
            return config.cwd
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, model_path: str, 
                      data: Dict[str, pd.DataFrame],
                      eval_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model"""
        logger.info(f"Evaluating model from: {model_path}")
        
        # Prepare environment
        env = self.prepare_environment(data)
        
        # Load trained agent
        agent_type = self.model_config.agent_type
        agent_class = self.agent_map[agent_type]
        
        # Create agent instance
        agent = agent_class(
            net_dims=self.model_config.net_dims,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=self.model_config.device
        )
        
        # Load model weights
        agent.load(model_path)
        
        # Evaluate
        total_rewards = []
        total_returns = []
        
        for episode in range(eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_return = 0
            
            for step in range(env.max_step):
                action = agent.get_action(state)
                state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if done:
                    episode_return = info.get('portfolio_return', 0)
                    break
            
            total_rewards.append(episode_reward)
            total_returns.append(episode_return)
        
        # Calculate evaluation metrics
        eval_results = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'min_return': np.min(total_returns),
            'max_return': np.max(total_returns)
        }
        
        logger.info(f"Evaluation completed - Mean return: {eval_results['mean_return']:.4f}")
        return eval_results
    
    def compare_agents(self, data: Dict[str, pd.DataFrame], 
                      agent_types: List[str] = None) -> Dict[str, Dict]:
        """Compare different agent types"""
        if agent_types is None:
            agent_types = ['PPO', 'A2C', 'DQN']
        
        logger.info(f"Comparing agents: {agent_types}")
        
        results = {}
        
        for agent_type in agent_types:
            try:
                logger.info(f"Training {agent_type} agent...")
                model_path = self.train_model(data, agent_type)
                
                logger.info(f"Evaluating {agent_type} agent...")
                eval_results = self.evaluate_model(model_path, data)
                
                results[agent_type] = {
                    'model_path': model_path,
                    'evaluation': eval_results
                }
                
                logger.info(f"{agent_type} completed - Mean return: {eval_results['mean_return']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train/evaluate {agent_type}: {e}")
                results[agent_type] = {'error': str(e)}
        
        return results


def train_with_elegantrl(trading_config: TradingConfig, 
                        model_config: ModelConfig,
                        data: Dict[str, pd.DataFrame],
                        agent_type: str = None,
                        gpu_id: int = 0) -> str:
    """Convenience function for ElegantRL training"""
    trainer = ElegantRLTrainer(trading_config, model_config)
    return trainer.train_model(data, agent_type, gpu_id)


def evaluate_with_elegantrl(trading_config: TradingConfig,
                           model_config: ModelConfig,
                           model_path: str,
                           data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Convenience function for ElegantRL evaluation"""
    trainer = ElegantRLTrainer(trading_config, model_config)
    return trainer.evaluate_model(model_path, data) 