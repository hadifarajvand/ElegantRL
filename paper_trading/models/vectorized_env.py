"""
Vectorized Trading Environment for Parallel Training
"""

import numpy as np
import pandas as pd
import torch as th
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
import logging

from .trading_env import EnhancedStockTradingEnv

logger = logging.getLogger(__name__)

ARY = np.ndarray
TEN = th.Tensor


class EnhancedStockTradingVecEnv(gym.vector.VectorEnv):
    """
    Vectorized Stock Trading Environment for Parallel Training
    
    Features:
    - Parallel environment execution
    - Shared data across environments
    - Batch action processing
    - Efficient memory management
    - Synchronized environment states
    """
    
    def __init__(self, 
                 data_dict: Dict[str, pd.DataFrame],
                 num_envs: int = 4,
                 initial_capital: float = 1000000.0,
                 max_stock_quantity: int = 100,
                 transaction_cost_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 max_position_size: float = 0.2,
                 min_cash_reserve: float = 0.1,
                 max_leverage: float = 1.5,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 reward_scale: float = 2 ** -12,
                 gamma: float = 0.99,
                 if_random_reset: bool = True):
        
        self.num_envs = num_envs
        self.data_dict = data_dict
        self.symbols = list(data_dict.keys())
        self.num_stocks = len(self.symbols)
        
        # Create individual environments
        self.envs = []
        for i in range(num_envs):
            env = EnhancedStockTradingEnv(
                data=data_dict,
                initial_capital=initial_capital,
                max_stock_quantity=max_stock_quantity,
                transaction_cost_pct=transaction_cost_pct,
                slippage_pct=slippage_pct,
                max_position_size=max_position_size,
                min_cash_reserve=min_cash_reserve,
                max_leverage=max_leverage,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                reward_scale=reward_scale,
                gamma=gamma,
                if_random_reset=if_random_reset
            )
            self.envs.append(env)
        
        # Get observation and action spaces from first environment
        sample_env = self.envs[0]
        self.state_dim = sample_env.state_dim
        self.action_dim = sample_env.action_dim
        self.max_step = sample_env.max_step
        
        # Define vectorized spaces
        self.single_observation_space = sample_env.observation_space
        self.single_action_space = sample_env.action_space
        
        # Initialize vectorized environment
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        
        # Environment information
        self.env_name = 'EnhancedStockTradingVecEnv-v1'
        self.if_discrete = False
        self.target_return = +np.inf
        
        logger.info(f"Vectorized environment created with {num_envs} environments")
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ARY, Dict]:
        """Reset all environments"""
        states = []
        infos = []
        
        for i, env in enumerate(self.envs):
            if seed is not None:
                env_seed = seed + i
            else:
                env_seed = None
            
            state, info = env.reset(seed=env_seed)
            states.append(state)
            infos.append(info)
        
        # Stack states into batch
        states_array = np.stack(states, axis=0)
        
        # Combine infos
        combined_info = {
            'individual': infos,
            'num_envs': self.num_envs
        }
        
        return states_array, combined_info
    
    def step(self, actions: ARY) -> Tuple[ARY, ARY, ARY, ARY, Dict]:
        """Execute actions in all environments"""
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        
        states = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, truncated, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
        
        # Stack results into batches
        states_array = np.stack(states, axis=0)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array(dones, dtype=bool)
        truncateds_array = np.array(truncateds, dtype=bool)
        
        # Combine infos
        combined_info = {
            'individual': infos,
            'num_envs': self.num_envs
        }
        
        return states_array, rewards_array, dones_array, truncateds_array, combined_info
    
    def get_state(self) -> ARY:
        """Get current states from all environments"""
        states = [env.get_state() for env in self.envs]
        return np.stack(states, axis=0)
    
    def render(self):
        """Render all environments"""
        for i, env in enumerate(self.envs):
            print(f"Environment {i}:")
            env.render()
    
    def get_portfolio_stats(self) -> List[Dict[str, float]]:
        """Get portfolio statistics from all environments"""
        stats = []
        for env in self.envs:
            stats.append(env.get_portfolio_stats())
        return stats
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
    
    def get_env_info(self) -> Dict[str, Any]:
        """Get information about the vectorized environment"""
        return {
            'num_envs': self.num_envs,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'max_step': self.max_step,
            'symbols': self.symbols,
            'num_stocks': self.num_stocks
        }


class AsyncStockTradingVecEnv(gym.vector.AsyncVectorEnv):
    """
    Asynchronous Vectorized Trading Environment
    
    Features:
    - Asynchronous environment execution
    - Non-blocking operations
    - Improved performance for I/O bound operations
    """
    
    def __init__(self, 
                 env_fns: List[callable],
                 num_envs: int = 4,
                 **kwargs):
        
        super().__init__(env_fns, **kwargs)
        self.num_envs = num_envs
        
        logger.info(f"Asynchronous vectorized environment created with {num_envs} environments")


def create_vectorized_env(data_dict: Dict[str, pd.DataFrame],
                         num_envs: int = 4,
                         **kwargs) -> EnhancedStockTradingVecEnv:
    """Convenience function to create vectorized environment"""
    return EnhancedStockTradingVecEnv(data_dict, num_envs, **kwargs)


def create_async_vectorized_env(data_dict: Dict[str, pd.DataFrame],
                               num_envs: int = 4,
                               **kwargs) -> AsyncStockTradingVecEnv:
    """Convenience function to create asynchronous vectorized environment"""
    
    def make_env(env_id):
        def _make_env():
            return EnhancedStockTradingEnv(
                data=data_dict,
                **kwargs
            )
        return _make_env
    
    env_fns = [make_env(i) for i in range(num_envs)]
    return AsyncStockTradingVecEnv(env_fns, num_envs) 