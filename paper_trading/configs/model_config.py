"""
Model Configuration for DRL Agents
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for DRL model training and inference"""
    
    # Agent Type
    agent_type: str = "PPO"  # PPO, A2C, DQN, DDPG, SAC, TD3
    
    # Network Architecture
    net_dims: List[int] = None  # [128, 64] for MLP
    state_dim: int = 0  # Will be set automatically
    action_dim: int = 0  # Will be set automatically
    
    # Training Parameters
    learning_rate: float = 3e-4
    batch_size: int = 2048
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for PPO/A2C
    
    # PPO Specific
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # SAC Specific
    alpha: float = 0.2  # Temperature parameter
    tau: float = 0.005  # Target network update rate
    
    # DQN Specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training Schedule
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000
    log_freq: int = 1000
    
    # Environment
    num_envs: int = 1
    max_episode_steps: int = 1000
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    num_threads: int = 4
    
    # Random Seed
    seed: int = 42
    
    def __post_init__(self):
        if self.net_dims is None:
            self.net_dims = [128, 64]
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'ModelConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, file_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'agent_type': self.agent_type,
            'net_dims': self.net_dims,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_ratio': self.clip_ratio,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'alpha': self.alpha,
            'tau': self.tau,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'total_timesteps': self.total_timesteps,
            'eval_freq': self.eval_freq,
            'save_freq': self.save_freq,
            'log_freq': self.log_freq,
            'num_envs': self.num_envs,
            'max_episode_steps': self.max_episode_steps,
            'device': self.device,
            'num_threads': self.num_threads,
            'seed': self.seed
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Predefined configurations for different agent types
PPO_CONFIG = ModelConfig(
    agent_type="PPO",
    learning_rate=3e-4,
    batch_size=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5
)

A2C_CONFIG = ModelConfig(
    agent_type="A2C",
    learning_rate=3e-4,
    batch_size=2048,
    gamma=0.99,
    gae_lambda=0.95,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5
)

SAC_CONFIG = ModelConfig(
    agent_type="SAC",
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    alpha=0.2,
    tau=0.005
)

DQN_CONFIG = ModelConfig(
    agent_type="DQN",
    learning_rate=1e-3,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
) 