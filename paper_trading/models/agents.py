"""
DRL Agents for Paper Trading System
"""

import numpy as np
import torch as th
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for DRL agents"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from current state"""
        raise NotImplementedError
    
    def update(self, batch: Dict) -> Dict:
        """Update agent with batch of experience"""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save agent to file"""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load agent from file"""
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Random agent for baseline comparison"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        super().__init__(state_dim, action_dim, device)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Return random action"""
        return np.random.randn(self.action_dim)
    
    def update(self, batch: Dict) -> Dict:
        """No update for random agent"""
        return {}
    
    def save(self, path: str):
        """Save agent (no parameters to save)"""
        pass
    
    def load(self, path: str):
        """Load agent (no parameters to load)"""
        pass


class SimplePPOAgent(BaseAgent):
    """Simple PPO agent implementation"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu", 
                 learning_rate: float = 3e-4, hidden_dim: int = 64):
        super().__init__(state_dim, action_dim, device)
        
        # Network architecture
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ).to(device)
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Optimizers
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from current state"""
        state_tensor = th.tensor(state, dtype=th.float32, device=self.device).unsqueeze(0)
        
        with th.no_grad():
            action = self.actor(state_tensor)
        
        return action.cpu().numpy().flatten()
    
    def update(self, batch: Dict) -> Dict:
        """Update agent with PPO"""
        states = th.tensor(batch['states'], dtype=th.float32, device=self.device)
        actions = th.tensor(batch['actions'], dtype=th.float32, device=self.device)
        rewards = th.tensor(batch['rewards'], dtype=th.float32, device=self.device)
        next_states = th.tensor(batch['next_states'], dtype=th.float32, device=self.device)
        dones = th.tensor(batch['dones'], dtype=th.float32, device=self.device)
        
        # Calculate advantages (simplified)
        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = rewards + 0.99 * next_values * (1 - dones) - values
        
        # Actor loss
        action_probs = self.actor(states)
        ratio = th.exp(th.log(action_probs + 1e-8) - th.log(actions + 1e-8))
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        actor_loss = -th.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values, rewards + 0.99 * next_values * (1 - dones))
        
        # Total loss
        total_loss = actor_loss + self.value_loss_coef * critic_loss
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, path: str):
        """Save agent to file"""
        th.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load(self, path: str):
        """Load agent from file"""
        checkpoint = th.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class SimpleDQNAgent(BaseAgent):
    """Simple DQN agent implementation"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu",
                 learning_rate: float = 1e-3, hidden_dim: int = 64):
        super().__init__(state_dim, action_dim, device)
        
        # Network architecture
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = th.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # DQN parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randn(self.action_dim)
        
        state_tensor = th.tensor(state, dtype=th.float32, device=self.device).unsqueeze(0)
        
        with th.no_grad():
            q_values = self.q_network(state_tensor)
            action = th.argmax(q_values, dim=1)
        
        # Convert to continuous action space
        action_continuous = np.zeros(self.action_dim)
        action_continuous[action.item()] = 1.0
        
        return action_continuous
    
    def update(self, batch: Dict) -> Dict:
        """Update agent with DQN"""
        states = th.tensor(batch['states'], dtype=th.float32, device=self.device)
        actions = th.tensor(batch['actions'], dtype=th.long, device=self.device)
        rewards = th.tensor(batch['rewards'], dtype=th.float32, device=self.device)
        next_states = th.tensor(batch['next_states'], dtype=th.float32, device=self.device)
        dones = th.tensor(batch['dones'], dtype=th.float32, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon
        }
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        """Save agent to file"""
        th.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load(self, path: str):
        """Load agent from file"""
        checkpoint = th.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def create_agent(agent_type: str, state_dim: int, action_dim: int, 
                device: str = "cpu", **kwargs) -> BaseAgent:
    """Factory function to create agents"""
    if agent_type.lower() == "random":
        return RandomAgent(state_dim, action_dim, device)
    elif agent_type.lower() == "ppo":
        return SimplePPOAgent(state_dim, action_dim, device, **kwargs)
    elif agent_type.lower() == "dqn":
        return SimpleDQNAgent(state_dim, action_dim, device, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 