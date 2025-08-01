"""
Advanced Trading Agents for Paper Trading System
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from elegantrl.agents import AgentPPO, AgentA2C, AgentDQN, AgentSAC, AgentDDPG, AgentTD3

logger = logging.getLogger(__name__)


class TradingAgentPPO(AgentPPO):
    """
    PPO Agent Optimized for Trading
    
    Features:
    - Trading-specific network architecture
    - Custom loss functions for financial objectives
    - Risk-adjusted reward shaping
    - Position-aware action processing
    """
    
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, 
                 device: str = "cpu", **kwargs):
        super().__init__(net_dims, state_dim, action_dim, device, **kwargs)
        
        # Trading-specific parameters
        self.position_penalty = 0.01  # Penalty for large positions
        self.turnover_penalty = 0.001  # Penalty for high turnover
        self.risk_aversion = 0.1  # Risk aversion parameter
        
        logger.info(f"TradingAgentPPO initialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    def get_trading_action(self, state: np.ndarray, 
                          current_positions: Dict[str, float] = None) -> np.ndarray:
        """Get action with trading-specific processing"""
        # Convert state to tensor
        state_tensor = th.tensor(state, dtype=th.float32, device=self.device).unsqueeze(0)
        
        # Get action from network
        with th.no_grad():
            action = self.act(state_tensor)
            action = action.cpu().numpy().flatten()
        
        # Apply trading-specific constraints
        if current_positions is not None:
            action = self._apply_position_constraints(action, current_positions)
        
        # Apply risk management
        action = self._apply_risk_management(action, state)
        
        return action
    
    def _apply_position_constraints(self, action: np.ndarray, 
                                  positions: Dict[str, float]) -> np.ndarray:
        """Apply position-based constraints to actions"""
        # Limit position size based on current holdings
        for i, (symbol, position) in enumerate(positions.items()):
            if i < len(action):
                # Reduce action if position is already large
                if abs(position) > 0.5:  # 50% position threshold
                    action[i] *= 0.5
                elif abs(position) > 0.3:  # 30% position threshold
                    action[i] *= 0.7
        
        return action
    
    def _apply_risk_management(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply risk management to actions"""
        # Extract cash and portfolio information from state
        cash_norm = state[0] if len(state) > 0 else 0
        
        # Reduce action magnitude if cash is low
        if cash_norm < -0.5:  # Low cash threshold
            action *= 0.5
        
        # Apply volatility-based scaling
        # This would require volatility information in state
        return action
    
    def compute_trading_loss(self, states: th.Tensor, actions: th.Tensor, 
                           rewards: th.Tensor, next_states: th.Tensor,
                           dones: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        """Compute loss with trading-specific components"""
        # Standard PPO loss
        loss_dict = super().compute_loss(states, actions, rewards, next_states, dones, **kwargs)
        
        # Add trading-specific loss components
        trading_loss = self._compute_trading_loss(states, actions, rewards)
        loss_dict['trading_loss'] = trading_loss
        
        # Combine losses
        total_loss = loss_dict['total_loss'] + self.risk_aversion * trading_loss
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _compute_trading_loss(self, states: th.Tensor, actions: th.Tensor, 
                            rewards: th.Tensor) -> th.Tensor:
        """Compute trading-specific loss components"""
        # Position penalty
        position_loss = self.position_penalty * th.mean(th.abs(actions))
        
        # Turnover penalty (if we have previous actions)
        turnover_loss = self.turnover_penalty * th.mean(th.abs(actions))
        
        # Risk-adjusted reward loss
        risk_loss = -th.mean(rewards * th.exp(-self.risk_aversion * th.abs(actions)))
        
        return position_loss + turnover_loss + risk_loss


class TradingAgentSAC(AgentSAC):
    """
    SAC Agent for Continuous Trading Actions
    
    Features:
    - Continuous action space optimization
    - Entropy regularization for exploration
    - Risk-aware policy learning
    - Temperature adaptation
    """
    
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int,
                 device: str = "cpu", **kwargs):
        super().__init__(net_dims, state_dim, action_dim, device, **kwargs)
        
        # Trading-specific parameters
        self.temperature = 0.2  # Initial temperature
        self.temperature_decay = 0.995  # Temperature decay rate
        self.min_temperature = 0.01  # Minimum temperature
        self.risk_penalty = 0.05  # Risk penalty coefficient
        
        logger.info(f"TradingAgentSAC initialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    def get_trading_action(self, state: np.ndarray, 
                          explore: bool = True) -> np.ndarray:
        """Get action with exploration and risk management"""
        # Convert state to tensor
        state_tensor = th.tensor(state, dtype=th.float32, device=self.device).unsqueeze(0)
        
        # Get action from network
        with th.no_grad():
            action = self.act(state_tensor)
            action = action.cpu().numpy().flatten()
        
        if explore:
            # Add exploration noise
            noise = np.random.normal(0, 0.1, action.shape)
            action += noise
        
        # Apply risk constraints
        action = self._apply_risk_constraints(action, state)
        
        return action
    
    def _apply_risk_constraints(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply risk constraints to continuous actions"""
        # Clip actions to reasonable bounds
        action = np.clip(action, -1.0, 1.0)
        
        # Scale based on portfolio state
        cash_norm = state[0] if len(state) > 0 else 0
        
        # Reduce action magnitude if cash is low
        if cash_norm < -0.3:
            action *= 0.7
        
        return action
    
    def update_temperature(self):
        """Update temperature for exploration"""
        self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)
    
    def compute_trading_loss(self, states: th.Tensor, actions: th.Tensor,
                           rewards: th.Tensor, next_states: th.Tensor,
                           dones: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        """Compute loss with trading-specific components"""
        # Standard SAC loss
        loss_dict = super().compute_loss(states, actions, rewards, next_states, dones, **kwargs)
        
        # Add risk-aware loss
        risk_loss = self._compute_risk_loss(states, actions, rewards)
        loss_dict['risk_loss'] = risk_loss
        
        # Update temperature
        self.update_temperature()
        
        return loss_dict
    
    def _compute_risk_loss(self, states: th.Tensor, actions: th.Tensor,
                          rewards: th.Tensor) -> th.Tensor:
        """Compute risk-aware loss component"""
        # Penalize high-risk actions
        risk_penalty = self.risk_penalty * th.mean(th.abs(actions))
        
        # Entropy regularization for exploration
        entropy_loss = -self.temperature * th.mean(th.abs(actions))
        
        return risk_penalty + entropy_loss


class TradingAgentDQN(AgentDQN):
    """
    DQN Agent for Discrete Trading Actions
    
    Features:
    - Discrete action space for trading decisions
    - Experience replay with trading-specific sampling
    - Risk-aware Q-learning
    - Position-based action discretization
    """
    
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int,
                 device: str = "cpu", **kwargs):
        super().__init__(net_dims, state_dim, action_dim, device, **kwargs)
        
        # Trading-specific parameters
        self.action_discretization = 11  # Number of discrete actions per stock
        self.position_thresholds = [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]  # Position thresholds
        self.risk_discount = 0.95  # Risk discount factor
        
        logger.info(f"TradingAgentDQN initialized - State dim: {state_dim}, Action dim: {action_dim}")
    
    def discretize_action(self, continuous_action: np.ndarray) -> np.ndarray:
        """Convert continuous action to discrete action"""
        discrete_action = np.zeros_like(continuous_action, dtype=int)
        
        for i, action in enumerate(continuous_action):
            # Map continuous action to discrete bins
            if action < -0.5:
                discrete_action[i] = 0
            elif action < -0.3:
                discrete_action[i] = 1
            elif action < -0.1:
                discrete_action[i] = 2
            elif action < 0:
                discrete_action[i] = 3
            elif action < 0.1:
                discrete_action[i] = 4
            elif action < 0.3:
                discrete_action[i] = 5
            elif action < 0.5:
                discrete_action[i] = 6
            else:
                discrete_action[i] = 7
        
        return discrete_action
    
    def get_trading_action(self, state: np.ndarray, 
                          epsilon: float = 0.1) -> np.ndarray:
        """Get discrete trading action with epsilon-greedy exploration"""
        if np.random.random() < epsilon:
            # Random action
            action = np.random.randint(0, self.action_discretization, size=self.action_dim)
        else:
            # Q-network action
            q_values = self.get_q_values(state)
            action = np.argmax(q_values, axis=-1)
        
        return action
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions"""
        state_tensor = th.tensor(state, dtype=th.float32, device=self.device).unsqueeze(0)
        
        with th.no_grad():
            q_values = self.act(state_tensor)
        
        return q_values.cpu().numpy()
    
    def compute_trading_loss(self, states: th.Tensor, actions: th.Tensor,
                           rewards: th.Tensor, next_states: th.Tensor,
                           dones: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        """Compute loss with trading-specific components"""
        # Standard DQN loss
        loss_dict = super().compute_loss(states, actions, rewards, next_states, dones, **kwargs)
        
        # Add risk-aware loss
        risk_loss = self._compute_risk_loss(states, actions, rewards)
        loss_dict['risk_loss'] = risk_loss
        
        return loss_dict
    
    def _compute_risk_loss(self, states: th.Tensor, actions: th.Tensor,
                          rewards: th.Tensor) -> th.Tensor:
        """Compute risk-aware loss component"""
        # Penalize actions that lead to high risk
        risk_penalty = self.risk_discount * th.mean(th.abs(actions.float()))
        
        return risk_penalty


class EnsembleTradingAgent:
    """
    Ensemble of Multiple Trading Agents
    
    Features:
    - Multiple agent types
    - Weighted action combination
    - Dynamic weight adjustment
    - Performance-based ensemble
    """
    
    def __init__(self, agents: List, weights: Optional[List[float]] = None):
        self.agents = agents
        self.weights = weights or [1.0 / len(agents)] * len(agents)
        self.performance_history = [[] for _ in agents]
        
        logger.info(f"EnsembleTradingAgent initialized with {len(agents)} agents")
    
    def get_ensemble_action(self, state: np.ndarray, 
                          method: str = "weighted") -> np.ndarray:
        """Get ensemble action using specified method"""
        actions = []
        
        for agent in self.agents:
            if hasattr(agent, 'get_trading_action'):
                action = agent.get_trading_action(state)
            else:
                action = agent.get_action(state)
            actions.append(action)
        
        if method == "weighted":
            return self._weighted_combination(actions)
        elif method == "voting":
            return self._voting_combination(actions)
        elif method == "best":
            return self._best_agent_action(actions, state)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _weighted_combination(self, actions: List[np.ndarray]) -> np.ndarray:
        """Combine actions using weighted average"""
        weighted_action = np.zeros_like(actions[0])
        
        for action, weight in zip(actions, self.weights):
            weighted_action += weight * action
        
        return weighted_action
    
    def _voting_combination(self, actions: List[np.ndarray]) -> np.ndarray:
        """Combine actions using voting mechanism"""
        # For discrete actions, use majority voting
        # For continuous actions, use median
        actions_array = np.array(actions)
        return np.median(actions_array, axis=0)
    
    def _best_agent_action(self, actions: List[np.ndarray], 
                          state: np.ndarray) -> np.ndarray:
        """Select action from best performing agent"""
        # This would require performance tracking
        # For now, use the first agent
        return actions[0]
    
    def update_weights(self, performances: List[float]):
        """Update agent weights based on performance"""
        total_performance = sum(performances)
        
        if total_performance > 0:
            self.weights = [p / total_performance for p in performances]
        else:
            # Equal weights if all performances are zero
            self.weights = [1.0 / len(self.agents)] * len(self.agents)
        
        logger.info(f"Updated ensemble weights: {self.weights}")


def create_trading_agent(agent_type: str, net_dims: List[int], 
                        state_dim: int, action_dim: int, 
                        device: str = "cpu", **kwargs):
    """Factory function to create trading agents"""
    agent_map = {
        'PPO': TradingAgentPPO,
        'SAC': TradingAgentSAC,
        'DQN': TradingAgentDQN,
        'A2C': AgentA2C,
        'DDPG': AgentDDPG,
        'TD3': AgentTD3
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    agent_class = agent_map[agent_type]
    return agent_class(net_dims, state_dim, action_dim, device, **kwargs) 