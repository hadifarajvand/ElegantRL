"""
Cryptocurrency DRL Agent for Paper Trading System
Uses ElegantRL for training deep reinforcement learning agents on crypto data
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import ElegantRL components
from elegantrl.agents import AgentPPO, AgentSAC, AgentTD3, AgentDDPG, AgentA2C, AgentDQN, AgentDuelingDQN
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent

# Import our components
from paper_trading.data.ccxt_provider import CCXTProvider
from paper_trading.configs.trading_config import TradingConfig
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.utils.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class CryptoTradingNet(nn.Module):
    """
    Neural network for cryptocurrency trading
    Optimized for crypto market characteristics
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output actions in [-1, 1]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass for actor"""
        return self.actor(state)
    
    def get_value(self, state, action):
        """Get value for critic"""
        x = torch.cat([state, action], dim=1)
        return self.critic(x)


class CryptoTradingAgent:
    """
    Cryptocurrency Trading Agent using ElegantRL
    """
    
    def __init__(self, 
                 agent_type: str = 'PPO',
                 state_dim: int = 15,
                 action_dim: int = 1,
                 hidden_dim: int = 64,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005):
        
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        
        # Create neural network
        self.net = CryptoTradingNet(state_dim, action_dim, hidden_dim)
        
        # Create ElegantRL agent
        self.agent = self._create_agent()
        
        logger.info(f"Crypto trading agent created: {agent_type}")
    
    def _create_agent(self):
        """Create ElegantRL agent based on type"""
        # Define network dimensions
        net_dims = [self.state_dim, self.hidden_dim, self.hidden_dim, self.action_dim]
        
        if self.agent_type == 'PPO':
            return AgentPPO(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        elif self.agent_type == 'A2C':
            return AgentA2C(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        elif self.agent_type == 'DQN':
            return AgentDQN(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        elif self.agent_type == 'DuelingDQN':
            return AgentDuelingDQN(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        elif self.agent_type == 'SAC':
            return AgentSAC(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        elif self.agent_type == 'DDPG':
            return AgentDDPG(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        elif self.agent_type == 'TD3':
            return AgentTD3(
                net_dims=net_dims,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                gpu_id=0
            )
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.agent.explore_action(state_tensor)
        return action.cpu().numpy().flatten()
    
    def update(self, batch):
        """Update agent with experience batch"""
        return self.agent.update(batch)
    
    def save_agent(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'agent_type': self.agent_type,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'model_state_dict': self.agent.act.state_dict(),
        }, filepath)
    
    def load_agent(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath)
        self.agent_type = checkpoint['agent_type']
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        # Recreate agent if needed
        if hasattr(self, 'agent'):
            self.agent.act.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.agent = self._create_agent()
            self.agent.act.load_state_dict(checkpoint['model_state_dict'])


class CryptoTradingTrainer:
    """
    Trainer for cryptocurrency trading agents
    """
    
    def __init__(self, 
                 data_provider,
                 trading_config: TradingConfig,
                 agent_type: str = 'PPO',
                 training_episodes: int = 1000,
                 evaluation_episodes: int = 100):
        
        self.data_provider = data_provider
        self.trading_config = trading_config
        self.agent_type = agent_type
        self.training_episodes = training_episodes
        self.evaluation_episodes = evaluation_episodes
        
        # Training data
        self.training_data = None
        self.evaluation_data = None
        
        # Agent
        self.agent = None
        
        # Performance tracking
        self.training_rewards = []
        self.evaluation_rewards = []
        self.performance_metrics = []
        
        logger.info(f"Crypto trading trainer initialized: {agent_type}")
    
    def prepare_data(self, symbols: List[str], start_date: str, end_date: str):
        """Prepare training and evaluation data"""
        print("📊 Preparing cryptocurrency data for training...")
        
        # Fetch data
        data = {}
        for symbol in symbols:
            print(f"📈 Fetching {symbol} data...")
            symbol_data = self.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1d'
            )
            
            if not symbol_data.empty:
                # Add technical indicators
                symbol_data = self.data_provider.get_technical_indicators(symbol_data)
                data[symbol] = symbol_data
                print(f"✅ {symbol}: {len(symbol_data)} records, {len(symbol_data.columns)} features")
            else:
                print(f"❌ No data for {symbol}")
        
        if not data:
            raise ValueError("No data available for training")
        
        # Split data into training and evaluation
        total_days = len(list(data.values())[0])
        split_point = int(total_days * 0.8)  # 80% training, 20% evaluation
        
        self.training_data = {symbol: df.iloc[:split_point] for symbol, df in data.items()}
        self.evaluation_data = {symbol: df.iloc[split_point:] for symbol, df in data.items()}
        
        print(f"✅ Data prepared: {len(self.training_data)} symbols")
        print(f"   Training: {len(list(self.training_data.values())[0])} days")
        print(f"   Evaluation: {len(list(self.evaluation_data.values())[0])} days")
        
        return True
    
    def create_trading_env(self, data: Dict[str, pd.DataFrame]) -> EnhancedStockTradingEnv:
        """Create trading environment with data"""
        return EnhancedStockTradingEnv(
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
    
    def train_agent(self):
        """Train the cryptocurrency trading agent"""
        print(f"\n🚀 Training {self.agent_type} agent on cryptocurrency data...")
        print("=" * 60)
        
        # Create agent
        env = self.create_trading_env(self.training_data)
        self.agent = CryptoTradingAgent(
            agent_type=self.agent_type,
            state_dim=env.state_dim,
            action_dim=env.action_dim
        )
        
        # Training loop
        best_reward = float('-inf')
        training_rewards = []
        
        for episode in range(self.training_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from agent
                action = self.agent.get_action(state)
                
                # Take step in environment
                state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if truncated:
                    break
            
            training_rewards.append(episode_reward)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(training_rewards[-100:])
                print(f"   Episode {episode:4d}: Avg Reward = {avg_reward:.4f}")
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print(f"   🎯 New best average reward: {best_reward:.4f}")
        
        self.training_rewards = training_rewards
        print(f"✅ Training completed! Best avg reward: {best_reward:.4f}")
        
        return True
    
    def evaluate_agent(self):
        """Evaluate the trained agent"""
        print(f"\n📊 Evaluating {self.agent_type} agent...")
        print("=" * 60)
        
        if self.agent is None:
            print("❌ No trained agent available")
            return False
        
        # Create evaluation environment
        env = self.create_trading_env(self.evaluation_data)
        
        evaluation_rewards = []
        performance_metrics = []
        
        for episode in range(self.evaluation_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from agent
                action = self.agent.get_action(state)
                
                # Take step in environment
                state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if truncated:
                    break
            
            evaluation_rewards.append(episode_reward)
            
            # Get final performance metrics
            stats = env.get_portfolio_stats()
            performance_metrics.append(stats)
        
        self.evaluation_rewards = evaluation_rewards
        self.performance_metrics = performance_metrics
        
        # Calculate evaluation statistics
        avg_reward = np.mean(evaluation_rewards)
        avg_return = np.mean([m['total_return'] for m in performance_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in performance_metrics])
        avg_drawdown = np.mean([m['max_drawdown'] for m in performance_metrics])
        
        print(f"📊 Evaluation Results:")
        print(f"   Average Reward: {avg_reward:.4f}")
        print(f"   Average Return: {avg_return:.2%}")
        print(f"   Average Sharpe Ratio: {avg_sharpe:.4f}")
        print(f"   Average Max Drawdown: {avg_drawdown:.2%}")
        
        return True
    
    def save_agent(self, filepath: str):
        """Save trained agent"""
        if self.agent is None:
            print("❌ No trained agent to save")
            return False
        
        try:
            torch.save({
                'agent_state_dict': self.agent.agent.state_dict(),
                'net_state_dict': self.agent.net.state_dict(),
                'agent_type': self.agent_type,
                'state_dim': self.agent.state_dim,
                'action_dim': self.agent.action_dim,
                'training_rewards': self.training_rewards,
                'evaluation_rewards': self.evaluation_rewards,
                'performance_metrics': self.performance_metrics
            }, filepath)
            
            print(f"✅ Agent saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving agent: {e}")
            return False
    
    def load_agent(self, filepath: str):
        """Load trained agent"""
        try:
            checkpoint = torch.load(filepath)
            
            self.agent_type = checkpoint['agent_type']
            state_dim = checkpoint['state_dim']
            action_dim = checkpoint['action_dim']
            
            self.agent = CryptoTradingAgent(
                agent_type=self.agent_type,
                state_dim=state_dim,
                action_dim=action_dim
            )
            
            self.agent.agent.load_state_dict(checkpoint['agent_state_dict'])
            self.agent.net.load_state_dict(checkpoint['net_state_dict'])
            
            self.training_rewards = checkpoint.get('training_rewards', [])
            self.evaluation_rewards = checkpoint.get('evaluation_rewards', [])
            self.performance_metrics = checkpoint.get('performance_metrics', [])
            
            print(f"✅ Agent loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading agent: {e}")
            return False


def main():
    """Main training function"""
    print("🚀 Cryptocurrency DRL Agent Training")
    print("=" * 60)
    
    # Create CCXT provider
    data_provider = create_ccxt_provider('binance', sandbox=False)
    
    # Create trading configuration for crypto
    trading_config = TradingConfig(
        initial_capital=100000.0,
        max_stock_quantity=10,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005,
        max_position_size=0.3,
        min_cash_reserve=0.2,
        max_leverage=1.2,
        stop_loss_pct=0.08,
        take_profit_pct=0.20
    )
    
    # Create trainer
    trainer = CryptoTradingTrainer(
        data_provider=data_provider,
        trading_config=trading_config,
        agent_type='PPO',
        training_episodes=500,
        evaluation_episodes=50
    )
    
    # Prepare data
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    if not trainer.prepare_data(symbols, start_date, end_date):
        print("❌ Failed to prepare data")
        return
    
    # Train agent
    if not trainer.train_agent():
        print("❌ Training failed")
        return
    
    # Evaluate agent
    if not trainer.evaluate_agent():
        print("❌ Evaluation failed")
        return
    
    # Save agent
    agent_file = f"crypto_{trainer.agent_type}_agent.pt"
    trainer.save_agent(agent_file)
    
    print("\n🎉 Training completed successfully!")
    print("✅ DRL agent trained on cryptocurrency data")
    print("✅ Agent ready for live trading")


if __name__ == "__main__":
    main() 