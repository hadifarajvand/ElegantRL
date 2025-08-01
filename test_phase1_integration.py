#!/usr/bin/env python3
"""
Test Phase 1: Core Integration Implementations
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_elegantrl_integration():
    """Test ElegantRL integration"""
    print("🔧 Testing ElegantRL Integration...")
    
    try:
        from paper_trading.training.elegantrl_integration import ElegantRLTrainer, train_with_elegantrl
        from paper_trading.configs.trading_config import TradingConfig
        from paper_trading.configs.model_config import ModelConfig, PPO_CONFIG
        
        # Create configurations
        trading_config = TradingConfig(initial_capital=100000)
        model_config = PPO_CONFIG
        
        # Create trainer
        trainer = ElegantRLTrainer(trading_config, model_config)
        print("✅ ElegantRLTrainer created successfully")
        
        # Create mock data
        mock_data = {
            'AAPL': pd.DataFrame({
                'close': np.random.randn(100) * 10 + 150,
                'volume': np.random.randint(1000, 10000, 100),
                'open': np.random.randn(100) * 5 + 150,
                'high': np.random.randn(100) * 5 + 155,
                'low': np.random.randn(100) * 5 + 145
            })
        }
        
        # Test environment preparation
        env = trainer.prepare_environment(mock_data)
        print(f"✅ Environment prepared - State dim: {env.state_dim}, Action dim: {env.action_dim}")
        
        # Test configuration creation
        config = trainer.create_elegantrl_config(env, 'PPO')
        print(f"✅ ElegantRL config created - Agent: {config.agent_class.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ ElegantRL integration test failed: {e}")
        return False

def test_vectorized_environment():
    """Test vectorized environment"""
    print("\n🔄 Testing Vectorized Environment...")
    
    try:
        from paper_trading.models.vectorized_env import EnhancedStockTradingVecEnv, create_vectorized_env
        
        # Create mock data
        mock_data = {
            'AAPL': pd.DataFrame({
                'close': np.random.randn(50) * 10 + 150,
                'volume': np.random.randint(1000, 10000, 50),
                'open': np.random.randn(50) * 5 + 150,
                'high': np.random.randn(50) * 5 + 155,
                'low': np.random.randn(50) * 5 + 145
            })
        }
        
        # Create vectorized environment
        vec_env = create_vectorized_env(mock_data, num_envs=4, initial_capital=100000)
        print(f"✅ Vectorized environment created - {vec_env.num_envs} environments")
        
        # Test reset
        states, info = vec_env.reset()
        print(f"✅ Environment reset - States shape: {states.shape}")
        
        # Test step
        actions = np.random.randn(4, 1) * 0.1  # 4 environments, 1 action each
        next_states, rewards, dones, truncateds, info = vec_env.step(actions)
        print(f"✅ Environment step - Rewards shape: {rewards.shape}")
        
        # Test environment info
        env_info = vec_env.get_env_info()
        print(f"✅ Environment info: {env_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vectorized environment test failed: {e}")
        return False

def test_advanced_agents():
    """Test advanced agent implementations"""
    print("\n🤖 Testing Advanced Agents...")
    
    try:
        from paper_trading.models.advanced_agents import (
            TradingAgentPPO, TradingAgentSAC, TradingAgentDQN,
            EnsembleTradingAgent, create_trading_agent
        )
        
        # Test agent creation
        net_dims = [128, 64]
        state_dim = 33
        action_dim = 1
        
        # Test PPO agent
        ppo_agent = create_trading_agent('PPO', net_dims, state_dim, action_dim)
        print("✅ TradingAgentPPO created successfully")
        
        # Test SAC agent
        sac_agent = create_trading_agent('SAC', net_dims, state_dim, action_dim)
        print("✅ TradingAgentSAC created successfully")
        
        # Test DQN agent
        dqn_agent = create_trading_agent('DQN', net_dims, state_dim, action_dim)
        print("✅ TradingAgentDQN created successfully")
        
        # Test trading actions
        test_state = np.random.randn(state_dim)
        
        ppo_action = ppo_agent.get_trading_action(test_state)
        print(f"✅ PPO trading action: {ppo_action.shape}")
        
        sac_action = sac_agent.get_trading_action(test_state)
        print(f"✅ SAC trading action: {sac_action.shape}")
        
        dqn_action = dqn_agent.get_trading_action(test_state)
        print(f"✅ DQN trading action: {dqn_action.shape}")
        
        # Test ensemble agent
        agents = [ppo_agent, sac_agent, dqn_agent]
        ensemble = EnsembleTradingAgent(agents)
        print("✅ EnsembleTradingAgent created successfully")
        
        ensemble_action = ensemble.get_ensemble_action(test_state, method="weighted")
        print(f"✅ Ensemble action: {ensemble_action.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced agents test failed: {e}")
        return False

def test_integration_workflow():
    """Test complete integration workflow"""
    print("\n🔄 Testing Integration Workflow...")
    
    try:
        from paper_trading.training.elegantrl_integration import ElegantRLTrainer
        from paper_trading.models.vectorized_env import create_vectorized_env
        from paper_trading.models.advanced_agents import create_trading_agent
        from paper_trading.configs.trading_config import TradingConfig
        from paper_trading.configs.model_config import PPO_CONFIG
        
        # Create configurations
        trading_config = TradingConfig(initial_capital=100000)
        model_config = PPO_CONFIG
        
        # Create mock data
        mock_data = {
            'AAPL': pd.DataFrame({
                'close': np.random.randn(50) * 10 + 150,
                'volume': np.random.randint(1000, 10000, 50),
                'open': np.random.randn(50) * 5 + 150,
                'high': np.random.randn(50) * 5 + 155,
                'low': np.random.randn(50) * 5 + 145
            })
        }
        
        # Test vectorized environment
        vec_env = create_vectorized_env(mock_data, num_envs=2)
        print("✅ Vectorized environment created")
        
        # Test trainer with vectorized environment
        trainer = ElegantRLTrainer(trading_config, model_config)
        print("✅ ElegantRL trainer created")
        
        # Test agent creation
        agent = create_trading_agent('PPO', [128, 64], vec_env.state_dim, vec_env.action_dim)
        print("✅ Trading agent created")
        
        # Test environment-agent interaction
        states, info = vec_env.reset()
        actions = agent.get_trading_action(states[0])  # Use first environment state
        next_states, rewards, dones, truncateds, info = vec_env.step([actions, actions])
        print("✅ Environment-agent interaction successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Phase 1: Core Integration Implementations")
    print("=" * 60)
    
    tests = [
        ("ElegantRL Integration", test_elegantrl_integration),
        ("Vectorized Environment", test_vectorized_environment),
        ("Advanced Agents", test_advanced_agents),
        ("Integration Workflow", test_integration_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PHASE 1 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Phase 1 implementation completed successfully!")
    else:
        print("⚠️ Some Phase 1 tests failed. Please check the errors above.") 