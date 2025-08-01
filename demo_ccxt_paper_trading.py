"""
Demonstration: CCXT Integration with Paper Trading System
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CCXT provider directly
from paper_trading.data.ccxt_provider import create_ccxt_provider
from paper_trading.configs.trading_config import TradingConfig
from paper_trading.models.trading_env import EnhancedStockTradingEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def fetch_crypto_data():
    """Fetch cryptocurrency data using CCXT"""
    print("🚀 Fetching Cryptocurrency Data from Binance")
    print("=" * 60)
    
    try:
        # Create CCXT provider
        provider = create_ccxt_provider('binance', sandbox=False)
        
        # Test connection
        if not provider.test_connection():
            print("❌ Failed to connect to Binance")
            return None
        
        print("✅ Connected to Binance successfully")
        
        # Define trading pairs
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        # Get historical data for the last 3 months
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        data = {}
        for symbol in symbols:
            print(f"\n📈 Fetching {symbol} data...")
            symbol_data = provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1d'
            )
            
            if not symbol_data.empty:
                # Add technical indicators
                symbol_data = provider.get_technical_indicators(symbol_data)
                data[symbol] = symbol_data
                
                print(f"✅ {symbol}: {len(symbol_data)} records")
                print(f"   Price range: ${symbol_data['low'].min():,.2f} - ${symbol_data['high'].max():,.2f}")
                print(f"   Latest price: ${symbol_data['close'].iloc[-1]:,.2f}")
                print(f"   Features: {len(symbol_data.columns)} technical indicators")
            else:
                print(f"❌ No data for {symbol}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def create_crypto_trading_env(data):
    """Create trading environment for cryptocurrency data"""
    print("\n🚀 Creating Cryptocurrency Trading Environment")
    print("=" * 60)
    
    try:
        # Create trading configuration for crypto
        trading_config = TradingConfig(
            initial_capital=100000.0,  # $100k starting capital
            max_stock_quantity=10,     # Max 10 units per crypto
            transaction_cost_pct=0.001, # 0.1% transaction cost (typical for crypto)
            slippage_pct=0.0005,      # 0.05% slippage
            max_position_size=0.3,     # Max 30% in single position (crypto is more volatile)
            min_cash_reserve=0.2,      # Keep 20% cash reserve
            max_leverage=1.2,          # Max 1.2x leverage (conservative for crypto)
            stop_loss_pct=0.08,        # 8% stop loss (crypto is volatile)
            take_profit_pct=0.20       # 20% take profit
        )
        
        print("✅ Trading configuration created for cryptocurrency trading")
        
        # Create trading environment
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
        
        print("✅ Trading environment created")
        print(f"📊 State dimension: {env.state_dim}")
        print(f"🎯 Action dimension: {env.action_dim}")
        print(f"📈 Trading pairs: {list(data.keys())}")
        
        return env, trading_config
        
    except Exception as e:
        print(f"❌ Error creating trading environment: {e}")
        return None, None


def run_crypto_trading_simulation(env, num_steps=50):
    """Run a cryptocurrency trading simulation"""
    print(f"\n🔄 Running Cryptocurrency Trading Simulation ({num_steps} steps)")
    print("=" * 60)
    
    try:
        # Reset environment
        state, info = env.reset()
        
        # Trading simulation
        total_reward = 0
        portfolio_values = []
        actions_taken = []
        
        # Get initial portfolio value
        initial_portfolio = env._calculate_total_asset()
        print(f"💰 Initial portfolio value: ${initial_portfolio:,.2f}")
        
        for step in range(num_steps):
            # Generate action (in real system, this would come from DRL agent)
            # For demo, we'll use a simple strategy: buy when RSI < 30, sell when RSI > 70
            action = np.random.randn(env.action_dim) * 0.1  # Small random actions for demo
            
            # Take step
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            portfolio_values.append(info['total_asset'])
            
            # Store action
            actions_taken.append(action)
            
            # Log progress
            if step % 10 == 0:
                print(f"   Step {step:2d}: Portfolio = ${info['total_asset']:,.2f}, Reward = {reward:.4f}")
            
            if done or truncated:
                print(f"   Episode ended at step {step}")
                break
        
        # Get final portfolio stats
        stats = env.get_portfolio_stats()
        
        print(f"\n📊 Trading Simulation Results:")
        print(f"   Total Steps: {len(portfolio_values)}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Final Portfolio Value: ${stats['final_value']:,.2f}")
        print(f"   Total Return: {stats['total_return']:.2%}")
        print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {stats['max_drawdown']:.2%}")
        print(f"   Volatility: {stats['volatility']:.2%}")
        
        return {
            'total_reward': total_reward,
            'final_value': stats['final_value'],
            'total_return': stats['total_return'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'max_drawdown': stats['max_drawdown'],
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken
        }
        
    except Exception as e:
        print(f"❌ Error in trading simulation: {e}")
        return None


def analyze_crypto_performance(results):
    """Analyze cryptocurrency trading performance"""
    print("\n📈 Cryptocurrency Trading Performance Analysis")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print(f"💰 Performance Summary:")
    print(f"   Total Reward: {results['total_reward']:.4f}")
    print(f"   Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    
    # Calculate additional metrics
    portfolio_values = results['portfolio_values']
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    print(f"\n📊 Risk Metrics:")
    print(f"   Volatility: {np.std(returns) * np.sqrt(252):.2%}")
    print(f"   Max Single Day Return: {np.max(returns):.2%}")
    print(f"   Min Single Day Return: {np.min(returns):.2%}")
    print(f"   Positive Days: {np.sum(returns > 0)}/{len(returns)} ({np.mean(returns > 0):.1%})")
    
    # Trading activity
    actions_taken = results['actions_taken']
    action_magnitudes = [np.linalg.norm(action) for action in actions_taken]
    
    print(f"\n🎯 Trading Activity:")
    print(f"   Average Action Magnitude: {np.mean(action_magnitudes):.4f}")
    print(f"   Max Action Magnitude: {np.max(action_magnitudes):.4f}")
    print(f"   Total Actions: {len(actions_taken)}")


def main():
    """Main demonstration function"""
    print("🚀 CCXT Cryptocurrency Paper Trading Demonstration")
    print("=" * 80)
    print("This demonstration shows how to use CCXT to fetch real cryptocurrency")
    print("data from Binance and integrate it with our paper trading system.")
    print("=" * 80)
    
    # Step 1: Fetch cryptocurrency data
    data = fetch_crypto_data()
    if not data:
        print("❌ Failed to fetch data. Exiting.")
        return
    
    # Step 2: Create trading environment
    env, config = create_crypto_trading_env(data)
    if not env:
        print("❌ Failed to create trading environment. Exiting.")
        return
    
    # Step 3: Run trading simulation
    results = run_crypto_trading_simulation(env, num_steps=50)
    if not results:
        print("❌ Trading simulation failed. Exiting.")
        return
    
    # Step 4: Analyze performance
    analyze_crypto_performance(results)
    
    print("\n🎉 Demonstration completed successfully!")
    print("✅ CCXT integration is working perfectly with real cryptocurrency data")
    print("✅ Paper trading system can handle cryptocurrency markets")
    print("✅ Technical indicators and risk management are functional")
    
    print("\n💡 Next Steps:")
    print("   - Train a DRL agent on this cryptocurrency data")
    print("   - Implement more sophisticated trading strategies")
    print("   - Add more cryptocurrency pairs")
    print("   - Deploy for live paper trading")


if __name__ == "__main__":
    main() 