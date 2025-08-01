"""
Test CCXT Integration with Paper Trading System
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trading.data.market_data import create_ccxt_provider, DataManager
from paper_trading.configs.trading_config import TradingConfig
from paper_trading.models.trading_env import EnhancedStockTradingEnv
from paper_trading.utils.metrics import PerformanceMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_ccxt_data_fetching():
    """Test CCXT data fetching functionality"""
    print("🚀 Testing CCXT Data Fetching")
    print("=" * 50)
    
    try:
        # Create CCXT provider
        provider = create_ccxt_provider('binance', sandbox=False)
        
        # Test connection
        if not provider.test_connection():
            print("❌ Failed to connect to Binance")
            return False
        
        print("✅ Connected to Binance successfully")
        
        # Get available symbols
        symbols = provider.get_available_symbols()
        print(f"📊 Available symbols: {len(symbols)}")
        print(f"Sample symbols: {symbols[:5]}")
        
        # Test symbols for trading
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        # Get historical data
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        data = {}
        for symbol in test_symbols:
            print(f"\n📈 Fetching data for {symbol}...")
            symbol_data = provider.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            
            if not symbol_data.empty:
                data[symbol] = symbol_data
                print(f"✅ {symbol}: {len(symbol_data)} records")
                print(f"   Price range: ${symbol_data['low'].min():,.2f} - ${symbol_data['high'].max():,.2f}")
                print(f"   Latest price: ${symbol_data['close'].iloc[-1]:,.2f}")
            else:
                print(f"❌ No data for {symbol}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_paper_trading_with_ccxt():
    """Test paper trading system with CCXT data"""
    print("\n🚀 Testing Paper Trading with CCXT Data")
    print("=" * 50)
    
    try:
        # Get CCXT data
        data = test_ccxt_data_fetching()
        if not data:
            print("❌ No data available for testing")
            return False
        
        # Create trading configuration
        trading_config = TradingConfig(
            initial_capital=100000.0,  # $100k starting capital
            max_stock_quantity=10,     # Max 10 units per crypto
            transaction_cost_pct=0.001, # 0.1% transaction cost
            slippage_pct=0.0005,      # 0.05% slippage
            max_position_size=0.2,     # Max 20% in single position
            min_cash_reserve=0.1,      # Keep 10% cash reserve
            max_leverage=1.5,          # Max 1.5x leverage
            stop_loss_pct=0.05,        # 5% stop loss
            take_profit_pct=0.15       # 15% take profit
        )
        
        print("✅ Trading configuration created")
        
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
        
        # Run a simple trading simulation
        print("\n🔄 Running trading simulation...")
        
        state, info = env.reset()
        total_reward = 0
        step_count = 0
        
        # Simulate 10 trading steps
        for step in range(10):
            # Generate random action (in real system, this would come from DRL agent)
            action = env.action_space.sample()
            
            # Take step
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Log progress
            if step_count % 5 == 0:
                print(f"   Step {step_count}: Portfolio Value = ${info['total_asset']:,.2f}, Reward = {reward:.4f}")
            
            if done or truncated:
                break
        
        # Get final portfolio stats
        stats = env.get_portfolio_stats()
        
        print(f"\n📊 Trading Simulation Results:")
        print(f"   Total Steps: {step_count}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Final Portfolio Value: ${stats['total_value']:,.2f}")
        print(f"   Total Return: {stats['total_return']:.2%}")
        print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {stats['max_drawdown']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in paper trading test: {e}")
        return False


def test_real_time_data():
    """Test real-time data fetching"""
    print("\n🚀 Testing Real-time Data Fetching")
    print("=" * 50)
    
    try:
        # Create CCXT provider
        provider = create_ccxt_provider('binance', sandbox=False)
        
        # Test symbols
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        
        # Get real-time prices
        prices = provider.get_realtime_data(symbols)
        
        if prices:
            print("✅ Real-time prices fetched:")
            for symbol, price in prices.items():
                print(f"   {symbol}: ${price:,.2f}")
        else:
            print("❌ No real-time data available")
        
        return len(prices) > 0
        
    except Exception as e:
        print(f"❌ Error in real-time data test: {e}")
        return False


def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\n🚀 Testing Technical Indicators")
    print("=" * 50)
    
    try:
        # Create CCXT provider
        provider = create_ccxt_provider('binance', sandbox=False)
        
        # Get historical data
        data = provider.get_historical_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-31',
            interval='1d'
        )
        
        if data.empty:
            print("❌ No data available for technical indicators")
            return False
        
        # Calculate technical indicators
        data_with_indicators = provider.get_technical_indicators(data)
        
        print(f"✅ Technical indicators calculated")
        print(f"   Original columns: {len(data.columns)}")
        print(f"   With indicators: {len(data_with_indicators.columns)}")
        
        # Show some key indicators
        indicator_columns = ['SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']
        available_indicators = [col for col in indicator_columns if col in data_with_indicators.columns]
        
        if available_indicators:
            print(f"\n📊 Sample indicators (latest values):")
            latest_data = data_with_indicators.iloc[-1]
            for indicator in available_indicators:
                value = latest_data[indicator]
                print(f"   {indicator}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in technical indicators test: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 CCXT Integration Test for Paper Trading System")
    print("=" * 60)
    
    # Test 1: Data fetching
    data_fetching_success = test_ccxt_data_fetching()
    
    # Test 2: Paper trading simulation
    if data_fetching_success:
        paper_trading_success = test_paper_trading_with_ccxt()
    else:
        paper_trading_success = False
    
    # Test 3: Real-time data
    realtime_success = test_real_time_data()
    
    # Test 4: Technical indicators
    indicators_success = test_technical_indicators()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Data Fetching: {'PASS' if data_fetching_success else 'FAIL'}")
    print(f"✅ Paper Trading: {'PASS' if paper_trading_success else 'FAIL'}")
    print(f"✅ Real-time Data: {'PASS' if realtime_success else 'FAIL'}")
    print(f"✅ Technical Indicators: {'PASS' if indicators_success else 'FAIL'}")
    
    total_tests = 4
    passed_tests = sum([data_fetching_success, paper_trading_success, realtime_success, indicators_success])
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! CCXT integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main() 