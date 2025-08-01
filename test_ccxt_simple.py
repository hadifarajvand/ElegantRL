"""
Simple CCXT Test for Paper Trading System
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CCXT provider directly
from paper_trading.data.ccxt_provider import create_ccxt_provider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_ccxt_basic():
    """Test basic CCXT functionality"""
    print("🚀 Testing Basic CCXT Functionality")
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_historical_data():
    """Test historical data fetching"""
    print("\n🚀 Testing Historical Data Fetching")
    print("=" * 50)
    
    try:
        # Create CCXT provider
        provider = create_ccxt_provider('binance', sandbox=False)
        
        # Test symbols
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
                timeframe='1d'
            )
            
            if not symbol_data.empty:
                data[symbol] = symbol_data
                print(f"✅ {symbol}: {len(symbol_data)} records")
                print(f"   Price range: ${symbol_data['low'].min():,.2f} - ${symbol_data['high'].max():,.2f}")
                print(f"   Latest price: ${symbol_data['close'].iloc[-1]:,.2f}")
                print(f"   Columns: {list(symbol_data.columns)}")
            else:
                print(f"❌ No data for {symbol}")
        
        return len(data) > 0  # Return boolean instead of dict
        
    except Exception as e:
        print(f"❌ Error: {e}")
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
            timeframe='1d'
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


def test_exchange_info():
    """Test exchange information"""
    print("\n🚀 Testing Exchange Information")
    print("=" * 50)
    
    try:
        # Create CCXT provider
        provider = create_ccxt_provider('binance', sandbox=False)
        
        # Get exchange info
        info = provider.get_exchange_info()
        
        if info:
            print("✅ Exchange information retrieved:")
            print(f"   Name: {info.get('name', 'N/A')}")
            print(f"   URL: {info.get('url', 'N/A')}")
            print(f"   Rate Limit: {info.get('rateLimit', 'N/A')}")
            print(f"   Sample Symbols: {info.get('symbols', [])[:5]}")
        else:
            print("❌ No exchange information available")
        
        return len(info) > 0
        
    except Exception as e:
        print(f"❌ Error in exchange info test: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Simple CCXT Test for Paper Trading System")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_success = test_ccxt_basic()
    
    # Test 2: Historical data
    historical_success = test_historical_data()
    
    # Test 3: Real-time data
    realtime_success = test_real_time_data()
    
    # Test 4: Technical indicators
    indicators_success = test_technical_indicators()
    
    # Test 5: Exchange info
    exchange_success = test_exchange_info()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Basic Functionality: {'PASS' if basic_success else 'FAIL'}")
    print(f"✅ Historical Data: {'PASS' if historical_success else 'FAIL'}")
    print(f"✅ Real-time Data: {'PASS' if realtime_success else 'FAIL'}")
    print(f"✅ Technical Indicators: {'PASS' if indicators_success else 'FAIL'}")
    print(f"✅ Exchange Information: {'PASS' if exchange_success else 'FAIL'}")
    
    total_tests = 5
    passed_tests = sum([basic_success, historical_success, realtime_success, indicators_success, exchange_success])
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! CCXT integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main() 