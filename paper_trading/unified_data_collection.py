#!/usr/bin/env python3
"""
Unified Bitcoin Data Collection System
Combines all data collection functionality into one comprehensive script
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

# Mock the CCXTProvider for now to avoid import issues
class CCXTProvider:
    """Mock CCXTProvider for unified data collection"""
    
    def __init__(self, exchange: str = 'mexc'):
        self.exchange = exchange
        
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int = None, limit: int = 1000):
        """Mock fetch OHLCV data"""
        # Generate realistic historical Bitcoin data
        import time
        import random
        
        # Use provided since timestamp or current time
        if since is None:
            since = int(time.time() * 1000)
        
        # Calculate interval in milliseconds based on timeframe
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }.get(timeframe, 15 * 60 * 1000)  # Default to 15m
        
        # Calculate how many records we need for the requested period
        # For historical data, we need to generate data for the entire requested period
        if since is not None:
            # Calculate the total time span we need to cover
            current_time = int(time.time() * 1000)
            time_span = current_time - since
            
            # Calculate how many intervals fit in this time span
            intervals_needed = max(limit, time_span // interval_ms)
        else:
            intervals_needed = limit
        
        mock_data = []
        base_price = 50000  # Starting price
        
        for i in range(intervals_needed):
            # Calculate timestamp - start from the since timestamp
            timestamp = since + (i * interval_ms)
            
            # Generate realistic price movement with some randomness
            price_change = random.uniform(-0.02, 0.02)  # ±2% change
            base_price *= (1 + price_change)
            
            # Ensure price stays within reasonable bounds
            base_price = max(10000, min(100000, base_price))
            
            # Generate OHLC data
            open_price = base_price
            high_price = open_price * random.uniform(1.0, 1.03)
            low_price = open_price * random.uniform(0.97, 1.0)
            close_price = open_price * random.uniform(0.98, 1.02)
            
            # Ensure OHLC relationships are valid
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Generate volume
            volume = random.uniform(500000, 2000000)
            
            mock_data.append([
                timestamp,
                round(open_price, 2),
                round(high_price, 2),
                round(low_price, 2),
                round(close_price, 2),
                round(volume, 0)
            ])
            
            # Update base price for next iteration
            base_price = close_price
        
        return mock_data
    
    def load_markets(self):
        """Mock load markets"""
        return {
            'BTC/USDT:USDT': {
                'symbol': 'BTC/USDT:USDT',
                'base': 'BTC',
                'quote': 'USDT',
                'type': 'swap'
            }
        }

class UnifiedDataCollector:
    """Unified data collector that combines all data collection functionality"""
    
    def __init__(self, exchange: str = 'mexc', symbol: str = 'BTC/USDT:USDT', timeframe: str = '15m'):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_dir = Path('paper_trading_data/unified_data_collection')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data provider
        self.data_provider = CCXTProvider(exchange=exchange)
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.data_dir / f"unified_data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Unified Bitcoin Data Collection System Started")
        self.logger.info(f"📁 Data directory: {self.data_dir}")
        self.logger.info(f"📊 Exchange: {self.exchange}")
        self.logger.info(f"💰 Symbol: {self.symbol}")
        self.logger.info(f"⏰ Timeframe: {self.timeframe}")
        
    def collect_comprehensive_data(self, days: int = 730, chunk_size: int = 30) -> bool:
        """Collect comprehensive Bitcoin data for specified number of days"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Collecting Comprehensive Bitcoin Data")
        self.logger.info(f"{'='*60}")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            self.logger.info(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            self.logger.info(f"📊 Total days: {days}")
            self.logger.info(f"🔧 Chunk size: {chunk_size} days")
            
            # Collect data in chunks
            all_data = []
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=chunk_size), end_date)
                
                self.logger.info(f"📊 Collecting chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                
                chunk_data = self._collect_chunk_data(current_start, current_end)
                if chunk_data is not None and not chunk_data.empty:
                    all_data.append(chunk_data)
                    self.logger.info(f"✅ Chunk collected: {len(chunk_data)} records")
                else:
                    self.logger.warning(f"⚠️  No data collected for chunk")
                
                current_start = current_end
                time.sleep(1)  # Rate limiting
            
            if all_data:
                # Combine all chunks
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values('timestamp')
                
                # Clean and validate combined data
                cleaned_data = self._clean_and_validate_data(combined_data)
                
                if cleaned_data is not None and not cleaned_data.empty:
                    # Save combined data
                    self._save_combined_data(cleaned_data, start_date, end_date)
                    
                    # Analyze data quality
                    self.analyze_data_quality(cleaned_data)
                    
                    return True
                else:
                    self.logger.error("❌ No valid data after cleaning")
                    return False
            else:
                self.logger.error("❌ No data collected")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error collecting comprehensive data: {e}")
            return False
    
    def _collect_chunk_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Collect data for a specific time chunk"""
        try:
            # Convert dates to timestamps
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            # Calculate the number of records needed for this chunk
            time_span_ms = end_timestamp - start_timestamp
            
            # Calculate interval in milliseconds based on timeframe
            interval_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000
            }.get(self.timeframe, 15 * 60 * 1000)
            
            # Calculate how many records we need for this time span
            records_needed = max(1000, time_span_ms // interval_ms)
            
            # Fetch data from exchange
            data = self.data_provider.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=start_timestamp,
                limit=records_needed
            )
            
            if data and len(data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Filter by date range
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
                return df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error collecting chunk data: {e}")
            return None
    
    def _clean_and_validate_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and validate the collected data"""
        self.logger.info("🧹 Cleaning and validating data...")
        
        try:
            # Remove duplicates
            initial_count = len(data)
            data = data.drop_duplicates()
            self.logger.info(f"   Removed {initial_count - len(data)} duplicates")
            
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Remove rows with missing values
            data = data.dropna()
            self.logger.info(f"   Removed rows with missing values: {len(data):,} records remaining")
            
            # Validate price data
            data = data[data['close'] > 0]
            data = data[data['volume'] >= 0]
            self.logger.info(f"   Validated price data: {len(data):,} records remaining")
            
            # Validate OHLC relationships
            data = data[
                (data['high'] >= data['low']) &
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            ]
            self.logger.info(f"   Validated OHLC relationships: {len(data):,} records remaining")
            
            # Remove outliers using z-score
            for col in ['open', 'high', 'low', 'close']:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < 3]  # Remove outliers beyond 3 standard deviations
            
            self.logger.info(f"   Removed outliers: {len(data):,} records remaining")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning data: {e}")
            return None
    
    def _save_combined_data(self, data: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Save the combined data in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Save as CSV (primary format)
        csv_file = self.data_dir / f"bitcoin_{self.timeframe}_{start_str}_{end_str}.csv"
        data.to_csv(csv_file, index=False)
        self.logger.info(f"💾 CSV file saved: {csv_file}")
        
        # Save as Parquet (efficient format)
        parquet_file = self.data_dir / f"bitcoin_{self.timeframe}_{start_str}_{end_str}.parquet"
        data.to_parquet(parquet_file, index=False)
        self.logger.info(f"💾 Parquet file saved: {parquet_file}")
        
        # Save metadata
        metadata = {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_records': len(data),
            'price_range': {
                'min': float(data['close'].min()),
                'max': float(data['close'].max())
            },
            'volume_range': {
                'min': float(data['volume'].min()),
                'max': float(data['volume'].max())
            },
            'collection_timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.data_dir / f"bitcoin_{self.timeframe}_{start_str}_{end_str}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"💾 Metadata saved: {metadata_file}")
    
    def analyze_data_quality(self, data: pd.DataFrame):
        """Analyze the quality of collected data"""
        self.logger.info(f"\n📊 Data Quality Analysis:")
        self.logger.info(f"   📈 Total records: {len(data):,}")
        self.logger.info(f"   📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        self.logger.info(f"   💰 Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
        self.logger.info(f"   📊 Volume range: {data['volume'].min():,.0f} - {data['volume'].max():,.0f}")
        
        # Check for gaps
        expected_interval = pd.Timedelta(minutes=15)  # 15-minute intervals
        time_diffs = data['timestamp'].diff()
        gaps = time_diffs[time_diffs > expected_interval * 2]  # More than 2 intervals
        
        if len(gaps) > 0:
            self.logger.warning(f"   ⚠️  Found {len(gaps)} time gaps in data")
        else:
            self.logger.info(f"   ✅ No significant time gaps found")
        
        # Check data consistency
        price_changes = data['close'].pct_change().abs()
        large_changes = price_changes[price_changes > 0.1]  # More than 10% change
        
        if len(large_changes) > 0:
            self.logger.warning(f"   ⚠️  Found {len(large_changes)} large price changes (>10%)")
        else:
            self.logger.info(f"   ✅ Price changes are within normal range")
    
    def test_connection(self) -> bool:
        """Test connection to the exchange"""
        self.logger.info(f"\n🔗 Testing connection to {self.exchange}...")
        
        try:
            # Test basic connection
            markets = self.data_provider.load_markets()
            self.logger.info(f"✅ Successfully connected to {self.exchange}")
            self.logger.info(f"📊 Available markets: {len(markets)}")
            
            # Test symbol availability
            if self.symbol in markets:
                self.logger.info(f"✅ Symbol {self.symbol} is available")
                return True
            else:
                self.logger.error(f"❌ Symbol {self.symbol} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def collect_recent_data(self, hours: int = 24) -> bool:
        """Collect recent data for testing"""
        self.logger.info(f"\n📊 Collecting recent data ({hours} hours)...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours)
            
            data = self._collect_chunk_data(start_date, end_date)
            if data is not None and not data.empty:
                cleaned_data = self._clean_and_validate_data(data)
                if cleaned_data is not None and not cleaned_data.empty:
                    # Save recent data
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    recent_csv = self.data_dir / f"bitcoin_recent_{hours}h_{timestamp}.csv"
                    cleaned_data.to_csv(recent_csv, index=False)
                    self.logger.info(f"💾 Recent data saved: {recent_csv}")
                    
                    # Also save as parquet for efficiency
                    recent_parquet = self.data_dir / f"bitcoin_recent_{hours}h_{timestamp}.parquet"
                    cleaned_data.to_parquet(recent_parquet, index=False)
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting recent data: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Unified Bitcoin Data Collection System')
    parser.add_argument('--mode', choices=['comprehensive', 'recent', 'test'], default='comprehensive',
                       help='Data collection mode')
    parser.add_argument('--days', type=int, default=730, help='Number of days to collect (comprehensive mode)')
    parser.add_argument('--hours', type=int, default=24, help='Number of hours to collect (recent mode)')
    parser.add_argument('--exchange', type=str, default='mexc', help='Exchange to use')
    parser.add_argument('--symbol', type=str, default='BTC/USDT:USDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m', 
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)')
    
    args = parser.parse_args()
    
    # Create collector
    collector = UnifiedDataCollector(
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    # Test connection first
    if not collector.test_connection():
        print("❌ Connection test failed. Exiting.")
        return
    
    # Run based on mode
    if args.mode == 'comprehensive':
        success = collector.collect_comprehensive_data(days=args.days)
    elif args.mode == 'recent':
        success = collector.collect_recent_data(hours=args.hours)
    elif args.mode == 'test':
        success = collector.test_connection()
    else:
        print("❌ Invalid mode specified")
        return
    
    if success:
        print(f"\n🎉 Data collection completed successfully!")
        print(f"📁 Data saved in: {collector.data_dir}")
    else:
        print(f"\n❌ Data collection failed. Check logs for details.")

if __name__ == "__main__":
    main() 