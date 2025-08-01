#!/usr/bin/env python3
import pandas as pd

# Load the dataset
df = pd.read_parquet('paper_trading_data/bitcoin_data/bitcoin_15m_combined_20230731_20250730.parquet')

print("📊 Bitcoin Dataset Information:")
print("=" * 50)
print(f"Total records: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

# Calculate data splits
total_records = len(df)
train_end = int(total_records * 0.7)
val_end = int(total_records * 0.9)

train_data = df.iloc[:train_end]
val_data = df.iloc[train_end:val_end]
test_data = df.iloc[val_end:]

print("\n📈 Data Splits:")
print("=" * 50)
print(f"Training: {len(train_data):,} records (70%)")
print(f"  Date range: {train_data['timestamp'].min()} to {train_data['timestamp'].max()}")
print(f"  Price range: ${train_data['close'].min():,.2f} - ${train_data['close'].max():,.2f}")

print(f"\nValidation: {len(val_data):,} records (20%)")
print(f"  Date range: {val_data['timestamp'].min()} to {val_data['timestamp'].max()}")
print(f"  Price range: ${val_data['close'].min():,.2f} - ${val_data['close'].max():,.2f}")

print(f"\nTesting: {len(test_data):,} records (10%)")
print(f"  Date range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
print(f"  Price range: ${test_data['close'].min():,.2f} - ${test_data['close'].max():,.2f}")

print("\n🎯 Training Configuration:")
print("=" * 50)
print("Episodes per agent: 200")
print("State dimension: 15 features")
print("Action dimension: 1 (buy/sell/hold)")
print("Initial capital: $100,000")
print("Timeframe: 15-minute intervals") 