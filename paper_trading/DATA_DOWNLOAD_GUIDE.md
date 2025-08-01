# 📊 Bitcoin Data Download Guide

## 🎯 **Complete Data Collection System**

This guide shows you how to download Bitcoin data in different timeframes and date ranges using our unified data collection system.

## 📋 **Available Timeframes**

| Timeframe | Description | Records per Day | Best For |
|-----------|-------------|-----------------|----------|
| `1m` | 1 minute | 1,440 | High-frequency trading |
| `5m` | 5 minutes | 288 | Short-term analysis |
| `15m` | 15 minutes | 96 | Medium-term analysis |
| `30m` | 30 minutes | 48 | Swing trading |
| `1h` | 1 hour | 24 | Daily analysis |
| `4h` | 4 hours | 6 | Weekly analysis |
| `1d` | 1 day | 1 | Long-term analysis |

## 🚀 **Download Commands**

### **1. 2 Years of Historical Data (Recommended for Training)**

```bash
# 15-minute data (most common for DRL training)
python unified_data_collection.py --mode comprehensive --days 730 --timeframe 15m

# 1-hour data (faster processing)
python unified_data_collection.py --mode comprehensive --days 730 --timeframe 1h

# 5-minute data (high frequency)
python unified_data_collection.py --mode comprehensive --days 730 --timeframe 5m
```

### **2. 1 Month of Data (Quick Testing)**

```bash
# 15-minute data
python unified_data_collection.py --mode comprehensive --days 30 --timeframe 15m

# 1-hour data
python unified_data_collection.py --mode comprehensive --days 30 --timeframe 1h

# 5-minute data
python unified_data_collection.py --mode comprehensive --days 30 --timeframe 5m
```

### **3. 1 Week of Data (Quick Demo)**

```bash
# 15-minute data
python unified_data_collection.py --mode comprehensive --days 7 --timeframe 15m

# 1-hour data
python unified_data_collection.py --mode comprehensive --days 7 --timeframe 1h

# 5-minute data
python unified_data_collection.py --mode comprehensive --days 7 --timeframe 5m
```

### **4. Recent Data (Last 24 Hours)**

```bash
# 15-minute data
python unified_data_collection.py --mode recent --hours 24 --timeframe 15m

# 5-minute data
python unified_data_collection.py --mode recent --hours 24 --timeframe 5m

# 1-minute data
python unified_data_collection.py --mode recent --hours 24 --timeframe 1m
```

### **5. Connection Test**

```bash
python unified_data_collection.py --mode test
```

## 📁 **Output Files**

For each download, you'll get:

1. **CSV File** (Primary format): `bitcoin_{timeframe}_{start_date}_{end_date}.csv`
2. **Parquet File** (Efficient format): `bitcoin_{timeframe}_{start_date}_{end_date}.parquet`
3. **Metadata File**: `bitcoin_{timeframe}_{start_date}_{end_date}_metadata.json`
4. **Log File**: Detailed collection logs

## 📊 **Data Format**

All CSV files contain:
```csv
timestamp,open,high,low,close,volume
2025-07-25 11:56:04.012,46499.55,47349.75,45553.7,46926.57,1708850.0
2025-07-25 12:01:04.012,46869.32,47638.78,45801.81,47638.78,1359471.0
```

## 🎯 **Recommended Downloads by Use Case**

### **For DRL Training (Best Quality)**
```bash
python unified_data_collection.py --mode comprehensive --days 730 --timeframe 15m
```
- **Records**: ~24,000 data points
- **Time Range**: 2 years
- **Use**: Training deep reinforcement learning agents

### **For Quick Testing**
```bash
python unified_data_collection.py --mode comprehensive --days 30 --timeframe 1h
```
- **Records**: ~720 data points
- **Time Range**: 1 month
- **Use**: Quick model testing and validation

### **For High-Frequency Analysis**
```bash
python unified_data_collection.py --mode comprehensive --days 7 --timeframe 5m
```
- **Records**: ~2,000 data points
- **Time Range**: 1 week
- **Use**: High-frequency trading strategies

### **For Long-Term Analysis**
```bash
python unified_data_collection.py --mode comprehensive --days 730 --timeframe 1d
```
- **Records**: ~730 data points
- **Time Range**: 2 years
- **Use**: Long-term trend analysis

## 📈 **Data Quality Features**

- ✅ **Automatic Cleaning**: Removes duplicates, outliers, and invalid data
- ✅ **OHLC Validation**: Ensures proper high/low/open/close relationships
- ✅ **Volume Validation**: Validates volume data integrity
- ✅ **Time Gap Detection**: Identifies missing data periods
- ✅ **Price Change Analysis**: Detects unusual price movements

## 🔧 **Advanced Options**

### **Custom Date Ranges**
```bash
# 6 months of data
python unified_data_collection.py --mode comprehensive --days 180 --timeframe 15m

# 3 months of data
python unified_data_collection.py --mode comprehensive --days 90 --timeframe 1h

# 1 year of data
python unified_data_collection.py --mode comprehensive --days 365 --timeframe 4h
```

### **Different Symbols**
```bash
# Ethereum data
python unified_data_collection.py --mode comprehensive --days 30 --timeframe 15m --symbol ETH/USDT:USDT

# Bitcoin data (default)
python unified_data_collection.py --mode comprehensive --days 30 --timeframe 15m --symbol BTC/USDT:USDT
```

## 📊 **Expected File Sizes**

| Timeframe | 1 Month | 6 Months | 1 Year | 2 Years |
|-----------|---------|----------|--------|---------|
| 1m | ~40 MB | ~240 MB | ~480 MB | ~960 MB |
| 5m | ~8 MB | ~48 MB | ~96 MB | ~192 MB |
| 15m | ~3 MB | ~16 MB | ~32 MB | ~64 MB |
| 1h | ~0.5 MB | ~3 MB | ~6 MB | ~12 MB |
| 1d | ~0.1 MB | ~0.5 MB | ~1 MB | ~2 MB |

## 🎉 **Success Indicators**

When download completes successfully, you'll see:
- ✅ Connection test passed
- ✅ Data collected with realistic records
- ✅ CSV and Parquet files saved
- ✅ Data quality analysis completed
- ✅ No significant errors in logs

## 🚨 **Troubleshooting**

### **If download fails:**
1. Check internet connection
2. Verify exchange availability
3. Try smaller date ranges first
4. Check log files for detailed errors

### **If data seems incomplete:**
1. Verify the date range is correct
2. Check for time gaps in the data
3. Ensure the timeframe is appropriate for the date range

---

**🎯 Ready to download your Bitcoin data! Choose the timeframe and date range that best fits your needs.** 