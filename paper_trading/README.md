# 🚀 Unified Bitcoin DRL Trading System

## 📊 **Overview**
This directory contains the unified Bitcoin DRL trading system with all functionality consolidated into 5 main scripts.

## 🎯 **Core Scripts**

### 1. **`unified_training.py`** - Training System
- Train single or all DRL agents
- Comprehensive logging and evaluation
- Support for all 7 agent types (PPO, SAC, TD3, DDPG, A2C, DQN, DuelingDQN)

**Usage:**
```bash
# Train single agent
python unified_training.py --mode single --agent-type SAC --episodes 200

# Train all agents
python unified_training.py --mode all --episodes 200
```

### 2. **`unified_data_collection.py`** - Data Collection System
- Collect comprehensive Bitcoin historical data
- Data validation and cleaning
- Multiple output formats (Parquet, CSV, JSON)

**Usage:**
```bash
# Collect 2 years of data
python unified_data_collection.py --mode comprehensive --days 730

# Collect recent data
python unified_data_collection.py --mode recent --hours 24

# Test connection
python unified_data_collection.py --mode test
```

### 3. **`unified_live_trading.py`** - Live Trading System
- Real-time trading with DRL agents or simple strategies
- Portfolio management and risk control
- Session logging and performance tracking

**Usage:**
```bash
# Live trading with DRL agent
python unified_live_trading.py --model-path path/to/model.pt --duration 60

# Demo trading with simple strategy
python unified_live_trading.py --no-drl --duration 30
```

### 4. **`unified_backtesting.py`** - Backtesting System
- Comprehensive backtesting with DRL agents
- Simple strategy comparison
- Performance metrics and equity curves

**Usage:**
```bash
# Backtest DRL agent
python unified_backtesting.py --mode drl --model-path path/to/model.pt --test-days 30

# Backtest simple strategy
python unified_backtesting.py --mode simple --test-days 30

# Backtest both
python unified_backtesting.py --mode both --model-path path/to/model.pt --test-days 30
```

### 5. **`unified_pipeline.py`** - Complete Pipeline System
- End-to-end workflow from data collection to live trading
- Modular pipeline components
- Comprehensive logging and reporting

**Usage:**
```bash
# Complete pipeline
python unified_pipeline.py --mode complete --agent-type SAC --episodes 200

# Individual pipeline steps
python unified_pipeline.py --mode data --data-days 730
python unified_pipeline.py --mode train --agent-type SAC --episodes 200
python unified_pipeline.py --mode backtest --model-path path/to/model.pt
python unified_pipeline.py --mode live --model-path path/to/model.pt
```

## 🎯 **Quick Start**

1. **Collect Data:**
   ```bash
   python unified_data_collection.py --mode comprehensive --days 730
   ```

2. **Train Agent:**
   ```bash
   python unified_training.py --mode single --agent-type SAC --episodes 200
   ```

3. **Backtest:**
   ```bash
   python unified_backtesting.py --mode drl --model-path path/to/model.pt --test-days 30
   ```

4. **Live Trading:**
   ```bash
   python unified_live_trading.py --model-path path/to/model.pt --duration 60
   ```

## 📁 **Directory Structure**
```
paper_trading/
├── unified_training.py          # Training system
├── unified_data_collection.py   # Data collection system
├── unified_live_trading.py      # Live trading system
├── unified_backtesting.py       # Backtesting system
├── unified_pipeline.py          # Complete pipeline
├── cleanup_script.py           # Cleanup utility
├── scripts_summary.md          # Scripts documentation
├── README_BITCOIN_TRADING.md   # Detailed documentation
├── requirements.txt            # Dependencies
├── main.py                    # Main entry point
└── __init__.py               # Package initialization
```

## 🎉 **Features**
- ✅ **Unified Scripts**: All functionality consolidated into 5 main scripts
- ✅ **Comprehensive Logging**: Detailed logging for all operations
- ✅ **Multiple Agents**: Support for all 7 DRL agent types
- ✅ **Data Management**: Robust data collection and validation
- ✅ **Live Trading**: Real-time trading with risk management
- ✅ **Backtesting**: Comprehensive performance evaluation
- ✅ **Pipeline System**: End-to-end workflow automation

## 📊 **Results**
All results are saved in organized directories:
- `paper_trading_data/unified_training_results/` - Training results
- `paper_trading_data/unified_data_collection/` - Collected data
- `paper_trading_data/unified_live_trading/` - Live trading results
- `paper_trading_data/unified_backtesting/` - Backtesting results
- `paper_trading_data/unified_pipeline/` - Pipeline results

---
*This unified system provides a complete Bitcoin DRL trading solution with all functionality consolidated into maintainable, well-documented scripts.*
