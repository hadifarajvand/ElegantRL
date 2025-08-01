#!/usr/bin/env python3
"""
Cleanup Script for Paper Trading Directory
Removes unnecessary files and logs to clean up the directory
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def cleanup_paper_trading_directory():
    """Clean up the paper_trading directory"""
    print("🧹 Starting cleanup of paper_trading directory...")
    
    # Files to remove (old scripts that are now unified)
    files_to_remove = [
        # Old training scripts
        'train_bitcoin_drl.py',
        'train_all_available_agents.py',
        'train_all_agents.py',
        'train_with_bitcoin_data.py',
        'enhanced_training_with_logging.py',
        'comprehensive_training_demo.py',
        'custom_training.py',
        'train_btc_drl_agents.py',
        
        # Old data collection scripts
        'collect_bitcoin_data.py',
        'test_mexc_data.py',
        'test_mexc_symbols.py',
        'test_mexc_working.py',
        
        # Old live trading scripts
        'simple_mexc_trading.py',
        'live_bitcoin_testing.py',
        'live_trading_mexc.py',
        'live_trading.py',
        'simple_live_trading.py',
        'mexc_live_trading.py',
        'run_live_trading.py',
        'run_live_with_agents.py',
        
        # Old backtesting scripts
        'comprehensive_backtesting.py',
        'run_backtesting.py',
        
        # Old pipeline scripts
        'final_comprehensive_pipeline.py',
        'comprehensive_paper_trading.py',
        'simple_pipeline.py',
        'run_complete_pipeline.py',
        
        # Old demo scripts
        'demo_bitcoin_trading.py',
        
        # Log files
        'comprehensive_training_demo_20250730_225603.log',
        'bitcoin_training.log',
        
        # Report files
        'comprehensive_training_report_20250730_225724.json',
        'simple_pipeline_report.json',
        
        # Old documentation files
        'agents_comparison_summary.md',
        'training_summary.md',
        'comprehensive_agents_comparison.md',
        
        # Check data ranges script (temporary)
        'check_data_ranges.py'
    ]
    
    # Directories to remove (if empty or contain only old files)
    dirs_to_remove = [
        'comprehensive_paper_trading',
        'data_cache',
        'results'
    ]
    
    # Files to keep (essential files)
    files_to_keep = [
        'unified_training.py',
        'unified_data_collection.py',
        'unified_live_trading.py',
        'unified_backtesting.py',
        'unified_pipeline.py',
        'cleanup_script.py',
        'scripts_summary.md',
        'README_BITCOIN_TRADING.md',
        'requirements.txt',
        'main.py',
        '__init__.py',
        'README.md'
    ]
    
    # Counters
    files_removed = 0
    dirs_removed = 0
    
    # Remove files
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"🗑️  Removed file: {file_name}")
                files_removed += 1
            except Exception as e:
                print(f"❌ Error removing {file_name}: {e}")
    
    # Remove directories
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"🗑️  Removed directory: {dir_name}")
                dirs_removed += 1
            except Exception as e:
                print(f"❌ Error removing directory {dir_name}: {e}")
    
    # Clean up __pycache__ directories
    pycache_dirs = list(Path('.').rglob('__pycache__'))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"🗑️  Removed __pycache__: {pycache_dir}")
            dirs_removed += 1
        except Exception as e:
            print(f"❌ Error removing __pycache__ {pycache_dir}: {e}")
    
    # Clean up .pyc files
    pyc_files = list(Path('.').rglob('*.pyc'))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print(f"🗑️  Removed .pyc file: {pyc_file}")
            files_removed += 1
        except Exception as e:
            print(f"❌ Error removing .pyc file {pyc_file}: {e}")
    
    # Clean up log files in subdirectories
    log_files = list(Path('.').rglob('*.log'))
    for log_file in log_files:
        try:
            log_file.unlink()
            print(f"🗑️  Removed log file: {log_file}")
            files_removed += 1
        except Exception as e:
            print(f"❌ Error removing log file {log_file}: {e}")
    
    # Clean up old report files
    report_files = list(Path('.').rglob('*_report_*.json'))
    for report_file in report_files:
        try:
            report_file.unlink()
            print(f"🗑️  Removed report file: {report_file}")
            files_removed += 1
        except Exception as e:
            print(f"❌ Error removing report file {report_file}: {e}")
    
    # Summary
    print(f"\n🎉 Cleanup completed!")
    print(f"📊 Files removed: {files_removed}")
    print(f"📁 Directories removed: {dirs_removed}")
    
    # Show remaining files
    remaining_files = [f for f in Path('.').iterdir() if f.is_file() and f.name.endswith('.py')]
    print(f"\n📋 Remaining Python files:")
    for file in sorted(remaining_files):
        print(f"   ✅ {file.name}")
    
    # Show remaining directories
    remaining_dirs = [d for d in Path('.').iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"\n📁 Remaining directories:")
    for dir in sorted(remaining_dirs):
        print(f"   📁 {dir.name}")
    
    print(f"\n✨ Directory cleaned up successfully!")

def create_new_readme():
    """Create a new README for the cleaned up directory"""
    readme_content = """# 🚀 Unified Bitcoin DRL Trading System

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
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("📝 Created new README.md")

if __name__ == "__main__":
    print("🧹 Paper Trading Directory Cleanup")
    print("=" * 50)
    
    # Confirm before proceeding
    response = input("Are you sure you want to clean up the directory? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cleanup cancelled.")
        sys.exit(0)
    
    # Run cleanup
    cleanup_paper_trading_directory()
    
    # Create new README
    create_new_readme()
    
    print("\n🎉 Cleanup completed successfully!")
    print("📁 Directory is now clean and organized with unified scripts.") 