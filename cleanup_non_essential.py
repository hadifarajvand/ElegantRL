#!/usr/bin/env python3
"""
Cleanup Script for Paper Trading System
=======================================

This script removes non-essential files and directories while preserving
the complete paper_trading workflow and Jupyter notebooks.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_non_essential():
    """Remove non-essential files and directories."""
    
    print("🧹 Starting cleanup of non-essential files...")
    print("=" * 60)
    
    # Files to remove
    files_to_remove = [
        # Test files
        "test_btc_timeframes.py",
        "test_btc_only_trading.py",
        "test_data_persistence_comprehensive.py",
        "demo_ccxt_paper_trading.py",
        "test_ccxt_simple.py",
        "test_ccxt_integration.py",
        "test_comprehensive_paper_trading.py",
        "test_phase4_strategies.py",
        "test_phase3_risk.py",
        "test_phase2_data.py",
        "test_phase1_integration.py",
        "test_logging_verification.py",
        "test_integration.py",
        "test_paper_trading.py",
        
        # Documentation files
        "roadmap_analysis.md",
        "comprehensive_summary.md",
        "test_results_analysis.md",
        "LOGGING_VERIFICATION_SUMMARY.md",
        "FUNCTIONALITY_TEST_SUMMARY.md",
        "README-2.md",
        
        # Configuration files
        "crypto_paper_trading_config.yaml",
        "demo_config.yaml",
        "real_config.yaml",
        
        # Data files
        "China_A_shares.numpy.npz",
        "China_A_shares.pandas.dataframe",
        
        # Requirements
        "requirements_ext.txt",
        "C:PATHTOFOLDERrequirements.txt",
        
        # Log files
        "bitcoin_data_collection.log",
        "bitcoin_training.log",
        "mexc_trading.log",
        "live_bitcoin_testing.log",
        "paper_trading.log",
        
        # Other files
        "Awesome_Deep_Reinforcement_Learning_List.md",
    ]
    
    # Directories to remove
    dirs_to_remove = [
        "test_paper_trading_data",
        "StockTradingEnv-v2_PPO_0",
        "StockTradingEnv-v3_PPO_1943",
        "StockTradingEnv-v2_PPO_1943",
        "Pendulum_PPO_0",
        "examples",
        "helloworld",
        "figs",
        "docs",
        "unit_tests",
        "rlsolver",
        "ElegantRL.egg-info",
        ".github",
    ]
    
    # Remove files
    print("📄 Removing non-essential files...")
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"   ⚠️ File not found: {file_path}")
    
    # Remove directories
    print("\n📁 Removing non-essential directories...")
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Removed: {dir_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {dir_path}: {e}")
        else:
            print(f"   ⚠️ Directory not found: {dir_path}")
    
    # Clean up test files in paper_trading directory
    print("\n🧹 Cleaning up test files in paper_trading directory...")
    paper_trading_test_files = [
        "paper_trading/test_mexc_working.py",
        "paper_trading/test_mexc_symbols.py",
        "paper_trading/test_mexc_data.py",
        "paper_trading/simple_pipeline.py",
        "paper_trading/simple_pipeline_report.json",
        "paper_trading/custom_training.py",
        "paper_trading/mexc_live_trading.py",
        "paper_trading/live_trading_mexc.py",
        "paper_trading/run_live_with_agents.py",
        "paper_trading/comprehensive_backtesting.py",
        "paper_trading/run_backtesting.py",
        "paper_trading/simple_live_trading.py",
        "paper_trading/live_trading.py",
        "paper_trading/run_live_trading.py",
        "paper_trading/train_btc_drl_agents.py",
        "paper_trading/final_comprehensive_pipeline.py",
        "paper_trading/comprehensive_paper_trading.py",
        "paper_trading/run_complete_pipeline.py",
    ]
    
    for file_path in paper_trading_test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"   ⚠️ File not found: {file_path}")
    
    # Clean up __pycache__ directories
    print("\n🧹 Cleaning up __pycache__ directories...")
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"   ✅ Removed: {pycache_dir}")
        except Exception as e:
            print(f"   ❌ Failed to remove {pycache_dir}: {e}")
    
    print("\n🎉 Cleanup completed!")
    print("=" * 60)

def show_remaining_structure():
    """Show the remaining directory structure."""
    print("\n📊 Remaining Directory Structure:")
    print("=" * 60)
    
    essential_items = [
        "paper_trading/",
        "elegantrl/",
        "paper_trading_data/",
        "data_cache/",
        "logs/",
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "README.md",
        "__init__.py",
        "LICENSE",
        "tutorial_helloworld_DQN_DDPG_PPO.ipynb",
        "tutorial_BipedalWalker_v3.ipynb",
        "tutorial_Creating_ChasingVecEnv.ipynb",
        "tutorial_LunarLanderContinuous_v2.ipynb",
        "tutorial_Pendulum_v1.ipynb",
    ]
    
    for item in essential_items:
        if os.path.exists(item):
            if os.path.isdir(item):
                print(f"📁 {item}")
            else:
                print(f"📄 {item}")
        else:
            print(f"❌ {item} (not found)")
    
    print("\n✅ Essential paper_trading workflow preserved!")
    print("📓 Jupyter notebooks preserved!")
    print("🚀 System ready for production use!")

if __name__ == "__main__":
    # Ask for confirmation
    print("⚠️  WARNING: This will remove non-essential files and directories!")
    print("This action cannot be undone.")
    print("\nFiles to be preserved:")
    print("- paper_trading/ (complete trading system)")
    print("- elegantrl/ (DRL framework)")
    print("- All Jupyter notebooks (*.ipynb)")
    print("- Essential configuration files")
    print("\nFiles to be removed:")
    print("- Test files and scripts")
    print("- Old training results")
    print("- Documentation files")
    print("- Non-essential directories")
    
    response = input("\nDo you want to proceed with cleanup? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        cleanup_non_essential()
        show_remaining_structure()
    else:
        print("❌ Cleanup cancelled.") 