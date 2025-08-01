#!/usr/bin/env python3
"""
Unified Pipeline System
Combines all pipeline functionality into one comprehensive script
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

class UnifiedPipeline:
    """Unified pipeline that combines all pipeline functionality"""
    
    def __init__(self):
        self.results_dir = Path('paper_trading_data/unified_pipeline')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Pipeline state
        self.pipeline_results = {}
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.results_dir / f"unified_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Unified Pipeline System Started")
        self.logger.info(f"📁 Results directory: {self.results_dir}")
        
    def run_data_collection_pipeline(self, days: int = 730) -> bool:
        """Run data collection pipeline"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 Data Collection Pipeline")
        self.logger.info(f"{'='*60}")
        
        try:
            # Import and run data collection
            from unified_data_collection import UnifiedDataCollector
            
            collector = UnifiedDataCollector()
            success = collector.collect_comprehensive_data(days=days)
            
            if success:
                self.pipeline_results['data_collection'] = {
                    'status': 'success',
                    'days_collected': days,
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.info("✅ Data collection pipeline completed successfully")
                return True
            else:
                self.pipeline_results['data_collection'] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.error("❌ Data collection pipeline failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in data collection pipeline: {e}")
            return False
    
    def run_training_pipeline(self, agent_type: str = 'SAC', episodes: int = 200) -> bool:
        """Run training pipeline"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 Training Pipeline: {agent_type}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Import and run training
            from unified_training import UnifiedBitcoinTrainer
            
            trainer = UnifiedBitcoinTrainer()
            result = trainer.train_single_agent(agent_type, episodes)
            
            if result:
                self.pipeline_results['training'] = {
                    'status': 'success',
                    'agent_type': agent_type,
                    'episodes': episodes,
                    'final_reward': result['final_reward'],
                    'model_path': result['model_path'],
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.info("✅ Training pipeline completed successfully")
                return True
            else:
                self.pipeline_results['training'] = {
                    'status': 'failed',
                    'agent_type': agent_type,
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.error("❌ Training pipeline failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in training pipeline: {e}")
            return False
    
    def run_backtesting_pipeline(self, agent_type: str = 'SAC', model_path: str = None, test_days: int = 30) -> bool:
        """Run backtesting pipeline"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 Backtesting Pipeline: {agent_type}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Import and run backtesting
            from unified_backtesting import UnifiedBacktester
            
            backtester = UnifiedBacktester()
            
            # Run DRL backtest if model path provided
            if model_path:
                drl_results = backtester.run_drl_backtest(
                    agent_type=agent_type,
                    model_path=model_path,
                    test_period_days=test_days
                )
                
                if drl_results:
                    self.pipeline_results['backtesting'] = {
                        'status': 'success',
                        'agent_type': agent_type,
                        'total_return': drl_results['total_return'],
                        'sharpe_ratio': drl_results['sharpe_ratio'],
                        'max_drawdown': drl_results['max_drawdown'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.logger.info("✅ Backtesting pipeline completed successfully")
                    return True
            
            # Run simple strategy backtest
            simple_results = backtester.run_simple_strategy_backtest(test_period_days=test_days)
            
            if simple_results:
                self.pipeline_results['backtesting'] = {
                    'status': 'success',
                    'strategy': 'simple',
                    'total_return': simple_results['total_return'],
                    'sharpe_ratio': simple_results['sharpe_ratio'],
                    'max_drawdown': simple_results['max_drawdown'],
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.info("✅ Backtesting pipeline completed successfully")
                return True
            else:
                self.pipeline_results['backtesting'] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                }
                self.logger.error("❌ Backtesting pipeline failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error in backtesting pipeline: {e}")
            return False
    
    def run_live_trading_pipeline(self, agent_type: str = 'SAC', model_path: str = None, duration_minutes: int = 60) -> bool:
        """Run live trading pipeline"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"💰 Live Trading Pipeline: {agent_type}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Import and run live trading
            from unified_live_trading import UnifiedLiveTrader
            
            trader = UnifiedLiveTrader(use_drl=model_path is not None)
            
            # Load DRL agent if model path provided
            if model_path:
                if not trader.load_drl_agent(agent_type, model_path):
                    self.logger.error("❌ Failed to load DRL agent for live trading")
                    return False
            
            # Run trading session
            trader.run_trading_session(duration_minutes=duration_minutes)
            
            self.pipeline_results['live_trading'] = {
                'status': 'success',
                'agent_type': agent_type,
                'duration_minutes': duration_minutes,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.info("✅ Live trading pipeline completed successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"❌ Error in live trading pipeline: {e}")
            return False
    
    def run_complete_pipeline(self, agent_type: str = 'SAC', episodes: int = 200, test_days: int = 30, live_duration: int = 60) -> bool:
        """Run complete end-to-end pipeline"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 COMPLETE END-TO-END PIPELINE")
        self.logger.info(f"{'='*80}")
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Data Collection
            self.logger.info(f"\n📊 Step 1: Data Collection")
            if not self.run_data_collection_pipeline():
                self.logger.error("❌ Pipeline failed at data collection step")
                return False
            
            # Step 2: Training
            self.logger.info(f"\n🎯 Step 2: Training")
            if not self.run_training_pipeline(agent_type, episodes):
                self.logger.error("❌ Pipeline failed at training step")
                return False
            
            # Get model path from training results
            model_path = self.pipeline_results['training'].get('model_path')
            
            # Step 3: Backtesting
            self.logger.info(f"\n🔍 Step 3: Backtesting")
            if not self.run_backtesting_pipeline(agent_type, model_path, test_days):
                self.logger.error("❌ Pipeline failed at backtesting step")
                return False
            
            # Step 4: Live Trading
            self.logger.info(f"\n💰 Step 4: Live Trading")
            if not self.run_live_trading_pipeline(agent_type, model_path, live_duration):
                self.logger.error("❌ Pipeline failed at live trading step")
                return False
            
            # Pipeline completed successfully
            pipeline_time = time.time() - pipeline_start_time
            
            self.pipeline_results['complete_pipeline'] = {
                'status': 'success',
                'total_time': pipeline_time,
                'agent_type': agent_type,
                'episodes': episodes,
                'test_days': test_days,
                'live_duration': live_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"\n🎉 COMPLETE PIPELINE SUCCESSFULLY COMPLETED!")
            self.logger.info(f"⏱️  Total pipeline time: {pipeline_time:.2f}s")
            
            # Save pipeline results
            self._save_pipeline_results()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error in complete pipeline: {e}")
            return False
    
    def _save_pipeline_results(self):
        """Save pipeline results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save pipeline results
            results_file = self.results_dir / f"pipeline_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.pipeline_results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Pipeline results saved: {results_file}")
            
            # Generate pipeline summary
            self._generate_pipeline_summary()
            
        except Exception as e:
            self.logger.error(f"❌ Error saving pipeline results: {e}")
    
    def _generate_pipeline_summary(self):
        """Generate pipeline summary"""
        try:
            self.logger.info(f"\n📊 PIPELINE SUMMARY:")
            
            if 'data_collection' in self.pipeline_results:
                status = self.pipeline_results['data_collection']['status']
                self.logger.info(f"   📊 Data Collection: {'✅' if status == 'success' else '❌'}")
            
            if 'training' in self.pipeline_results:
                status = self.pipeline_results['training']['status']
                agent_type = self.pipeline_results['training'].get('agent_type', 'Unknown')
                final_reward = self.pipeline_results['training'].get('final_reward', 0)
                self.logger.info(f"   🎯 Training ({agent_type}): {'✅' if status == 'success' else '❌'} (Reward: {final_reward:.3f})")
            
            if 'backtesting' in self.pipeline_results:
                status = self.pipeline_results['backtesting']['status']
                total_return = self.pipeline_results['backtesting'].get('total_return', 0)
                self.logger.info(f"   🔍 Backtesting: {'✅' if status == 'success' else '❌'} (Return: {total_return:.2%})")
            
            if 'live_trading' in self.pipeline_results:
                status = self.pipeline_results['live_trading']['status']
                self.logger.info(f"   💰 Live Trading: {'✅' if status == 'success' else '❌'}")
            
            if 'complete_pipeline' in self.pipeline_results:
                total_time = self.pipeline_results['complete_pipeline'].get('total_time', 0)
                self.logger.info(f"   ⏱️  Total Pipeline Time: {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating pipeline summary: {e}")

def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline System')
    parser.add_argument('--mode', choices=['data', 'train', 'backtest', 'live', 'complete'], default='complete',
                       help='Pipeline mode')
    parser.add_argument('--agent-type', type=str, default='SAC',
                       help='DRL agent type')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--test-days', type=int, default=30,
                       help='Number of days for backtesting')
    parser.add_argument('--live-duration', type=int, default=60,
                       help='Live trading duration in minutes')
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model')
    parser.add_argument('--data-days', type=int, default=730,
                       help='Number of days to collect data')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = UnifiedPipeline()
    
    # Run based on mode
    if args.mode == 'data':
        success = pipeline.run_data_collection_pipeline(args.data_days)
    elif args.mode == 'train':
        success = pipeline.run_training_pipeline(args.agent_type, args.episodes)
    elif args.mode == 'backtest':
        success = pipeline.run_backtesting_pipeline(args.agent_type, args.model_path, args.test_days)
    elif args.mode == 'live':
        success = pipeline.run_live_trading_pipeline(args.agent_type, args.model_path, args.live_duration)
    elif args.mode == 'complete':
        success = pipeline.run_complete_pipeline(args.agent_type, args.episodes, args.test_days, args.live_duration)
    else:
        print("❌ Invalid mode specified")
        return
    
    if success:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved in: {pipeline.results_dir}")
    else:
        print(f"\n❌ Pipeline failed. Check logs for details.")

if __name__ == "__main__":
    main() 