# Bitcoin DRL Trading System

A comprehensive deep reinforcement learning trading system for Bitcoin using ElegantRL framework, featuring data collection, model training, and live testing capabilities.

## 🚀 Features

- **Real-time Data Collection**: Collect Bitcoin data from MEXC exchange
- **DRL Model Training**: Train PPO, SAC, and TD3 agents on historical data
- **Live Testing**: Test trained models on real-time data
- **Comprehensive Metrics**: Track performance, returns, and risk metrics
- **Paper Trading**: Safe testing environment with virtual portfolio
- **Multi-timeframe Support**: 15m, 1h, 4h, 1d timeframes
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands

## 📁 System Components

### 1. Enhanced MEXC Trading (`simple_mexc_trading.py`)
- Real-time Bitcoin trading with MEXC exchange
- Support for both simple strategies and DRL agents
- Paper trading with virtual portfolio
- Comprehensive logging and results tracking

### 2. Bitcoin DRL Training (`train_bitcoin_drl.py`)
- Collect historical Bitcoin data
- Train DRL agents (PPO, SAC, TD3)
- Model evaluation and backtesting
- Performance metrics calculation

### 3. Live Bitcoin Testing (`live_bitcoin_testing.py`)
- Live testing of trained models
- Real-time data monitoring
- Portfolio tracking and performance analysis
- Threaded data collection

## 🛠️ Installation

### Prerequisites
```bash
pip install ccxt pandas numpy torch elegantrl gymnasium
```

### Setup
1. Clone the repository
2. Install dependencies
3. Configure exchange API keys (optional for public data)

## 📊 Usage Examples

### 1. Data Collection

Collect 90 days of Bitcoin data:
```bash
python paper_trading/train_bitcoin_drl.py --mode collect --data-days 90
```

### 2. Model Training

Train a PPO agent for 1000 episodes:
```bash
python paper_trading/train_bitcoin_drl.py --mode train --agent-type PPO --episodes 1000
```

### 3. Model Evaluation

Evaluate a trained model:
```bash
python paper_trading/train_bitcoin_drl.py --mode evaluate --model-path path/to/model.pth
```

### 4. Backtesting

Run backtest on historical data:
```bash
python paper_trading/train_bitcoin_drl.py --mode backtest \
    --model-path path/to/model.pth \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

### 5. Live Testing

Test trained model on live data:
```bash
python paper_trading/live_bitcoin_testing.py \
    --model-path path/to/model.pth \
    --duration 24 \
    --initial-capital 100000
```

### 6. Enhanced MEXC Trading

Run paper trading session:
```bash
python paper_trading/simple_mexc_trading.py --mode trade --session-minutes 60
```

Train and use DRL agent:
```bash
# First train
python paper_trading/simple_mexc_trading.py --mode train --training-episodes 1000

# Then use trained model
python paper_trading/simple_mexc_trading.py --mode trade --use-drl --model-path path/to/model.pth
```

## 🎯 Training Process

### 1. Data Collection
- Collects historical Bitcoin data from MEXC
- Adds technical indicators (RSI, MACD, Moving Averages)
- Cleans and prepares data for training

### 2. Environment Setup
- Creates trading environment with realistic constraints
- Implements transaction costs and slippage
- Manages portfolio and position sizing

### 3. Agent Training
- Trains DRL agents using ElegantRL
- Implements validation during training
- Saves best models based on validation performance

### 4. Evaluation
- Evaluates trained models on test data
- Calculates performance metrics
- Generates detailed reports

## 📈 Performance Metrics

The system tracks comprehensive metrics:

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of executed trades
- **Portfolio Value**: Current portfolio worth

## 🔧 Configuration

### Trading Parameters
```python
# Default configuration
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.1  # 10% per trade
TRANSACTION_COST = 0.001  # 0.1%
TIMEFRAME = '15m'
SYMBOL = 'BTC/USDT:USDT'
```

### DRL Agent Parameters
```python
# Training parameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
```

## 📁 Output Files

### Data Files
- `bitcoin_15m_YYYYMMDD_YYYYMMDD.parquet`: Historical data
- `training_results_YYYYMMDD_HHMMSS.json`: Training results
- `backtest_YYYY-MM-DD_YYYY-MM-DD_YYYYMMDD_HHMMSS.json`: Backtest results

### Model Files
- `bitcoin_drl_final_mexc_YYYYMMDD_HHMMSS.pth`: Final trained model
- `bitcoin_drl_epX_rewardY_YYYYMMDD_HHMMSS.pth`: Checkpoint models

### Log Files
- `bitcoin_training.log`: Training logs
- `live_bitcoin_testing.log`: Live testing logs
- `mexc_trading.log`: MEXC trading logs

## 🚨 Important Notes

### Risk Management
- This is a **paper trading system** - no real money is at risk
- Always test thoroughly before any live trading
- Monitor performance metrics closely
- Implement proper risk management strategies

### Data Quality
- Historical data quality affects model performance
- Clean and validate data before training
- Consider market conditions and regime changes

### Model Limitations
- DRL models may not generalize to all market conditions
- Regular retraining may be necessary
- Monitor for model drift and performance degradation

## 🔄 Workflow

### Complete Training Workflow
1. **Data Collection**: Collect historical Bitcoin data
2. **Data Preparation**: Clean and add technical indicators
3. **Environment Setup**: Create trading environment
4. **Model Training**: Train DRL agent
5. **Model Evaluation**: Evaluate on test data
6. **Backtesting**: Test on historical periods
7. **Live Testing**: Test on real-time data
8. **Performance Monitoring**: Track and analyze results

### Quick Start
```bash
# 1. Collect data
python paper_trading/train_bitcoin_drl.py --mode collect

# 2. Train model
python paper_trading/train_bitcoin_drl.py --mode train --episodes 500

# 3. Live test
python paper_trading/live_bitcoin_testing.py --model-path paper_trading_data/bitcoin_training/bitcoin_drl_final_mexc_*.pth --quick 60
```

## 🛡️ Safety Features

- **Paper Trading Only**: No real money trading
- **Error Handling**: Comprehensive error handling and logging
- **Graceful Shutdown**: Proper cleanup on interruption
- **Data Validation**: Input validation and data quality checks
- **Performance Monitoring**: Real-time performance tracking

## 📊 Example Results

### Training Results
```
Episode 500/1000 - Train Reward: 1250.45, Val Reward: 1180.32
Episode 1000/1000 - Train Reward: 1450.67, Val Reward: 1420.89

✅ Training completed!
📊 Final Training Reward: 1450.67
🏆 Best Validation Reward: 1420.89
💾 Best Model: paper_trading_data/bitcoin_training/bitcoin_drl_ep950_reward1420.89_20240101_120000.pth
```

### Live Testing Results
```
📊 LIVE TEST COMPLETED
============================================================
📈 Total Return: +12.45%
💰 Final Portfolio: $112,450.00
🎯 Total Trades: 24
⏱️ Duration: 24:00:00
============================================================
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check internet connection
   - Verify exchange API status
   - Check API rate limits

2. **Data Issues**
   - Ensure sufficient historical data
   - Check data quality and completeness
   - Verify symbol format

3. **Training Issues**
   - Adjust learning rate
   - Increase training episodes
   - Check for data normalization

4. **Performance Issues**
   - Monitor system resources
   - Optimize batch sizes
   - Use GPU if available

## 📚 Advanced Usage

### Custom Strategies
- Implement custom trading strategies
- Add new technical indicators
- Modify reward functions

### Multi-Asset Trading
- Extend to multiple cryptocurrencies
- Implement portfolio optimization
- Add correlation analysis

### Risk Management
- Implement stop-loss mechanisms
- Add position sizing algorithms
- Create dynamic risk models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## 📄 License

This project is for educational and research purposes. Use at your own risk.

## ⚠️ Disclaimer

This software is for educational purposes only. It does not constitute financial advice. Trading cryptocurrencies involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making any investment decisions.

---

**Happy Trading! 🚀📈** 