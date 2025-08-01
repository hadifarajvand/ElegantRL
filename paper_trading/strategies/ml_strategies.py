"""
Machine Learning Strategies for Paper Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

logger = logging.getLogger(__name__)


class MLStrategy:
    """
    Machine Learning Trading Strategy
    
    Features:
    - Multiple ML models
    - Feature engineering
    - Model training and validation
    - Real-time prediction
    - Model performance tracking
    """
    
    def __init__(self, symbols: List[str], **kwargs):
        self.symbols = symbols
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'target'
        self.lookback_period = kwargs.get('lookback_period', 20)
        self.prediction_threshold = kwargs.get('prediction_threshold', 0.6)
        
        # Model parameters
        self.model_type = kwargs.get('model_type', 'random_forest')
        self.retrain_frequency = kwargs.get('retrain_frequency', 30)  # days
        self.min_training_samples = kwargs.get('min_training_samples', 100)
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        
        logger.info(f"MLStrategy initialized for {len(symbols)} symbols")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            features = data.copy()
            
            # Price-based features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = features['close'].rolling(window=period).mean()
                features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
                features[f'price_sma_ratio_{period}'] = features['close'] / features[f'sma_{period}']
            
            # Volatility features
            for period in [5, 10, 20]:
                features[f'volatility_{period}'] = features['returns'].rolling(window=period).std()
                features[f'volatility_annualized_{period}'] = features[f'volatility_{period}'] * np.sqrt(252)
            
            # Volume features
            features['volume_ratio'] = features['volume'] / features['volume'].rolling(window=20).mean()
            features['volume_price_trend'] = features['returns'] * features['volume']
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(features['close'])
            features['macd'] = self._calculate_macd(features['close'])
            features['bb_position'] = self._calculate_bollinger_position(features['close'])
            
            # Momentum features
            for period in [1, 3, 5, 10]:
                features[f'momentum_{period}'] = features['close'].pct_change(period)
            
            # Target variable (next day return)
            features['target'] = features['close'].shift(-1) / features['close'] - 1
            features['target_binary'] = (features['target'] > 0).astype(int)
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        position = (prices - lower_band) / (upper_band - lower_band)
        return position
    
    def train_model(self, symbol: str, data: pd.DataFrame) -> bool:
        """Train ML model for a symbol"""
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            
            if len(features_df) < self.min_training_samples:
                logger.warning(f"Insufficient data for {symbol}: {len(features_df)} samples")
                return False
            
            # Define feature columns
            feature_columns = [
                'returns', 'log_returns', 'volume_ratio', 'volume_price_trend',
                'rsi', 'macd', 'bb_position'
            ]
            
            # Add moving average features
            for period in [5, 10, 20, 50]:
                feature_columns.extend([
                    f'sma_{period}', f'ema_{period}', f'price_sma_ratio_{period}'
                ])
            
            # Add volatility features
            for period in [5, 10, 20]:
                feature_columns.extend([
                    f'volatility_{period}', f'volatility_annualized_{period}'
                ])
            
            # Add momentum features
            for period in [1, 3, 5, 10]:
                feature_columns.append(f'momentum_{period}')
            
            # Filter available features
            available_features = [col for col in feature_columns if col in features_df.columns]
            
            if len(available_features) < 5:
                logger.warning(f"Insufficient features for {symbol}")
                return False
            
            # Prepare training data
            X = features_df[available_features]
            y = features_df['target_binary']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif self.model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            elif self.model_type == 'logistic_regression':
                model = LogisticRegression(random_state=42)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return False
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_columns = available_features
            
            # Store performance
            self.model_performance[symbol] = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'last_trained': pd.Timestamp.now()
            }
            
            logger.info(f"Model trained for {symbol} - Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False
    
    def predict(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Make prediction for a symbol"""
        try:
            if symbol not in self.models:
                logger.warning(f"No trained model for {symbol}")
                return None
            
            # Prepare features
            features_df = self.prepare_features(data)
            
            if len(features_df) == 0:
                return None
            
            # Get latest features
            latest_features = features_df[self.feature_columns].iloc[-1:]
            
            # Scale features
            scaler = self.scalers[symbol]
            features_scaled = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models[symbol]
            prediction_proba = model.predict_proba(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence
            confidence = max(prediction_proba)
            
            # Determine action
            if prediction == 1 and confidence > self.prediction_threshold:
                action = 'buy'
            elif prediction == 0 and confidence > self.prediction_threshold:
                action = 'sell'
            else:
                action = 'hold'
            
            # Store prediction
            prediction_record = {
                'symbol': symbol,
                'timestamp': data.index[-1] if hasattr(data, 'index') else None,
                'prediction': prediction,
                'confidence': confidence,
                'action': action,
                'features': latest_features.to_dict('records')[0]
            }
            
            self.prediction_history.append(prediction_record)
            
            return {
                'action': action,
                'confidence': confidence,
                'prediction': prediction,
                'prediction_proba': prediction_proba
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def retrain_models(self, market_data: Dict) -> bool:
        """Retrain all models"""
        try:
            success_count = 0
            
            for symbol in self.symbols:
                if symbol in market_data:
                    data = market_data[symbol]
                    if isinstance(data, pd.DataFrame) and len(data) > self.min_training_samples:
                        success = self.train_model(symbol, data)
                        if success:
                            success_count += 1
            
            logger.info(f"Retrained {success_count}/{len(self.symbols)} models")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance summary"""
        return self.model_performance
    
    def save_models(self, directory: str = 'models'):
        """Save trained models"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            for symbol, model in self.models.items():
                model_path = os.path.join(directory, f'{symbol}_model.pkl')
                scaler_path = os.path.join(directory, f'{symbol}_scaler.pkl')
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[symbol], scaler_path)
            
            logger.info(f"Models saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, directory: str = 'models'):
        """Load trained models"""
        try:
            for symbol in self.symbols:
                model_path = os.path.join(directory, f'{symbol}_model.pkl')
                scaler_path = os.path.join(directory, f'{symbol}_scaler.pkl')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
            
            logger.info(f"Models loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")


class EnsembleMLStrategy:
    """
    Ensemble Machine Learning Strategy
    
    Features:
    - Multiple ML models
    - Ensemble voting
    - Model diversity
    - Performance weighting
    """
    
    def __init__(self, symbols: List[str], **kwargs):
        self.symbols = symbols
        self.models = {}
        self.ensemble_method = kwargs.get('ensemble_method', 'voting')
        self.model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        # Initialize individual strategies
        self.strategies = {}
        for model_type in self.model_types:
            self.strategies[model_type] = MLStrategy(symbols, model_type=model_type, **kwargs)
        
        logger.info(f"EnsembleMLStrategy initialized with {len(self.model_types)} models")
    
    def train_ensemble(self, market_data: Dict) -> bool:
        """Train ensemble of models"""
        try:
            success_count = 0
            
            for model_type, strategy in self.strategies.items():
                success = strategy.retrain_models(market_data)
                if success:
                    success_count += 1
            
            logger.info(f"Trained {success_count}/{len(self.model_types)} ensemble models")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return False
    
    def predict_ensemble(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """Make ensemble prediction"""
        try:
            predictions = {}
            
            # Get predictions from all models
            for model_type, strategy in self.strategies.items():
                prediction = strategy.predict(symbol, data)
                if prediction:
                    predictions[model_type] = prediction
            
            if not predictions:
                return None
            
            # Ensemble aggregation
            if self.ensemble_method == 'voting':
                return self._voting_ensemble(predictions)
            elif self.ensemble_method == 'weighted':
                return self._weighted_ensemble(predictions)
            else:
                return self._average_ensemble(predictions)
                
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return None
    
    def _voting_ensemble(self, predictions: Dict[str, Dict]) -> Dict:
        """Voting ensemble method"""
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_confidence = 0
        
        for model_type, prediction in predictions.items():
            action = prediction['action']
            confidence = prediction['confidence']
            
            if action == 'buy':
                buy_votes += 1
            elif action == 'sell':
                sell_votes += 1
            else:
                hold_votes += 1
            
            total_confidence += confidence
        
        # Determine final action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_action = 'buy'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_action = 'sell'
        else:
            final_action = 'hold'
        
        avg_confidence = total_confidence / len(predictions)
        
        return {
            'action': final_action,
            'confidence': avg_confidence,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'hold_votes': hold_votes,
            'ensemble_method': 'voting'
        }
    
    def _weighted_ensemble(self, predictions: Dict[str, Dict]) -> Dict:
        """Weighted ensemble method"""
        weighted_action = 0
        total_weight = 0
        
        for model_type, prediction in predictions.items():
            # Use model performance as weight
            weight = prediction.get('confidence', 0.5)
            
            action_value = 1 if prediction['action'] == 'buy' else (-1 if prediction['action'] == 'sell' else 0)
            weighted_action += action_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return {'action': 'hold', 'confidence': 0.5}
        
        # Determine final action
        final_score = weighted_action / total_weight
        
        if final_score > 0.1:
            final_action = 'buy'
        elif final_score < -0.1:
            final_action = 'sell'
        else:
            final_action = 'hold'
        
        return {
            'action': final_action,
            'confidence': abs(final_score),
            'weighted_score': final_score,
            'ensemble_method': 'weighted'
        }
    
    def _average_ensemble(self, predictions: Dict[str, Dict]) -> Dict:
        """Average ensemble method"""
        confidences = [pred['confidence'] for pred in predictions.values()]
        avg_confidence = np.mean(confidences)
        
        # Simple majority vote
        actions = [pred['action'] for pred in predictions.values()]
        final_action = max(set(actions), key=actions.count)
        
        return {
            'action': final_action,
            'confidence': avg_confidence,
            'num_models': len(predictions),
            'ensemble_method': 'average'
        }
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get ensemble performance summary"""
        performance = {}
        
        for model_type, strategy in self.strategies.items():
            performance[model_type] = strategy.get_model_performance()
        
        return performance


def create_ml_strategy(symbols: List[str], **kwargs) -> MLStrategy:
    """Convenience function to create ML strategy"""
    return MLStrategy(symbols, **kwargs)


def create_ensemble_ml_strategy(symbols: List[str], **kwargs) -> EnsembleMLStrategy:
    """Convenience function to create ensemble ML strategy"""
    return EnsembleMLStrategy(symbols, **kwargs) 