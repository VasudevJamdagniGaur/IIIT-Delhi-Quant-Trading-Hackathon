# strategies/strategy_teamname_ml.py
# NO INTERNET - DO NOT CHANGE CLASS NAME OR METHOD SIGNATURES
# Machine Learning strategy using RandomForest for hackathon submission

import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import validate_signals, assert_no_network_imports, check_dataframe_format

# Weights file path (relative to strategy file location)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "weights_vasudevjamdagnigaur_ml.pkl")

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for ML model from OHLCV data.
    
    Features include:
    - Price-based: returns, rolling statistics, price ratios
    - Technical indicators: RSI, Bollinger Bands, MACD-like
    - Volume-based: volume ratios, volume-price relationship
    - Momentum: rate of change, acceleration
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy().sort_index()
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['ret_1'] = df['close'].pct_change().fillna(0)
    features['ret_5'] = df['close'].pct_change(5).fillna(0)
    features['ret_15'] = df['close'].pct_change(15).fillna(0)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        features[f'ret_mean_{window}'] = features['ret_1'].rolling(window, min_periods=1).mean()
        features[f'ret_std_{window}'] = features['ret_1'].rolling(window, min_periods=1).std().fillna(0)
        features[f'price_zscore_{window}'] = (
            (df['close'] - df['close'].rolling(window, min_periods=1).mean()) / 
            (df['close'].rolling(window, min_periods=1).std().fillna(1))
        ).fillna(0)
    
    # Technical indicators
    # RSI (Relative Strength Index)
    delta = df['close'].diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    for period in [14, 30]:
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    for window in [20, 40]:
        sma = df['close'].rolling(window, min_periods=1).mean()
        std = df['close'].rolling(window, min_periods=1).std()
        features[f'bb_upper_{window}'] = (df['close'] - (sma + 2 * std)) / (std + 1e-8)
        features[f'bb_lower_{window}'] = (df['close'] - (sma - 2 * std)) / (std + 1e-8)
    
    # MACD-like features
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['close']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
    features['volume_price_trend'] = (
        features['ret_1'] * np.log1p(features['volume_ratio'])
    ).fillna(0)
    
    # High-Low spread and volatility
    features['hl_spread'] = (df['high'] - df['low']) / df['close']
    features['volatility_5'] = features['hl_spread'].rolling(5, min_periods=1).mean()
    features['volatility_20'] = features['hl_spread'].rolling(20, min_periods=1).mean()
    
    # Momentum and acceleration
    features['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    features['acceleration'] = features['ret_1'] - features['ret_1'].shift(1)
    
    # Price position within recent range
    for window in [10, 30]:
        high_window = df['high'].rolling(window, min_periods=1).max()
        low_window = df['low'].rolling(window, min_periods=1).min()
        features[f'price_position_{window}'] = (
            (df['close'] - low_window) / (high_window - low_window + 1e-8)
        ).fillna(0.5)
    
    # Fill any remaining NaN values
    features = features.fillna(0)
    
    return features

class Strategy:
    """
    Machine Learning Strategy using RandomForest
    
    Approach:
    - Engineer comprehensive technical features from OHLCV data
    - Train RandomForest to predict next-minute price direction
    - Use probability thresholds to generate BUY/SELL/HOLD signals
    - Include position tracking for realistic trading simulation
    """
    
    def __init__(self):
        self.name = "vasudevjamdagnigaur - ML_RandomForest"
        self.description = "RandomForest classifier on engineered features with dynamic thresholds and position tracking."
        
        # Model and preprocessing
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Trading parameters (tuned in fit())
        self.buy_threshold = 0.55
        self.sell_threshold = 0.45
        self.confidence_threshold = 0.1  # Minimum confidence for any signal
        
        # Model parameters
        self.random_state = 42
        self.n_estimators = 200
        self.max_depth = 10
        self.min_samples_split = 20
        self.min_samples_leaf = 10
        
        # Internal state
        self._fitted = False
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train RandomForest model on historical data.
        
        Process:
        1. Engineer features from training data
        2. Create target variable (next-minute price direction)
        3. Train RandomForest with cross-validation
        4. Optimize probability thresholds
        5. Save trained model to weights file
        
        Args:
            train_df: Training DataFrame with OHLCV data
        """
        assert_no_network_imports()
        check_dataframe_format(train_df)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        print("Starting ML strategy training...")
        
        # Prepare data
        df = train_df.sort_index().copy()
        
        # Engineer features
        print("Engineering features...")
        features = feature_engineer(df)
        self.feature_names = features.columns.tolist()
        
        # Create target: predict if next minute's close > current close
        target = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Remove last row (no target available)
        features = features.iloc[:-1]
        target = target.iloc[:-1]
        
        # Remove any rows with NaN targets
        valid_idx = ~target.isna()
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        print(f"Training on {len(features)} samples with {len(self.feature_names)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features.values)
        y = target.values
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train RandomForest
        print("Training RandomForest...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        # Validate model
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation accuracy: {accuracy:.3f}")
        
        # Optimize thresholds based on validation set
        self._optimize_thresholds(y_val, y_val_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'confidence_threshold': self.confidence_threshold
        }
        
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        joblib.dump(model_data, WEIGHTS_PATH)
        print(f"Model saved to {WEIGHTS_PATH}")
        
        self._fitted = True
        
    def _optimize_thresholds(self, y_true, y_proba):
        """
        Optimize buy/sell thresholds based on validation performance.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
        """
        best_score = 0
        best_buy = 0.55
        best_sell = 0.45
        
        # Grid search over threshold combinations
        for buy_thresh in np.arange(0.5, 0.8, 0.05):
            for sell_thresh in np.arange(0.2, 0.5, 0.05):
                if buy_thresh <= sell_thresh:
                    continue
                    
                # Simulate trading signals
                signals = []
                position = 0
                
                for prob in y_proba:
                    if position == 0 and prob >= buy_thresh:
                        signals.append(1)  # BUY
                        position = 1
                    elif position == 1 and prob <= sell_thresh:
                        signals.append(-1)  # SELL
                        position = 0
                    else:
                        signals.append(0)  # HOLD
                
                # Calculate simple performance metric
                if len(signals) > 0:
                    signal_accuracy = np.mean([
                        (s == 1 and y_true[i] == 1) or 
                        (s == -1 and y_true[i] == 0) or 
                        (s == 0)
                        for i, s in enumerate(signals)
                    ])
                    
                    if signal_accuracy > best_score:
                        best_score = signal_accuracy
                        best_buy = buy_thresh
                        best_sell = sell_thresh
        
        self.buy_threshold = best_buy
        self.sell_threshold = best_sell
        
        print(f"Optimized thresholds: BUY >= {self.buy_threshold:.3f}, SELL <= {self.sell_threshold:.3f}")
        
    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using trained ML model.
        
        Args:
            test_df: Test DataFrame with OHLCV data
            
        Returns:
            pd.Series with "BUY"/"SELL"/"HOLD" signals
        """
        assert_no_network_imports()
        check_dataframe_format(test_df)
        
        df = test_df.sort_index().copy()
        
        # Load model if not already loaded
        if self.model is None:
            try:
                print(f"Loading model from {WEIGHTS_PATH}")
                model_data = joblib.load(WEIGHTS_PATH)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.buy_threshold = model_data.get('buy_threshold', 0.55)
                self.sell_threshold = model_data.get('sell_threshold', 0.45)
                self.confidence_threshold = model_data.get('confidence_threshold', 0.1)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load model ({e}). Using HOLD-only fallback.")
                return pd.Series("HOLD", index=df.index)
        
        # Engineer features
        features = feature_engineer(df)
        
        # Ensure feature consistency
        if self.feature_names and set(features.columns) != set(self.feature_names):
            print("Warning: Feature mismatch detected. Using available features.")
            # Use only common features
            common_features = list(set(features.columns) & set(self.feature_names))
            if not common_features:
                print("No common features found. Using HOLD-only fallback.")
                return pd.Series("HOLD", index=df.index)
            features = features[common_features]
        
        # Scale features
        X_scaled = self.scaler.transform(features.values)
        
        # Get predictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of price going up
        
        # Generate signals with position tracking
        position = 0  # 0 = cash, 1 = long
        signals = []
        
        for i, prob in enumerate(probabilities):
            signal = "HOLD"  # Default
            
            # Check confidence threshold
            confidence = abs(prob - 0.5)
            
            if confidence >= self.confidence_threshold:
                if position == 0:  # Currently in cash
                    if prob >= self.buy_threshold:
                        signal = "BUY"
                        position = 1
                        
                else:  # Currently holding position
                    if prob <= self.sell_threshold:
                        signal = "SELL"
                        position = 0
            
            signals.append(signal)
        
        # Create result series
        result_series = pd.Series(signals, index=df.index)
        
        # Validate output
        validate_signals(result_series, df)
        
        # Print signal summary
        signal_counts = result_series.value_counts()
        print(f"ML Signal distribution: {dict(signal_counts)}")
        
        return result_series

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=5000, freq='T')
    np.random.seed(42)
    
    # Generate more complex price data with trends and noise
    base_price = 100
    trend = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.1
    noise = np.random.normal(0, 0.02, len(dates))
    returns = trend + noise
    prices = base_price * np.exp(np.cumsum(returns))
    
    test_df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Test strategy
    strategy = Strategy()
    
    # Split data for training and testing
    train_size = int(0.7 * len(test_df))
    train_data = test_df.iloc[:train_size]
    test_data = test_df.iloc[train_size:]
    
    print("Fitting ML strategy...")
    strategy.fit(train_data)
    
    print("Generating predictions...")
    signals = strategy.predict(test_data)
    
    print(f"Generated {len(signals)} signals")
    print(f"Signal distribution: {signals.value_counts().to_dict()}")

