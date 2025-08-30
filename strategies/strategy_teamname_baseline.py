# strategies/strategy_teamname_baseline.py
# NO INTERNET - DO NOT CHANGE CLASS NAME OR METHOD SIGNATURES
# Rule-based SMA crossover strategy for hackathon submission

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import validate_signals, assert_no_network_imports, check_dataframe_format

class Strategy:
    """
    Simple Moving Average (SMA) Crossover Strategy
    
    Logic:
    - Calculate short-term and long-term SMAs of closing prices
    - Generate BUY signal when short SMA crosses above long SMA (and currently in cash)
    - Generate SELL signal when short SMA crosses below long SMA (and currently holding position)
    - Otherwise generate HOLD signal
    
    This is a trend-following strategy that aims to capture momentum.
    """
    
    def __init__(self):
        self.name = "vasudevjamdagnigaur - SMA_Crossover_Baseline"
        self.description = "Short/long SMA crossover with adaptive windows; long-only all-in/all-out trend following."
        
        # Default parameters (will be tuned in fit())
        self.short_window = 10
        self.long_window = 60
        self.min_trend_strength = 0.001  # Minimum price change to confirm trend
        
        # Internal state
        self._fitted = False
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Tune parameters using training data.
        
        Strategy:
        - Adapt SMA windows based on data length and volatility
        - Longer datasets get longer windows for stability
        - Higher volatility gets shorter windows for responsiveness
        
        Args:
            train_df: Training DataFrame with OHLCV data
        """
        assert_no_network_imports()
        check_dataframe_format(train_df)
        
        # Ensure data is sorted
        train_df = train_df.sort_index()
        
        # Adaptive window sizing based on data characteristics
        n_minutes = len(train_df)
        
        # Calculate price volatility to inform window selection
        returns = train_df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(60)  # Hourly volatility
        
        # Base window selection on dataset size
        if n_minutes > 50000:  # Very long dataset (>34 days)
            base_short, base_long = 30, 180
        elif n_minutes > 20000:  # Long dataset (>13 days)
            base_short, base_long = 20, 120
        elif n_minutes > 5000:   # Medium dataset (>3 days)
            base_short, base_long = 10, 60
        else:                    # Short dataset
            base_short, base_long = 5, 30
            
        # Adjust for volatility - higher volatility needs shorter windows
        if volatility > 0.02:  # High volatility
            vol_factor = 0.7
        elif volatility < 0.005:  # Low volatility
            vol_factor = 1.3
        else:  # Normal volatility
            vol_factor = 1.0
            
        self.short_window = max(3, int(base_short * vol_factor))
        self.long_window = max(self.short_window * 2, int(base_long * vol_factor))
        
        # Set minimum trend strength based on typical price movements
        price_changes = train_df['close'].diff().abs()
        self.min_trend_strength = price_changes.quantile(0.6)  # 60th percentile of price changes
        
        self._fitted = True
        
        print(f"SMA Strategy fitted with windows: short={self.short_window}, long={self.long_window}")
        print(f"Minimum trend strength: {self.min_trend_strength:.4f}")
        
    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for test data.
        
        Args:
            test_df: Test DataFrame with OHLCV data
            
        Returns:
            pd.Series with "BUY"/"SELL"/"HOLD" signals aligned to test_df.index
        """
        assert_no_network_imports()
        check_dataframe_format(test_df)
        
        if not self._fitted:
            # Use default parameters if not fitted
            print("Warning: Strategy not fitted, using default parameters")
        
        df = test_df.sort_index().copy()
        
        # Calculate SMAs with minimum periods to handle edge cases
        df['sma_short'] = df['close'].rolling(
            window=self.short_window, 
            min_periods=max(1, self.short_window // 2)
        ).mean()
        
        df['sma_long'] = df['close'].rolling(
            window=self.long_window,
            min_periods=max(1, self.long_window // 2)
        ).mean()
        
        # Calculate trend signals
        df['trend_up'] = df['sma_short'] > df['sma_long']
        df['trend_change'] = df['trend_up'] != df['trend_up'].shift(1)
        
        # Add trend strength filter
        df['price_momentum'] = df['close'].diff().abs()
        df['strong_trend'] = df['price_momentum'] >= self.min_trend_strength
        
        # Generate signals with position tracking
        position = 0  # 0 = cash, 1 = long
        signals = []
        
        for idx, row in df.iterrows():
            trend_up = row['trend_up']
            trend_change = row['trend_change']
            strong_trend = row['strong_trend']
            
            signal = "HOLD"  # Default
            
            if position == 0:  # Currently in cash
                # Buy when trend turns up with sufficient strength
                if trend_up and (trend_change or strong_trend):
                    signal = "BUY"
                    position = 1
                    
            else:  # Currently holding position
                # Sell when trend turns down or weakens significantly
                if not trend_up and (trend_change or strong_trend):
                    signal = "SELL"
                    position = 0
                    
            signals.append(signal)
        
        # Create result series
        result_series = pd.Series(signals, index=df.index)
        
        # Validate output meets hackathon requirements
        validate_signals(result_series, df)
        
        # Print signal summary for debugging
        signal_counts = result_series.value_counts()
        print(f"Signal distribution: {dict(signal_counts)}")
        
        return result_series

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=2000, freq='T')
    np.random.seed(42)
    
    # Generate trending price data
    base_price = 100
    trend = np.linspace(0, 0.1, len(dates))  # Upward trend
    noise = np.random.normal(0, 0.01, len(dates))
    returns = trend + noise
    prices = base_price * np.exp(np.cumsum(returns))
    
    test_df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)
    
    # Test strategy
    strategy = Strategy()
    strategy.fit(test_df.iloc[:1500])  # Fit on first 1500 bars
    signals = strategy.predict(test_df.iloc[1500:])  # Predict on last 500 bars
    
    print(f"Generated {len(signals)} signals")
    print(f"Signal distribution: {signals.value_counts().to_dict()}")

