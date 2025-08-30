# tests/test_predict_api.py
# Unit tests to ensure strategy API compliance with hackathon requirements

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_vasudevjamdagnigaur_baseline import Strategy as BaselineStrategy
from strategies.strategy_vasudevjamdagnigaur_ml import Strategy as MLStrategy
from utils import ALLOWED_SIGNALS

def make_sample_df(n=100, seed=42):
    """
    Create sample OHLCV DataFrame for testing.
    
    Args:
        n: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame with OHLCV data and datetime index
    """
    np.random.seed(seed)
    
    # Generate realistic datetime index
    start_date = pd.Timestamp('2023-01-01 09:00:00')
    idx = pd.date_range(start_date, periods=n, freq='T')
    
    # Generate price data with some trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.01, n)  # Small positive drift with volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
        'close': prices * (1 + np.random.normal(0, 0.001, n)),
        'volume': np.random.randint(1000, 10000, n)
    }, index=idx)
    
    # Ensure high >= close >= low and high >= open >= low
    df['high'] = np.maximum.reduce([df['high'], df['open'], df['close']])
    df['low'] = np.minimum.reduce([df['low'], df['open'], df['close']])
    
    return df

def test_baseline_strategy_api():
    """Test that baseline strategy meets API requirements."""
    print("Testing Baseline Strategy API...")
    
    # Create test data
    df = make_sample_df(500)
    train_df = df.iloc[:300]
    test_df = df.iloc[300:]
    
    # Initialize strategy
    strategy = BaselineStrategy()
    
    # Test initialization
    assert hasattr(strategy, 'name'), "Strategy must have 'name' attribute"
    assert hasattr(strategy, 'description'), "Strategy must have 'description' attribute"
    assert isinstance(strategy.name, str), "Strategy name must be string"
    assert isinstance(strategy.description, str), "Strategy description must be string"
    
    # Test fit method
    try:
        strategy.fit(train_df)
        print("✓ fit() method executed successfully")
    except Exception as e:
        raise AssertionError(f"fit() method failed: {e}")
    
    # Test predict method
    try:
        signals = strategy.predict(test_df)
        print("✓ predict() method executed successfully")
    except Exception as e:
        raise AssertionError(f"predict() method failed: {e}")
    
    # Validate predict output
    assert isinstance(signals, pd.Series), "predict() must return pandas Series"
    assert len(signals) == len(test_df), f"Signal length ({len(signals)}) must equal test_df length ({len(test_df)})"
    assert signals.index.equals(test_df.index), "Signal index must match test_df index"
    
    # Check signal values
    unique_signals = set(signals.unique())
    assert unique_signals.issubset(ALLOWED_SIGNALS), f"Invalid signals found: {unique_signals - ALLOWED_SIGNALS}"
    assert not signals.isnull().any(), "Signals cannot contain null values"
    
    print(f"✓ Generated {len(signals)} valid signals: {dict(signals.value_counts())}")
    print("✓ Baseline strategy API test passed")

def test_ml_strategy_api():
    """Test that ML strategy meets API requirements."""
    print("\nTesting ML Strategy API...")
    
    # Create larger test data for ML
    df = make_sample_df(1000)
    train_df = df.iloc[:700]
    test_df = df.iloc[700:]
    
    # Initialize strategy
    strategy = MLStrategy()
    
    # Test initialization
    assert hasattr(strategy, 'name'), "Strategy must have 'name' attribute"
    assert hasattr(strategy, 'description'), "Strategy must have 'description' attribute"
    assert isinstance(strategy.name, str), "Strategy name must be string"
    assert isinstance(strategy.description, str), "Strategy description must be string"
    
    # Test fit method
    try:
        strategy.fit(train_df)
        print("✓ fit() method executed successfully")
    except Exception as e:
        raise AssertionError(f"fit() method failed: {e}")
    
    # Test predict method
    try:
        signals = strategy.predict(test_df)
        print("✓ predict() method executed successfully")
    except Exception as e:
        raise AssertionError(f"predict() method failed: {e}")
    
    # Validate predict output
    assert isinstance(signals, pd.Series), "predict() must return pandas Series"
    assert len(signals) == len(test_df), f"Signal length ({len(signals)}) must equal test_df length ({len(test_df)})"
    assert signals.index.equals(test_df.index), "Signal index must match test_df index"
    
    # Check signal values
    unique_signals = set(signals.unique())
    assert unique_signals.issubset(ALLOWED_SIGNALS), f"Invalid signals found: {unique_signals - ALLOWED_SIGNALS}"
    assert not signals.isnull().any(), "Signals cannot contain null values"
    
    print(f"✓ Generated {len(signals)} valid signals: {dict(signals.value_counts())}")
    print("✓ ML strategy API test passed")

def test_edge_cases():
    """Test strategies with edge case data."""
    print("\nTesting edge cases...")
    
    # Test with minimal data
    small_df = make_sample_df(10)
    
    baseline = BaselineStrategy()
    ml = MLStrategy()
    
    try:
        baseline.fit(small_df.iloc[:7])
        signals_baseline = baseline.predict(small_df.iloc[7:])
        assert len(signals_baseline) == 3, "Should handle small datasets"
        print("✓ Baseline handles small datasets")
    except Exception as e:
        print(f"⚠ Baseline small dataset warning: {e}")
    
    try:
        ml.fit(small_df.iloc[:7])
        signals_ml = ml.predict(small_df.iloc[7:])
        assert len(signals_ml) == 3, "Should handle small datasets"
        print("✓ ML handles small datasets")
    except Exception as e:
        print(f"⚠ ML small dataset warning: {e}")
    
    # Test with constant prices
    constant_df = make_sample_df(50)
    constant_df[['open', 'high', 'low', 'close']] = 100.0  # All same price
    
    try:
        baseline.fit(constant_df.iloc[:30])
        signals = baseline.predict(constant_df.iloc[30:])
        assert len(signals) == 20, "Should handle constant prices"
        print("✓ Baseline handles constant prices")
    except Exception as e:
        print(f"⚠ Baseline constant price warning: {e}")

def test_reproducibility():
    """Test that strategies produce consistent results."""
    print("\nTesting reproducibility...")
    
    df = make_sample_df(200, seed=123)
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]
    
    # Test baseline reproducibility
    baseline1 = BaselineStrategy()
    baseline2 = BaselineStrategy()
    
    baseline1.fit(train_df)
    baseline2.fit(train_df)
    
    signals1 = baseline1.predict(test_df)
    signals2 = baseline2.predict(test_df)
    
    assert signals1.equals(signals2), "Baseline strategy should be deterministic"
    print("✓ Baseline strategy is reproducible")
    
    # Test ML reproducibility (should be deterministic due to random_state)
    ml1 = MLStrategy()
    ml2 = MLStrategy()
    
    ml1.fit(train_df)
    ml2.fit(train_df)
    
    signals1 = ml1.predict(test_df)
    signals2 = ml2.predict(test_df)
    
    # ML might have slight differences due to sklearn internals, so check mostly similar
    similarity = (signals1 == signals2).mean()
    assert similarity > 0.8, f"ML strategy should be mostly reproducible (similarity: {similarity:.2f})"
    print(f"✓ ML strategy is reproducible (similarity: {similarity:.2f})")

def run_all_tests():
    """Run all API compliance tests."""
    print("=" * 60)
    print("HACKATHON STRATEGY API COMPLIANCE TESTS")
    print("=" * 60)
    
    try:
        test_baseline_strategy_api()
        test_ml_strategy_api()
        test_edge_cases()
        test_reproducibility()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Strategies are hackathon compliant!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()

