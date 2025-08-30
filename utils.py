# utils.py
# NO INTERNET - Utility functions for hackathon trading strategies
import sys
import pandas as pd

ALLOWED_SIGNALS = {"BUY", "SELL", "HOLD"}

def validate_signals(signals, df):
    """
    Validate that signals Series meets hackathon requirements:
    - Same length and index as df
    - Only contains allowed signal values
    - No null values
    
    Args:
        signals: pd.Series with signal values
        df: pd.DataFrame to validate against
        
    Raises:
        ValueError: If validation fails
        
    Returns:
        bool: True if validation passes
    """
    if not isinstance(signals, pd.Series):
        raise ValueError("signals must be a pandas Series")
    if len(signals) != len(df):
        raise ValueError(f"signals length ({len(signals)}) must equal dataframe length ({len(df)})")
    if not signals.index.equals(df.index):
        raise ValueError("signals index must equal dataframe index")
    if signals.isnull().any():
        raise ValueError("signals contain null values")
    
    invalid = set(signals.unique()) - ALLOWED_SIGNALS
    if invalid:
        raise ValueError(f"signals contain invalid labels: {invalid}. Must be one of {ALLOWED_SIGNALS}")
    
    return True

def assert_no_network_imports():
    """
    Check that no network-related modules are imported.
    Raises RuntimeError if banned modules are found.
    
    This ensures compliance with hackathon rules: no internet calls.
    """
    banned_modules = [
        'requests', 'urllib', 'urllib2', 'urllib3', 'http.client', 
        'httplib', 'socket', 'websocket', 'aiohttp', 'httpx'
    ]
    
    imported_banned = []
    for mod in banned_modules:
        if mod in sys.modules:
            imported_banned.append(mod)
    
    if imported_banned:
        raise RuntimeError(
            f"Network modules imported: {imported_banned} - "
            "Remove all network calls to comply with hackathon rules"
        )

def check_dataframe_format(df, required_columns=None):
    """
    Validate that DataFrame has expected format for trading data.
    
    Args:
        df: pd.DataFrame to validate
        required_columns: list of required column names
        
    Returns:
        bool: True if format is valid
        
    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Check for ascending order
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in ascending order")
    
    return True

def safe_divide(numerator, denominator, fill_value=0.0):
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: numeric value or array
        denominator: numeric value or array  
        fill_value: value to use when denominator is zero
        
    Returns:
        Result of division with safe handling of zero denominators
    """
    if hasattr(denominator, '__iter__'):
        # Handle array-like denominators
        result = numerator / denominator
        if hasattr(result, 'fillna'):
            return result.fillna(fill_value)
        else:
            return result
    else:
        # Handle scalar denominators
        return fill_value if denominator == 0 else numerator / denominator

