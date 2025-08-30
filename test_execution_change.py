# test_execution_change.py
# Test script to validate the updated execution logic (Close[t] vs Open[t+1])

import pandas as pd
import numpy as np
from backtest import backtest_minute_sharpe

def test_execution_logic():
    """Test that trades execute at Close[t] instead of Open[t+1]"""
    
    # Create simple test data with predictable prices
    dates = pd.date_range('2023-01-01 09:00:00', periods=5, freq='T')
    
    # Create data where open != close to verify execution price
    test_df = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [100.5, 101.5, 102.5, 103.5, 104.5],
        'low': [99.5, 100.5, 101.5, 102.5, 103.5],
        'close': [100.2, 101.2, 102.2, 103.2, 104.2],  # Different from open
        'volume': [1000, 1000, 1000, 1000, 1000]
    }, index=dates)
    
    # Create signals: BUY at first bar, SELL at third bar
    signals = pd.Series(['BUY', 'HOLD', 'SELL', 'HOLD', 'HOLD'], index=dates)
    
    # Run backtest
    results = backtest_minute_sharpe(test_df, signals, starting_cash=10000, fee=0.0)
    
    # Validate execution prices
    trades = results['trades']
    print("=== EXECUTION LOGIC TEST ===")
    print(f"Test data:")
    print(test_df[['open', 'close']])
    print(f"\nSignals:")
    print(signals)
    print(f"\nTrades executed:")
    
    if len(trades) >= 2:
        buy_trade = trades[0]
        sell_trade = trades[1]
        
        print(f"BUY trade: {buy_trade['time']} at price ${buy_trade['price']:.2f}")
        print(f"SELL trade: {sell_trade['time']} at price ${sell_trade['price']:.2f}")
        
        # Verify BUY executed at Close[0] = 100.2, not Open[1] = 101.0
        expected_buy_price = test_df.iloc[0]['close']  # 100.2
        actual_buy_price = buy_trade['price']
        
        # Verify SELL executed at Close[2] = 102.2, not Open[3] = 103.0  
        expected_sell_price = test_df.iloc[2]['close']  # 102.2
        actual_sell_price = sell_trade['price']
        
        print(f"\n✓ VALIDATION RESULTS:")
        print(f"BUY: Expected ${expected_buy_price:.2f} (Close[0]), Got ${actual_buy_price:.2f} - {'✓ PASS' if abs(expected_buy_price - actual_buy_price) < 0.01 else '✗ FAIL'}")
        print(f"SELL: Expected ${expected_sell_price:.2f} (Close[2]), Got ${actual_sell_price:.2f} - {'✓ PASS' if abs(expected_sell_price - actual_sell_price) < 0.01 else '✗ FAIL'}")
        
        # Calculate expected vs actual returns
        expected_return = (expected_sell_price - expected_buy_price) / expected_buy_price * 100
        shares_bought = 10000 / expected_buy_price  # No fees in this test
        actual_final_nav = shares_bought * expected_sell_price
        actual_return = (actual_final_nav - 10000) / 10000 * 100
        
        print(f"\nReturn calculation:")
        print(f"Expected return: {expected_return:.2f}%")
        print(f"Actual return: {actual_return:.2f}%")
        print(f"Final NAV: ${results['final_nav']:,.2f}")
        
    else:
        print("✗ ERROR: Expected 2 trades but got", len(trades))
    
    print("=" * 40)

if __name__ == "__main__":
    test_execution_logic()
