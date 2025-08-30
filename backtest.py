# backtest.py
# NO INTERNET - Robust backtesting engine for hackathon trading strategies
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from utils import assert_no_network_imports, check_dataframe_format

def backtest_minute_sharpe(df: pd.DataFrame, signals: pd.Series, 
                          starting_cash: float = 10_000_000, 
                          fee: float = 0.001) -> Dict[str, Any]:
    """
    Simulate long-only, all-in/all-out trading with minute-level Sharpe calculation.
    
    Trading Rules:
    - Signals at time t execute at next bar's open (t+1)
    - Long-only: can only BUY when in cash, SELL when holding position
    - All-in/all-out: use entire cash balance for trades
    - Fees applied on both sides: reduce cash on BUY, reduce proceeds on SELL
    - Mark-to-market at each bar's close price
    
    Args:
        df: DataFrame with OHLCV data, sorted ascending datetime index
        signals: Series aligned to df.index with values "BUY"/"SELL"/"HOLD"
        starting_cash: Initial cash balance
        fee: Fractional fee per trade side (0.001 = 0.1%)
        
    Returns:
        dict with keys:
            - nav: pd.Series of net asset value at each timestamp
            - sharpe_annualized: float, annualized Sharpe ratio using minute returns
            - final_nav: float, final portfolio value
            - trades: list of trade dictionaries
            - total_return: float, total return percentage
            - max_drawdown: float, maximum drawdown percentage
            - win_rate: float, percentage of profitable trades
            - trade_count: int, total number of trades executed
    """
    # Validation
    assert_no_network_imports()
    check_dataframe_format(df, ['open', 'high', 'low', 'close', 'volume'])
    
    if not isinstance(signals, pd.Series):
        raise ValueError("signals must be a pandas Series")
    if len(df) != len(signals):
        raise ValueError("signals length must equal dataframe length")
    
    # Ensure data is sorted and aligned
    df = df.sort_index().copy()
    signals = signals.reindex(df.index)
    
    # Initialize state
    position = 0  # 0 = cash, 1 = long position
    cash = float(starting_cash)
    shares = 0.0
    pending_action = None
    
    # Track results
    navs = []
    trades = []
    
    for i, timestamp in enumerate(df.index):
        current_bar = df.loc[timestamp]
        
        # Execute pending action from previous bar's signal at current open
        if pending_action is not None:
            action = pending_action
            open_price = float(current_bar['open'])
            
            if action == "BUY" and position == 0:
                # Buy all-in: apply fee by reducing available cash
                available_cash = cash * (1 - fee)
                shares = available_cash / open_price
                cash = 0.0
                position = 1
                trades.append({
                    'time': timestamp,
                    'type': 'BUY',
                    'price': open_price,
                    'shares': shares,
                    'fee_paid': cash * fee if cash > 0 else available_cash * fee / (1 - fee)
                })
                
            elif action == "SELL" and position == 1:
                # Sell all: apply fee by reducing proceeds
                gross_proceeds = shares * open_price
                cash = gross_proceeds * (1 - fee)
                shares = 0.0
                position = 0
                trades.append({
                    'time': timestamp,
                    'type': 'SELL', 
                    'price': open_price,
                    'shares': shares,
                    'fee_paid': gross_proceeds * fee
                })
            
            pending_action = None
        
        # Read current signal for next bar execution
        current_signal = signals.loc[timestamp]
        if current_signal in ("BUY", "SELL"):
            pending_action = current_signal
        
        # Mark-to-market at close
        close_price = float(current_bar['close'])
        nav = cash + shares * close_price
        navs.append(nav)
    
    # Create results
    nav_series = pd.Series(navs, index=df.index)
    
    # Calculate performance metrics
    returns = nav_series.pct_change().fillna(0.0)
    
    # Annualized Sharpe (assuming 365*24*60 minutes per year)
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)
    sharpe_annualized = 0.0 if std_return == 0 else (mean_return / std_return) * np.sqrt(365 * 24 * 60)
    
    # Total return
    total_return = (nav_series.iloc[-1] / starting_cash - 1) * 100
    
    # Maximum drawdown
    running_max = nav_series.expanding().max()
    drawdowns = (nav_series - running_max) / running_max * 100
    max_drawdown = drawdowns.min()
    
    # Win rate calculation
    if len(trades) >= 2:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        profitable_trades = 0
        total_completed_trades = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_completed_trades):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            if sell_price > buy_price:
                profitable_trades += 1
        
        win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0.0
    else:
        win_rate = 0.0
    
    return {
        'nav': nav_series,
        'sharpe_annualized': float(sharpe_annualized),
        'final_nav': float(nav_series.iloc[-1]),
        'trades': trades,
        'total_return': float(total_return),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'trade_count': len(trades)
    }

def print_backtest_summary(results: Dict[str, Any], strategy_name: str = "Strategy"):
    """
    Print a formatted summary of backtest results.
    
    Args:
        results: Dictionary returned by backtest_minute_sharpe()
        strategy_name: Name of strategy for display
    """
    print(f"\n=== {strategy_name} Backtest Results ===")
    print(f"Final NAV: ${results['final_nav']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annualized Sharpe: {results['sharpe_annualized']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Total Trades: {results['trade_count']}")
    
    if results['trades']:
        print(f"First Trade: {results['trades'][0]['time']} - {results['trades'][0]['type']} @ ${results['trades'][0]['price']:.2f}")
        print(f"Last Trade: {results['trades'][-1]['time']} - {results['trades'][-1]['type']} @ ${results['trades'][-1]['price']:.2f}")

# Example usage for testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=1000, freq='T')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 100
    returns = np.random.normal(0, 0.001, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
        'close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Generate simple buy-and-hold signals
    sample_signals = pd.Series(['BUY'] + ['HOLD'] * (len(dates) - 1), index=dates)
    
    # Run backtest
    results = backtest_minute_sharpe(sample_df, sample_signals)
    print_backtest_summary(results, "Sample Buy-and-Hold")
