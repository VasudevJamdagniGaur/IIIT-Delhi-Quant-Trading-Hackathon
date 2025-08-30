# run_evaluator_example.py
# Example script showing how to use the evaluator engine

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create sample data if train.csv doesn't exist"""
    if not os.path.exists("data/train.csv"):
        print("Creating sample data for demonstration...")
        
        # Generate sample data
        dates = pd.date_range('2023-01-01 09:00:00', periods=10000, freq='T')
        np.random.seed(42)
        
        base_price = 100.0
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices * (1 + np.random.normal(0, 0.002, len(dates))),
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        df['high'] = np.maximum.reduce([df['high'], df['open'], df['close']])
        df['low'] = np.minimum.reduce([df['low'], df['open'], df['close']])
        
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/train.csv")
        print("✅ Sample data created at data/train.csv")

def run_evaluator_demo():
    """Run the complete evaluator workflow"""
    
    print("🚀 EVALUATOR ENGINE DEMO")
    print("=" * 50)
    
    # Step 1: Ensure we have sample data
    create_sample_data()
    
    # Step 2: Prepare data for evaluator format
    print("\n📊 Step 1: Preparing data...")
    try:
        os.system("python prepare_data.py --input data/train.csv --train-ratio 0.7")
        print("✅ Data prepared successfully")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    # Step 3: Run evaluator
    print("\n🏆 Step 2: Running evaluator...")
    print("This will:")
    print("  - Load both strategies")
    print("  - Fit on train data")  
    print("  - Generate signals on test data")
    print("  - Backtest with Open[t+1] execution")
    print("  - Rank by Sharpe ratio")
    print()
    
    try:
        os.system("python evaluator.py --train data/train_eval.csv --test data/test_eval.csv --strategies-dir strategies")
        print("\n✅ Evaluation completed!")
    except Exception as e:
        print(f"❌ Error running evaluator: {e}")
        return
    
    # Step 4: Show results
    print("\n📈 Step 3: Results summary...")
    
    if os.path.exists("eval_outputs/leaderboard.csv"):
        leaderboard = pd.read_csv("eval_outputs/leaderboard.csv")
        print("\n🏆 LEADERBOARD:")
        for i, row in leaderboard.iterrows():
            print(f"  {i+1}. {row['strategy_file']} - Sharpe: {row['sharpe_minute_annualized']:.4f}")
        
        print(f"\n📁 Detailed results saved in: eval_outputs/")
        print("  - leaderboard.csv: Overall rankings")
        print("  - strategy_*/equity_curve.csv: NAV over time")
        print("  - strategy_*/trades.csv: Individual trades")
        print("  - strategy_*/report.json: Detailed metrics")
    else:
        print("❌ No leaderboard found - check for errors above")

if __name__ == "__main__":
    run_evaluator_demo()
