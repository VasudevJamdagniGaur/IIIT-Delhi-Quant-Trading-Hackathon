# prepare_data.py
# Convert your train.csv to the format expected by evaluator.py and split into train/test

import pandas as pd
import argparse
from pathlib import Path

def prepare_data(input_csv: str, train_ratio: float = 0.7):
    """
    Convert your train.csv to evaluator format and split into train/test files.
    
    Expected input format: datetime index with OHLCV columns
    Expected output format: timestamp column (not index) with OHLCV columns
    """
    print(f"Loading data from {input_csv}...")
    
    # Load your existing train.csv (with datetime index)
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert index to timestamp column (evaluator expects timestamp as column, not index)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'timestamp'})
    
    # Ensure timestamp is in correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Split into train/test
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Save train and test files
    train_path = "data/train_eval.csv"
    test_path = "data/test_eval.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Test:  {test_path} ({len(test_df)} rows)")
    print(f"\nNow you can run:")
    print(f"  python evaluator.py --train {train_path} --test {test_path}")
    
    return train_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for evaluator.py")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV file")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction for training (default 0.7)")
    
    args = parser.parse_args()
    
    prepare_data(args.input, args.train_ratio)
