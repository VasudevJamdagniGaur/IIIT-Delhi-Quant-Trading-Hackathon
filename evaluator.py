# evaluator.py
# Evaluate all strategies matching strategy_*.py, rank by annualized Sharpe.
# Long-only, all-in/all-out, T+1 open execution, fee in bps per side.

import argparse, json, math, sys, importlib.util, traceback
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------
# Data utilities
# -------------------------
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError(f"{csv_path}: missing 'timestamp' column")

    # Normalize timestamp to UTC (handles ISO8601 and ms epoch)
    try:
        if np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    except Exception:
        # If parse fails (e.g., already tz-aware strings), try generic:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    expected = ['timestamp','open','high','low','close','volume']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    # Coerce numerics
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# -------------------------
# Strategy loader (FIXED)
# -------------------------
def import_strategy(path: Path):
    """
    Import a strategy module from file and return an instance of Strategy.
    IMPORTANT FIX: register module in sys.modules before exec_module so
    decorators (e.g., @dataclass) can resolve the module correctly.
    """
    name = path.stem  # e.g. strategy_team_strategy
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)

    # >>> Critical line: insert into sys.modules before executing
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "Strategy"):
        raise ValueError(f"{path.name}: must define class `Strategy`")
    return mod.Strategy()


# -------------------------
# Backtester (long-only + T+1 execution enforced)
# -------------------------
def backtest(test_df: pd.DataFrame, signals: pd.Series, fee_bps: float,
             initial_cash: float = 10_000_000.0, latency_bars: int = 1):
    """
    IMPORTANT: This evaluator uses Open[t+1] execution (signal at t executes at next bar's open)
    This is DIFFERENT from your backtest.py which uses Close[t] execution.
    """
    assert len(test_df) == len(signals), "signals length must match test_df length"

    fee_rate = fee_bps / 10_000.0
    cash = initial_cash
    btc  = 0.0
    position = "FLAT"  # or "LONG"

    opens = test_df['open'].to_numpy(dtype=float)
    closes = test_df['close'].to_numpy(dtype=float)
    times = test_df['timestamp'].astype(str).to_numpy()
    sig = signals.astype(str).str.upper().to_numpy()

    equity_rows = []
    trades = []

    for i in range(len(test_df)):
        # mark-to-market at close
        equity_rows.append({"timestamp": times[i], "equity": cash + btc * closes[i]})

        exec_i = i + latency_bars
        if exec_i >= len(test_df):
            continue  # cannot execute beyond end

        px = float(opens[exec_i])

        if sig[i] == "SELL" and position == "FLAT":
            raise RuntimeError("Shorting not allowed: SELL while FLAT")
        if sig[i] == "BUY" and position == "LONG":
            raise RuntimeError("Pyramiding not allowed: BUY while LONG")

        # Long-only, all-in/all-out state machine:
        if sig[i] == "BUY" and position == "FLAT":
            if cash > 0:
                fee = cash * fee_rate
                btc = (cash - fee) / max(px, 1e-12)  # BTC after paying fee
                trades.append({
                    "timestamp": str(times[exec_i]),
                    "side": "BUY",
                    "price": px,
                    "fee": float(fee),
                    "btc_after": float(btc),
                    "cash_after": 0.0
                })
                cash = 0.0
                position = "LONG"

        elif sig[i] == "SELL" and position == "LONG":
            if btc > 0:
                gross = btc * px
                fee = gross * fee_rate
                proceeds = gross - fee
                cash = proceeds
                trades.append({
                    "timestamp": str(times[exec_i]),
                    "side": "SELL",
                    "price": px,
                    "fee": float(fee),
                    "btc_after": 0.0,
                    "cash_after": float(cash)
                })
                btc = 0.0
                position = "FLAT"
        # HOLD or invalid transitions do nothing (no shorting, no pyramiding)

    equity_df = pd.DataFrame(equity_rows)

    # Minute returns from equity curve
    eq = equity_df['equity'].to_numpy(dtype=float)
    ret = np.zeros_like(eq)
    if len(eq) > 1:
        prev = np.maximum(eq[:-1], 1e-12)
        ret[1:] = (eq[1:] - eq[:-1]) / prev

    # Metrics
    total_return = (eq[-1] / eq[0]) - 1.0 if len(eq) > 1 else 0.0
    minutes_per_year = 525_600
    mean_r = np.mean(ret[1:]) if len(ret) > 1 else 0.0
    std_r  = np.std(ret[1:], ddof=1) if len(ret) > 2 else 0.0
    sharpe = (mean_r / std_r) * math.sqrt(minutes_per_year) if std_r > 0 else 0.0

    # Max drawdown
    if len(eq):
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / np.where(roll_max == 0, 1, roll_max)
        max_dd = float(dd.min())
    else:
        max_dd = 0.0

    # Trade diagnostics (closed round-trips approx by BUY->SELL count)
    wins = losses = closed = 0
    pnl_trades = []
    last_buy_price = None
    last_size = 0.0
    for t in trades:
        if t["side"] == "BUY":
            last_buy_price = t["price"]
            last_size = t["btc_after"]
        elif t["side"] == "SELL" and last_buy_price is not None and last_size > 0:
            r = (t["price"] / last_buy_price) - 1.0  # price delta proxy; fees already applied
            pnl_trades.append(r)
            wins += int(r > 0)
            losses += int(r <= 0)
            closed += 1
            last_buy_price = None
            last_size = 0.0

    win_rate = (wins / closed) if closed else 0.0
    avg_trade_return = float(np.mean(pnl_trades)) if pnl_trades else 0.0
    num_trades = len(trades)

    report = {
        "initial_cash": float(initial_cash),
        "final_equity": float(eq[-1]) if len(eq) else float(initial_cash),
        "total_return": float(total_return),
        "sharpe_minute_annualized": float(sharpe),
        "max_drawdown": float(max_dd),
        "num_trades": int(num_trades),
        "closed_trades": int(closed),
        "win_rate": float(win_rate),
        "avg_trade_return_roundtrip": float(avg_trade_return),
        "fee_bps_per_side": float(fee_bps),
        "latency_bars": int(latency_bars),
        "position_model": "long_only_all_in_all_out",
        "execution": "signal_at_t_executes_at_t+1_open"
    }

    return equity_df, pd.DataFrame(trades), report

# -------------------------
# Run one strategy file
# -------------------------
def eval_one(strategy_path: Path, train_df: pd.DataFrame, test_df: pd.DataFrame,
            outdir: Path, fee_bps: float, initial_cash: float, latency_bars: int):
    name = strategy_path.stem
    s_out = outdir / name
    s_out.mkdir(parents=True, exist_ok=True)

    try:
        strategy = import_strategy(strategy_path)
        # Fit on train data, then predict on test data
        strategy.fit(train_df.copy())
        signals = strategy.predict(test_df.copy())

        # Validate signals
        if not isinstance(signals, pd.Series):
            raise TypeError("predict() must return a pandas Series")
        if len(signals) != len(test_df):
            raise ValueError("signals length must match test_df length")
        allowed = {"BUY", "SELL", "HOLD"}
        uniq = set(pd.Series(signals).astype(str).str.upper().unique().tolist())
        if not uniq.issubset(allowed):
            raise ValueError(f"signals must be in {allowed}, found {uniq}")
        signals = pd.Series(signals).astype(str).str.upper()

        # Backtest with Open[t+1] execution
        equity_df, trades_df, report = backtest(
            test_df, signals, fee_bps=fee_bps,
            initial_cash=initial_cash, latency_bars=latency_bars
        )

        # Save outputs
        equity_df.to_csv(s_out / "equity_curve.csv", index=False)
        trades_df.to_csv(s_out / "trades.csv", index=False)
        with open(s_out / "report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Console summary
        print(f"\n=== {name} ===")
        print(f"Sharpe (annualized): {report['sharpe_minute_annualized']:.4f}")
        print(f"Total Return       : {report['total_return']*100:.2f}%")
        print(f"Max Drawdown       : {report['max_drawdown']*100:.2f}%")
        print(f"# Trades           : {report['num_trades']}, Win rate: {report['win_rate']*100:.2f}%")
        return {
            "strategy_file": name,
            "sharpe_minute_annualized": report["sharpe_minute_annualized"],
            "total_return": report["total_return"],
            "max_drawdown": report["max_drawdown"],
            "num_trades": report["num_trades"],
            "win_rate": report["win_rate"],
            "outdir": str(s_out)
        }

    except Exception as e:
        err_path = s_out / "error.txt"
        with open(err_path, "w") as f:
            f.write("Evaluation failed:\n")
            f.write("".join(traceback.format_exception(e)))
        print(f"\n=== {name} ===")
        print("FAILED (see error.txt)")
        return {
            "strategy_file": name,
            "sharpe_minute_annualized": float("-inf"),
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "num_trades": 0,
            "win_rate": float("nan"),
            "outdir": str(s_out)
        }

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate all strategy_*.py and rank by annualized Sharpe.")
    ap.add_argument("--train", required=True, help="Path to train.csv")
    ap.add_argument("--test", required=True, help="Path to test.csv")
    ap.add_argument("--strategies-dir", default="strategies", help="Directory containing strategy_*.py")
    ap.add_argument("--pattern", default="strategy_*.py", help="Glob pattern for strategy files")
    ap.add_argument("--fee-bps", type=float, default=10.0, help="Fee in basis points per side (default 10 = 0.10%)")
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0, help="Starting capital")
    ap.add_argument("--latency-bars", type=int, default=1, help="Execution delay in bars after a signal")
    ap.add_argument("--outdir", default="eval_outputs", help="Directory to write outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = load_df(args.train)
    test_df  = load_df(args.test)

    strat_dir = Path(args.strategies_dir)
    strat_paths = sorted(strat_dir.glob(args.pattern))
    if not strat_paths:
        print(f"No strategies found in {strat_dir!r} matching pattern {args.pattern!r}")
        sys.exit(1)

    print(f"Found {len(strat_paths)} strategies:")
    for p in strat_paths:
        print(" -", p.name)

    rows = []
    for p in strat_paths:
        rows.append(
            eval_one(p, train_df, test_df, outdir, args.fee_bps, args.initial_cash, args.latency_bars)
        )

    # Leaderboard by Sharpe
    lb = pd.DataFrame(rows)
    lb = lb.sort_values("sharpe_minute_annualized", ascending=False).reset_index(drop=True)
    lb_path = outdir / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    print("\n================ LEADERBOARD (by Sharpe, annualized) ================")
    for i, r in lb.iterrows():
        print(f"{i+1:2d}. {r['strategy_file']:40s}  Sharpe={r['sharpe_minute_annualized']:.4f}")
    print("=====================================================================")
    print(f"\nSaved leaderboard to: {lb_path}\nOutputs per strategy under: {outdir}/<strategy_name>/")

if __name__ == "__main__":
    main()
