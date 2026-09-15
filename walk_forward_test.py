import os
import pandas as pd
from engine import load_signals, run_backtest
from config import BacktestConfig
from optimizer import optimize_positions, backtest_optimized
from optimizer_cvar import optimize_positions_cvar, backtest_cvar
from metrics import compute_all_metrics

WINDOW_SIZE = 60  # trading days per window

signals = load_signals()
signals = signals.reset_index(drop=True)

n_days = len(signals)
n_windows = n_days // WINDOW_SIZE

print(f"Total days: {n_days}, Window size: {WINDOW_SIZE}, Number of windows: {n_windows}")

cfg = BacktestConfig(label="walk_forward")

all_results = []

for w in range(n_windows):
    start = w * WINDOW_SIZE
    end = min(start + WINDOW_SIZE, n_days)
    window_signals = signals.iloc[start:end].reset_index(drop=True)

    if len(window_signals) < 20:  # skip too-small trailing windows
        continue

    window_label = f"window_{w+1}"
    start_date = window_signals["Date"].iloc[0]
    end_date = window_signals["Date"].iloc[-1]
    print(f"\n{window_label}: {start_date.date()} to {end_date.date()} ({len(window_signals)} days)")

    # Naive
    _, naive_metrics = run_backtest(window_signals, cfg)
    naive_metrics.update({"strategy": "naive", "window": window_label,
                           "start_date": start_date, "end_date": end_date})
    all_results.append(naive_metrics)

    # Convex optimized
    opt_df = optimize_positions(window_signals, cfg)
    _, opt_metrics = backtest_optimized(opt_df, cfg)
    opt_metrics.update({"strategy": "convex_optimized", "window": window_label,
                         "start_date": start_date, "end_date": end_date})
    all_results.append(opt_metrics)

    # CVaR-constrained
    cvar_df = optimize_positions_cvar(window_signals, cfg, alpha=0.95, cvar_limit=0.006)
    _, cvar_metrics = backtest_cvar(cvar_df, cfg)
    cvar_metrics.update({"strategy": "cvar_constrained", "window": window_label,
                          "start_date": start_date, "end_date": end_date})
    all_results.append(cvar_metrics)

results_df = pd.DataFrame(all_results)

os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/walk_forward_results.csv", index=False)
print("\nSaved outputs/walk_forward_results.csv")

# Summary: how consistent is Sharpe across windows, per strategy?
print("\n=== Sharpe consistency across windows ===")
summary = results_df.groupby("strategy")["sharpe"].agg(["mean", "std", "min", "max"])
print(summary)
summary.to_csv("outputs/walk_forward_sharpe_summary.csv")
print("\nSaved outputs/walk_forward_sharpe_summary.csv")