import os
import pandas as pd
from engine import load_signals, run_backtest
from config import BacktestConfig
from optimizer import optimize_positions, backtest_optimized

signals = load_signals()
cfg = BacktestConfig(label="naive")

_, naive_metrics = run_backtest(signals, cfg)
opt_df = optimize_positions(signals, cfg)
_, opt_metrics = backtest_optimized(opt_df, cfg)

print("=== NAIVE (Member A baseline) ===")
print(naive_metrics)
print("\n=== CONVEX OPTIMIZED (Member B) ===")
print(opt_metrics)

os.makedirs("outputs", exist_ok=True)

results = pd.DataFrame([naive_metrics, opt_metrics])
results.to_csv("outputs/phase2_naive_vs_optimized.csv", index=False)
print("\nSaved results to outputs/phase2_naive_vs_optimized.csv")