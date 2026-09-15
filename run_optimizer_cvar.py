import os
import pandas as pd
from engine import load_signals, run_backtest
from config import BacktestConfig
from optimizer import optimize_positions, backtest_optimized
from optimizer_cvar import optimize_positions_cvar, backtest_cvar

signals = load_signals()
cfg = BacktestConfig(label="naive")

_, naive_metrics = run_backtest(signals, cfg)

opt_df = optimize_positions(signals, cfg)
_, opt_metrics = backtest_optimized(opt_df, cfg)

cvar_df = optimize_positions_cvar(signals, cfg, alpha=0.95, cvar_limit=0.006)
_, cvar_metrics = backtest_cvar(cvar_df, cfg)

print("=== NAIVE ===")
print(naive_metrics)
print("\n=== CONVEX OPTIMIZED (Phase 2) ===")
print(opt_metrics)
print("\n=== CVaR-CONSTRAINED (Phase 3) ===")
print(cvar_metrics)

os.makedirs("outputs", exist_ok=True)
results = pd.DataFrame([naive_metrics, opt_metrics, cvar_metrics])
results.to_csv("outputs/phase3_three_way_comparison.csv", index=False)
print("\nSaved outputs/phase3_three_way_comparison.csv")