import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine import load_signals, run_backtest, run_sensitivity_sweep
from config import BacktestConfig
from optimizer import optimize_positions, backtest_optimized

os.makedirs("graphs", exist_ok=True)

signals = load_signals()
cfg = BacktestConfig(label="naive")

naive_df, naive_metrics = run_backtest(signals, cfg)
opt_df_raw = optimize_positions(signals, cfg)
opt_df, opt_metrics = backtest_optimized(opt_df_raw, cfg)

# ---------- 1. Equity curve comparison ----------
plt.figure(figsize=(10, 5))
plt.plot(naive_df["Date"], naive_df["Portfolio_Value"], label="Naive (Member A)", linewidth=1.5)
plt.plot(opt_df["Date"], opt_df["Portfolio_Value"], label="Convex Optimized (Member B)", linewidth=1.5)
plt.title("Portfolio Value: Naive vs Convex-Optimized")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (₹)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/phase2_equity_curve.png", dpi=150)
plt.close()
print("Saved graphs/phase2_equity_curve.png")

# ---------- 2. Metrics bar chart ----------
metrics_to_plot = ["total_return_pct", "sharpe", "sortino", "calmar"]
labels = ["Total Return %", "Sharpe", "Sortino", "Calmar"]
naive_vals = [naive_metrics[m] for m in metrics_to_plot]
opt_vals = [opt_metrics[m] for m in metrics_to_plot]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, naive_vals, width, label="Naive")
plt.bar(x + width/2, opt_vals, width, label="Convex Optimized")
plt.xticks(x, labels)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Naive vs Convex-Optimized: Key Metrics")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/phase2_metrics_comparison.png", dpi=150)
plt.close()
print("Saved graphs/phase2_metrics_comparison.png")

# ---------- 3. Cost-sensitivity heatmap ----------
cost_grid = [5, 10, 15, 20, 25]
slip_grid = [2, 5, 8, 11, 14]

heat = np.zeros((len(cost_grid), len(slip_grid)))
for i, c in enumerate(cost_grid):
    for j, s in enumerate(slip_grid):
        cfg_i = BacktestConfig(cost_bps=c, slippage_bps=s, label=f"c{c}_s{s}")
        opt_df_i = optimize_positions(signals, cfg_i)
        _, m_i = backtest_optimized(opt_df_i, cfg_i)
        heat[i, j] = m_i["total_return_pct"]

plt.figure(figsize=(8, 6))
im = plt.imshow(heat, cmap="RdYlGn", aspect="auto")
plt.colorbar(im, label="Total Return %")
plt.xticks(range(len(slip_grid)), slip_grid)
plt.yticks(range(len(cost_grid)), cost_grid)
plt.xlabel("Slippage (bps)")
plt.ylabel("Transaction Cost (bps)")
plt.title("Convex Optimizer: Return Sensitivity to Cost Assumptions")

for i in range(len(cost_grid)):
    for j in range(len(slip_grid)):
        plt.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("graphs/phase2_cost_sensitivity_heatmap.png", dpi=150)
plt.close()
print("Saved graphs/phase2_cost_sensitivity_heatmap.png")