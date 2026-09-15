import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine import load_signals, run_backtest
from config import BacktestConfig
from optimizer import optimize_positions, backtest_optimized
from optimizer_cvar import optimize_positions_cvar, backtest_cvar

os.makedirs("graphs", exist_ok=True)

signals = load_signals()
cfg = BacktestConfig(label="naive")

naive_df, naive_metrics = run_backtest(signals, cfg)

opt_df_raw = optimize_positions(signals, cfg)
opt_df, opt_metrics = backtest_optimized(opt_df_raw, cfg)

cvar_df_raw = optimize_positions_cvar(signals, cfg, alpha=0.95, cvar_limit=0.006)
cvar_df, cvar_metrics = backtest_cvar(cvar_df_raw, cfg)

# ---------- 1. Equity curve comparison (three-way) ----------
plt.figure(figsize=(11, 5))
plt.plot(naive_df["Date"], naive_df["Portfolio_Value"], label="Naive (Member A)", linewidth=1.3)
plt.plot(opt_df["Date"], opt_df["Portfolio_Value"], label="Convex Optimized (Phase 2)", linewidth=1.3)
plt.plot(cvar_df["Date"], cvar_df["Portfolio_Value"], label="CVaR-Constrained (Phase 3)", linewidth=1.8)
plt.title("Portfolio Value: Naive vs Convex vs CVaR-Constrained")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (₹)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/phase3_equity_curve.png", dpi=150)
plt.close()
print("Saved graphs/phase3_equity_curve.png")

# ---------- 2. Metrics bar chart (three-way) ----------
metrics_to_plot = ["total_return_pct", "sharpe", "sortino", "calmar", "max_drawdown"]
labels = ["Total Return %", "Sharpe", "Sortino", "Calmar", "Max Drawdown"]

naive_vals = [naive_metrics[m] for m in metrics_to_plot]
opt_vals = [opt_metrics[m] for m in metrics_to_plot]
cvar_vals = [cvar_metrics[m] for m in metrics_to_plot]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(11, 5.5))
plt.bar(x - width, naive_vals, width, label="Naive")
plt.bar(x, opt_vals, width, label="Convex Optimized (Phase 2)")
plt.bar(x + width, cvar_vals, width, label="CVaR-Constrained (Phase 3)")
plt.xticks(x, labels)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Naive vs Convex vs CVaR-Constrained: Key Metrics")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/phase3_metrics_comparison.png", dpi=150)
plt.close()
print("Saved graphs/phase3_metrics_comparison.png")

# ---------- 3. Drawdown-over-time comparison ----------
def compute_drawdown_series(portfolio_value):
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    return drawdown

naive_dd = compute_drawdown_series(naive_df["Portfolio_Value"])
opt_dd = compute_drawdown_series(opt_df["Portfolio_Value"])
cvar_dd = compute_drawdown_series(cvar_df["Portfolio_Value"])

plt.figure(figsize=(11, 5))
plt.plot(naive_df["Date"], naive_dd * 100, label="Naive", linewidth=1.2)
plt.plot(opt_df["Date"], opt_dd * 100, label="Convex Optimized (Phase 2)", linewidth=1.2)
plt.plot(cvar_df["Date"], cvar_dd * 100, label="CVaR-Constrained (Phase 3)", linewidth=1.8)
plt.title("Drawdown Over Time: Naive vs Convex vs CVaR-Constrained")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/phase3_drawdown_comparison.png", dpi=150)
plt.close()
print("Saved graphs/phase3_drawdown_comparison.png")