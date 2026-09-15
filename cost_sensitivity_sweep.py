import os
import pandas as pd
from engine import load_signals, run_backtest
from config import BacktestConfig
from optimizer import optimize_positions, backtest_optimized
from optimizer_cvar import optimize_positions_cvar, backtest_cvar
from reporting import plot_sensitivity_heatmap

COST_BPS_GRID = [5, 10, 15, 20, 25]
SLIPPAGE_BPS_GRID = [5, 10, 15, 20, 25]

signals = load_signals()

all_results = []

for cost_bps in COST_BPS_GRID:
    for slippage_bps in SLIPPAGE_BPS_GRID:
        cfg = BacktestConfig(
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            label=f"cost{cost_bps}_slip{slippage_bps}"
        )

        _, naive_metrics = run_backtest(signals, cfg)
        naive_metrics["strategy"] = "naive"
        all_results.append(naive_metrics)

        opt_df = optimize_positions(signals, cfg)
        _, opt_metrics = backtest_optimized(opt_df, cfg)
        opt_metrics["strategy"] = "convex_optimized"
        all_results.append(opt_metrics)

        cvar_df = optimize_positions_cvar(signals, cfg, alpha=0.95, cvar_limit=0.006)
        _, cvar_metrics = backtest_cvar(cvar_df, cfg)
        cvar_metrics["strategy"] = "cvar_constrained"
        all_results.append(cvar_metrics)

        print(f"Done: cost={cost_bps}bps, slippage={slippage_bps}bps")

sweep_df = pd.DataFrame(all_results)

os.makedirs("outputs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

sweep_df.to_csv("outputs/cost_sensitivity_sweep_full.csv", index=False)
print("\nSaved outputs/cost_sensitivity_sweep_full.csv")

for strategy in ["naive", "convex_optimized", "cvar_constrained"]:
    strategy_df = sweep_df[sweep_df["strategy"] == strategy]
    plot_sensitivity_heatmap(
        strategy_df,
        metric_col="sharpe",
        out_path=f"graphs/heatmap_sharpe_{strategy}.png",
        title=f"Sharpe Ratio Sensitivity — {strategy.replace('_', ' ').title()}"
    )
    print(f"Saved graphs/heatmap_sharpe_{strategy}.png")

print("\nAll heatmaps generated.")