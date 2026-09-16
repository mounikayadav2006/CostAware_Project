import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})

results = pd.read_csv("outputs/walk_forward_results.csv")

windows = sorted(results["window"].unique(), key=lambda w: int(w.split("_")[1]))
strategies = ["naive", "convex_optimized", "cvar_constrained"]
colors = {"naive": "#888888", "convex_optimized": "#1f77b4", "cvar_constrained": "#2ca02c"}
labels = {"naive": "Naive", "convex_optimized": "Convex Optimized", "cvar_constrained": "CVaR-Constrained"}

x = np.arange(len(windows))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, strategy in enumerate(strategies):
    strategy_df = results[results["strategy"] == strategy].set_index("window").loc[windows]
    sharpe_values = strategy_df["sharpe"].values
    offset = (i - 1) * width
    ax.bar(x + offset, sharpe_values, width, label=labels[strategy], color=colors[strategy])

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Walk-Forward Window")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("Sharpe Ratio by Strategy Across 5 Rolling 60-Day Windows")
ax.set_xticks(x)
ax.set_xticklabels([w.replace("_", " ").title() for w in windows])
ax.legend()
plt.tight_layout()
plt.savefig("graphs/walk_forward_sharpe_by_window.png")
plt.close()

print("Saved graphs/walk_forward_sharpe_by_window.png")