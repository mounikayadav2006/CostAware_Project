import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
})


def plot_equity_curves(raw_df, real_df, out_path):
    plt.figure(figsize=(11, 5.5))
    plt.plot(raw_df["Date"], raw_df["Portfolio_Value"], "--", label="Raw / zero-cost (thesis-simple)")
    plt.plot(real_df["Date"], real_df["Portfolio_Value"], linewidth=2, label="Realistic-cost")
    plt.axhline(raw_df["Portfolio_Value"].iloc[0], color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Date"); plt.ylabel("Portfolio Value (₹)")
    plt.title("DCWRNN Strategy: Raw vs. Realistic-Cost Equity Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()


def plot_drawdown(df, out_path, label="Realistic-cost"):
    pv = df["Portfolio_Value"]
    dd = (pv - pv.cummax()) / pv.cummax() * 100
    plt.figure(figsize=(11, 4))
    plt.fill_between(df["Date"], dd, 0, color="firebrick", alpha=0.4)
    plt.plot(df["Date"], dd, color="firebrick", linewidth=1)
    plt.xlabel("Date"); plt.ylabel("Drawdown (%)")
    plt.title(f"Drawdown Over Time — {label}")
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def plot_sensitivity_heatmap(sweep_df, metric_col, out_path, title):
    pivot = sweep_df.pivot(index="cost_bps", columns="slippage_bps", values=metric_col)
    plt.figure(figsize=(7, 5.5))
    im = plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Slippage (bps)"); plt.ylabel("Transaction Cost (bps)")
    plt.title(title)
    plt.colorbar(im, label=metric_col)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def metrics_to_dataframe(*metric_dicts):
    return pd.DataFrame(metric_dicts)


def save_results_table(df, csv_path, latex_path, caption="Backtest results"):
    df.to_csv(csv_path, index=False)
    with open(latex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f", caption=caption))