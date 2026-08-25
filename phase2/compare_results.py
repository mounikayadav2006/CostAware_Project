import pandas as pd


INITIAL_CAPITAL = 100000


def calculate_metrics(
    returns,
    portfolio_values,
    turnover
):
    total_return = (
        portfolio_values.iloc[-1] / INITIAL_CAPITAL - 1
    ) * 100

    rolling_max = portfolio_values.cummax()

    drawdown = (
        portfolio_values - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min() * 100

    volatility = returns.std()

    if volatility != 0:
        sharpe = returns.mean() / volatility
    else:
        sharpe = 0

    return {
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Turnover": turnover,
        "Final Portfolio": portfolio_values.iloc[-1]
    }


# -----------------------------
# Phase 1
# -----------------------------

phase1 = pd.read_csv(
    "../signals/backtest_results.csv"
)

phase1_metrics = calculate_metrics(
    phase1["Net_Return"],
    phase1["Portfolio_Value"],
    phase1["Position_Size"].diff().abs().fillna(0).sum()
)


# -----------------------------
# Phase 2
# -----------------------------

phase2 = pd.read_csv(
    "results/phase2_results.csv"
)

phase2_metrics = calculate_metrics(
    phase2["Optimized_Net_Return"],
    phase2["Optimized_Portfolio_Value"],
    phase2["Optimized_Turnover"].sum()
)


# -----------------------------
# Comparison
# -----------------------------

comparison = pd.DataFrame(
    [phase1_metrics, phase2_metrics],
    index=["Phase 1", "Phase 2"]
)


print("\n========== PHASE 1 vs PHASE 2 ==========\n")

print(
    comparison.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


comparison.to_csv(
    "results/phase1_vs_phase2.csv"
)

print("\nComparison saved to:")
print("results/phase1_vs_phase2.csv")