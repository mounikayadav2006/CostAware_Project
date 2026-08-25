import pandas as pd
import numpy as np

INITIAL_CAPITAL = 100000


# ============================================================
# LOAD RESULTS
# ============================================================

p1 = pd.read_csv("../signals/backtest_results.csv")

p2 = pd.read_csv(
    "../phase2/results/phase2_results.csv"
)

mv = pd.read_csv(
    "results/mean_variance_results.csv"
)

# IMPORTANT:
# Final Phase 3 = WALK-FORWARD CVaR
wf = pd.read_csv(
    "results/walkforward_cvar_results.csv"
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    returns,
    portfolio,
    turnover
):

    returns = returns.dropna()

    final_value = portfolio.iloc[-1]

    total_return = (
        final_value / INITIAL_CAPITAL - 1
    ) * 100

    rolling_max = portfolio.cummax()

    drawdown = (
        portfolio / rolling_max - 1
    )

    max_drawdown = (
        drawdown.min() * 100
    )

    if returns.std() != 0:

        sharpe = (
            returns.mean()
            / returns.std()
        )

    else:

        sharpe = 0

    total_turnover = turnover.sum()

    # Historical realized VaR
    var95 = returns.quantile(0.05)

    # Historical realized CVaR
    tail = returns[
        returns <= var95
    ]

    cvar95 = tail.mean()

    return {
        "Total Return (%)": total_return,
        "Maximum Drawdown (%)": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Turnover": total_turnover,
        "VaR 95%": var95,
        "CVaR 95%": cvar95,
        "Final Portfolio": final_value
    }


# ============================================================
# PHASE 1 — BASELINE
# ============================================================

phase1 = calculate_metrics(

    p1["Net_Return"],

    p1["Portfolio_Value"],

    p1["Position_Size"]
    .diff()
    .abs()
    .fillna(0)
)


# ============================================================
# PHASE 2 — COST-AWARE OPTIMIZER
# ============================================================

phase2 = calculate_metrics(

    p2["Optimized_Net_Return"],

    p2["Optimized_Portfolio_Value"],

    p2["Optimized_Turnover"]
)


# ============================================================
# MEAN-VARIANCE
# ============================================================

mean_variance = calculate_metrics(

    mv["MV_Net_Return"],

    mv["MV_Portfolio_Value"],

    mv["MV_Turnover"]
)


# ============================================================
# PHASE 3 — WALK-FORWARD CVaR
# ============================================================

phase3 = calculate_metrics(

    wf["WF_CVaR_Net_Return"],

    wf["WF_CVaR_Portfolio_Value"],

    wf["WF_CVaR_Turnover"]
)


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(

    {
        "Phase 1 Baseline": phase1,
        "Phase 2 Cost-Aware": phase2,
        "Mean-Variance": mean_variance,
        "Phase 3 Walk-Forward CVaR": phase3
    }

).T


# ============================================================
# DISPLAY
# ============================================================

print()

print(
    "======================================================"
)

print(
    "           FINAL STRATEGY COMPARISON"
)

print(
    "======================================================"
)

print()

print(
    comparison.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

comparison.to_csv(
    "results/final_comparison.csv"
)

print()

print(
    "Saved to:"
)

print(
    "results/final_comparison.csv"
)