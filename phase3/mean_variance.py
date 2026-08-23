import pandas as pd
import numpy as np
import cvxpy as cp


# ============================================================
# PHASE 3: MEAN-VARIANCE PORTFOLIO OPTIMIZER
# ============================================================

INPUT_FILE = "../signals/backtest_results.csv"
OUTPUT_FILE = "results/mean_variance_results.csv"

INITIAL_CAPITAL = 100000

MAX_POSITION = 1.0
MAX_TURNOVER = 0.20
TRANSACTION_COST = 0.001

# Risk-aversion parameter
RISK_AVERSION = 5.0


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df["Market_Return"] = pd.to_numeric(
    df["Market_Return"],
    errors="coerce"
)

df["Prediction_Change"] = pd.to_numeric(
    df["Prediction_Change"],
    errors="coerce"
)

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 2. OPTIMIZATION FUNCTION
# ============================================================

def optimize_position(expected_return,
                      previous_position,
                      historical_returns):

    w = cp.Variable()

    turnover = cp.abs(
        w - previous_position
    )

    # Historical variance estimate
    variance = np.var(
        historical_returns
    )

    # Mean-variance objective
    objective = cp.Maximize(
        expected_return * w
        - RISK_AVERSION * variance * cp.square(w)
        - TRANSACTION_COST * turnover
    )

    constraints = [
        w >= -MAX_POSITION,
        w <= MAX_POSITION,
        turnover <= MAX_TURNOVER
    ]

    problem = cp.Problem(
        objective,
        constraints
    )

    problem.solve(
        solver="CLARABEL"
    )

    if problem.status not in [
        "optimal",
        "optimal_inaccurate"
    ]:
        return previous_position

    return float(w.value)


# ============================================================
# 3. WALK-FORWARD OPTIMIZATION
# ============================================================

positions = []
turnovers = []
net_returns = []
expected_returns = []

previous_position = 0.0


for i in range(len(df)):

    if i == 0:

        positions.append(0.0)
        turnovers.append(0.0)
        net_returns.append(0.0)
        expected_returns.append(np.nan)

        continue


    # Prediction available before current return
    expected_return = df.loc[
        i - 1,
        "Prediction_Change"
    ]


    # Historical returns available before current date
    scenarios = (
        df.loc[:i - 1, "Market_Return"]
        .dropna()
        .values
    )


    if len(scenarios) < 20:

        position = previous_position

    else:

        position = optimize_position(
            expected_return,
            previous_position,
            scenarios
        )


    market_return = df.loc[
        i,
        "Market_Return"
    ]


    turnover = abs(
        position - previous_position
    )


    transaction_cost = (
        turnover * TRANSACTION_COST
    )


    portfolio_return = (
        position * market_return
        - transaction_cost
    )


    positions.append(position)
    turnovers.append(turnover)
    net_returns.append(portfolio_return)
    expected_returns.append(expected_return)

    previous_position = position


# ============================================================
# 4. SAVE RESULTS
# ============================================================

df["Expected_Return"] = expected_returns
df["MV_Position"] = positions
df["MV_Turnover"] = turnovers
df["MV_Net_Return"] = net_returns


# ============================================================
# 5. PORTFOLIO VALUE
# ============================================================

df["MV_Net_Return"] = (
    df["MV_Net_Return"]
    .fillna(0)
)

df["MV_Portfolio_Value"] = (
    INITIAL_CAPITAL
    * (1 + df["MV_Net_Return"])
    .cumprod()
)


# ============================================================
# 6. PERFORMANCE METRICS
# ============================================================

final_value = (
    df["MV_Portfolio_Value"].iloc[-1]
)

total_return = (
    final_value / INITIAL_CAPITAL - 1
) * 100


rolling_max = (
    df["MV_Portfolio_Value"]
    .cummax()
)

drawdown = (
    df["MV_Portfolio_Value"]
    / rolling_max
    - 1
)

max_drawdown = (
    drawdown.min() * 100
)


returns = df["MV_Net_Return"]

if returns.std() != 0:

    sharpe = (
        returns.mean()
        / returns.std()
    )

else:

    sharpe = 0


total_turnover = (
    df["MV_Turnover"].sum()
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print()
print("========== MEAN-VARIANCE RESULTS ==========")

print(
    f"Total Return     : {total_return:.2f}%"
)

print(
    f"Maximum Drawdown : {max_drawdown:.2f}%"
)

print(
    f"Sharpe Ratio     : {sharpe:.4f}"
)

print(
    f"Total Turnover   : {total_turnover:.4f}"
)

print(
    f"Final Portfolio  : ₹{final_value:.2f}"
)


# ============================================================
# 8. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Results saved to:")
print(OUTPUT_FILE)