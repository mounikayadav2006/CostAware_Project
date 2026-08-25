import pandas as pd
import numpy as np
import cvxpy as cp


# ============================================================
# WALK-FORWARD CVaR-CONSTRAINED OPTIMIZER
# ============================================================

INPUT_FILE = "../signals/backtest_results.csv"
OUTPUT_FILE = "results/walkforward_cvar_results.csv"

INITIAL_CAPITAL = 100000

MAX_POSITION = 1.0
MAX_TURNOVER = 0.20
TRANSACTION_COST = 0.001

CONFIDENCE = 0.95
MIN_HISTORY = 60


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
# 2. HISTORICAL CVaR
# ============================================================

def historical_cvar(returns, confidence=0.95):

    returns = np.asarray(returns)

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) == 0:
        return 0.0

    var = np.quantile(
        returns,
        1 - confidence
    )

    tail = returns[
        returns <= var
    ]

    if len(tail) == 0:
        return abs(var)

    # Loss magnitude is positive
    return abs(tail.mean())


# ============================================================
# 3. CVaR OPTIMIZER
# ============================================================

def optimize_position(
    expected_return,
    previous_position,
    scenarios,
    cvar_limit
):

    w = cp.Variable()

    z = cp.Variable()

    excess_loss = cp.Variable(
        len(scenarios),
        nonneg=True
    )

    # Portfolio loss in each historical scenario
    losses = -w * scenarios

    constraints = [

        # Position limits
        w >= -MAX_POSITION,
        w <= MAX_POSITION,

        # Turnover limit
        cp.abs(
            w - previous_position
        ) <= MAX_TURNOVER,

        # Rockafellar-Uryasev CVaR formulation
        excess_loss >= losses - z,

        # CVaR <= limit
        z
        + (
            1
            / ((1 - CONFIDENCE) * len(scenarios))
        )
        * cp.sum(excess_loss)
        <= cvar_limit
    ]

    # Cost-aware expected return objective
    turnover = cp.abs(
        w - previous_position
    )

    objective = cp.Maximize(
        expected_return * w
        - TRANSACTION_COST * turnover
    )

    problem = cp.Problem(
        objective,
        constraints
    )

    try:

        problem.solve(
            solver="CLARABEL"
        )

    except Exception:

        return previous_position, cvar_limit

    if problem.status not in [
        "optimal",
        "optimal_inaccurate"
    ]:

        return previous_position, cvar_limit

    return float(w.value), float(
        z.value
        + (
            1
            / ((1 - CONFIDENCE) * len(scenarios))
        )
        * np.sum(
            np.maximum(
                -float(w.value) * scenarios
                - float(z.value),
                0
            )
        )
    )


# ============================================================
# 4. WALK-FORWARD LOOP
# ============================================================

positions = []
turnovers = []
net_returns = []
cvar_limits = []
cvar_values = []
expected_returns = []

previous_position = 0.0


for i in range(len(df)):

    # First observation has no previous return
    if i == 0:

        positions.append(0.0)
        turnovers.append(0.0)
        net_returns.append(0.0)
        cvar_limits.append(np.nan)
        cvar_values.append(0.0)
        expected_returns.append(np.nan)

        continue


    # --------------------------------------------------------
    # Information available BEFORE today's return
    # --------------------------------------------------------

    expected_return = df.loc[
        i - 1,
        "Prediction_Change"
    ]

    historical_returns = (
        df.loc[
            :i - 1,
            "Market_Return"
        ]
        .dropna()
        .values
    )


    # --------------------------------------------------------
    # Not enough history
    # --------------------------------------------------------

    if len(historical_returns) < MIN_HISTORY:

        position = previous_position

        cvar_limit = np.nan

        cvar_value = np.nan

    else:

        # Calculate CVaR using ONLY past observations
        cvar_limit = historical_cvar(
            historical_returns,
            CONFIDENCE
        )

        position, cvar_value = optimize_position(
            expected_return,
            previous_position,
            historical_returns,
            cvar_limit
        )


    # --------------------------------------------------------
    # Current market return
    # --------------------------------------------------------

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

    strategy_return = (
        position * market_return
        - transaction_cost
    )


    positions.append(position)
    turnovers.append(turnover)
    net_returns.append(strategy_return)
    cvar_limits.append(cvar_limit)
    cvar_values.append(cvar_value)
    expected_returns.append(expected_return)

    previous_position = position


# ============================================================
# 5. SAVE STRATEGY RESULTS
# ============================================================

df["Expected_Return"] = expected_returns

df["WF_CVaR_Position"] = positions

df["WF_CVaR_Turnover"] = turnovers

df["WF_CVaR_Limit"] = cvar_limits

df["WF_CVaR_Value"] = cvar_values

df["WF_CVaR_Net_Return"] = net_returns


# ============================================================
# 6. PORTFOLIO VALUE
# ============================================================

df["WF_CVaR_Net_Return"] = (
    df["WF_CVaR_Net_Return"]
    .fillna(0)
)

df["WF_CVaR_Portfolio_Value"] = (
    INITIAL_CAPITAL
    * (
        1 + df["WF_CVaR_Net_Return"]
    ).cumprod()
)


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

returns = df[
    "WF_CVaR_Net_Return"
]

final_value = (
    df["WF_CVaR_Portfolio_Value"]
    .iloc[-1]
)

total_return = (
    final_value / INITIAL_CAPITAL - 1
) * 100


rolling_max = (
    df["WF_CVaR_Portfolio_Value"]
    .cummax()
)

drawdown = (
    df["WF_CVaR_Portfolio_Value"]
    / rolling_max
    - 1
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


total_turnover = (
    df["WF_CVaR_Turnover"].sum()
)


# Realized CVaR

realized_var = returns.quantile(
    1 - CONFIDENCE
)

tail = returns[
    returns <= realized_var
]

realized_cvar = (
    tail.mean()
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print()
print("======================================================")
print("        WALK-FORWARD CVaR RESULTS")
print("======================================================")

print(
    f"Total Return        : {total_return:.2f}%"
)

print(
    f"Maximum Drawdown    : {max_drawdown:.2f}%"
)

print(
    f"Sharpe Ratio        : {sharpe:.4f}"
)

print(
    f"Total Turnover      : {total_turnover:.4f}"
)

print(
    f"Realized VaR 95%    : {realized_var:.6f}"
)

print(
    f"Realized CVaR 95%   : {realized_cvar:.6f}"
)

print(
    f"Final Portfolio     : ₹{final_value:.2f}"
)


# ============================================================
# 9. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Results saved to:")
print(OUTPUT_FILE)