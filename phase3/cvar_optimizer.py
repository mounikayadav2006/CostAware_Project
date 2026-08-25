import pandas as pd
import numpy as np
import cvxpy as cp


# ============================================================
# PHASE 3: CVaR-CONSTRAINED PORTFOLIO OPTIMIZER
# ============================================================

INPUT_FILE = "../signals/backtest_results.csv"
OUTPUT_FILE = "results/phase3_cvar_results.csv"

INITIAL_CAPITAL = 100000

# Portfolio constraints
MAX_POSITION = 1.0
MAX_TURNOVER = 0.20

# Transaction cost
TRANSACTION_COST = 0.001

# CVaR settings
ALPHA = 0.95
CVAR_LIMIT = 0.012758310111757801

# Objective parameters
RISK_AVERSION = 0.10


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df["Market_Return"] = pd.to_numeric(
    df["Market_Return"], errors="coerce"
)

df["Prediction_Change"] = pd.to_numeric(
    df["Prediction_Change"], errors="coerce"
)

df["Signal"] = pd.to_numeric(
    df["Signal"], errors="coerce"
)

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 2. HISTORICAL RETURN SCENARIOS
# ============================================================

historical_returns = (
    df["Market_Return"]
    .dropna()
    .values
)

print("Number of historical return scenarios:",
      len(historical_returns))


# ============================================================
# 3. FUNCTION FOR CVaR-CONSTRAINED POSITION
# ============================================================

def optimize_position(expected_return,
                      previous_position,
                      scenarios):

    # Portfolio position
    w = cp.Variable()

    # VaR threshold
    z = cp.Variable()

    # CVaR excess-loss variables
    u = cp.Variable(len(scenarios), nonneg=True)

    # Scenario losses
    losses = -scenarios * w

    # Turnover
    turnover = cp.abs(w - previous_position)

    # CVaR
    cvar = (
        z
        + (
            1 / ((1 - ALPHA) * len(scenarios))
        ) * cp.sum(u)
    )

    # Objective:
    # maximize expected return
    # while penalizing turnover
    objective = cp.Maximize(
        expected_return * w
        - TRANSACTION_COST * turnover
        - RISK_AVERSION * cvar
    )

    # Constraints
    constraints = [

        # Position limit
        w >= -MAX_POSITION,
        w <= MAX_POSITION,

        # Turnover limit
        turnover <= MAX_TURNOVER,

        # Rockafellar-Uryasev CVaR formulation
        u >= losses - z,

        # CVaR limit
        cvar <= CVAR_LIMIT
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
        return previous_position, np.nan

    return float(w.value), float(cvar.value)


# ============================================================
# 4. WALK-FORWARD OPTIMIZATION
# ============================================================

positions = []
turnovers = []
cvar_values = []
expected_returns = []
net_returns = []

previous_position = 0.0


for i in range(len(df)):

    # First observation has no return
    if i == 0:

        positions.append(0.0)
        turnovers.append(0.0)
        cvar_values.append(0.0)
        expected_returns.append(np.nan)
        net_returns.append(0.0)

        continue


    # --------------------------------------------------------
    # Expected return from previous prediction signal
    # --------------------------------------------------------

    prediction_change = df.loc[
        i - 1,
        "Prediction_Change"
    ]

    expected_return = (
        prediction_change
    )


    # --------------------------------------------------------
    # Use only returns available BEFORE current date
    # --------------------------------------------------------

    scenarios = (
        df.loc[:i - 1, "Market_Return"]
        .dropna()
        .values
    )

    # Need enough observations for a meaningful tail
    if len(scenarios) < 20:

        position = previous_position
        cvar_value = np.nan

    else:

        position, cvar_value = optimize_position(
            expected_return,
            previous_position,
            scenarios
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

    portfolio_return = (
        position * market_return
        - transaction_cost
    )

    positions.append(position)
    turnovers.append(turnover)
    cvar_values.append(cvar_value)
    expected_returns.append(expected_return)
    net_returns.append(portfolio_return)

    previous_position = position


# ============================================================
# 5. SAVE RESULTS
# ============================================================

df["Expected_Return"] = expected_returns
df["CVaR_Position"] = positions
df["CVaR_Turnover"] = turnovers
df["CVaR_Value"] = cvar_values
df["CVaR_Net_Return"] = net_returns


# ============================================================
# 6. PORTFOLIO VALUE
# ============================================================

df["CVaR_Net_Return"] = (
    df["CVaR_Net_Return"]
    .fillna(0)
)

df["CVaR_Portfolio_Value"] = (
    INITIAL_CAPITAL
    * (1 + df["CVaR_Net_Return"])
    .cumprod()
)


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

final_value = (
    df["CVaR_Portfolio_Value"].iloc[-1]
)

total_return = (
    final_value / INITIAL_CAPITAL - 1
) * 100


rolling_max = (
    df["CVaR_Portfolio_Value"]
    .cummax()
)

drawdown = (
    df["CVaR_Portfolio_Value"]
    / rolling_max
    - 1
)

max_drawdown = (
    drawdown.min() * 100
)


returns = df["CVaR_Net_Return"]

if returns.std() != 0:
    sharpe = (
        returns.mean()
        / returns.std()
    )
else:
    sharpe = 0


total_turnover = (
    df["CVaR_Turnover"]
    .sum()
)

total_transaction_cost = (
    df["CVaR_Turnover"]
    .sum()
    * TRANSACTION_COST
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print()
print("========== PHASE 3 CVaR RESULTS ==========")

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
    f"Transaction Costs   : "
    f"{total_transaction_cost:.6f}"
)

print(
    f"Final Portfolio     : "
    f"₹{final_value:.2f}"
)

print(
    f"CVaR Limit          : "
    f"{CVAR_LIMIT:.6f}"
)


# ============================================================
# 9. SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Results saved to:")
print(OUTPUT_FILE)