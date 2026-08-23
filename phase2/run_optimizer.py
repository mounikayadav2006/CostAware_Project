import pandas as pd
import numpy as np

from optimizer import optimize_position


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "../signals/backtest_results.csv"
OUTPUT_FILE = "results/phase2_results.csv"

TRANSACTION_COST = 0.001       # 0.1%
POSITION_LIMIT = 1.0           # maximum absolute position
TURNOVER_LIMIT = 0.20          # maximum daily position change


# --------------------------------------------------
# Load Phase 1 data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# Calculate optimizer signal
# --------------------------------------------------

# The DCWRNN prediction strength:
#
# Prediction_Change =
# (Predicted_Close - Actual_Close) / Actual_Close
#
# We use the previous day's prediction to determine
# the position for the next trading day.

df["Expected_Return"] = (
    df["Signal"] * df["Prediction_Change"].abs()
)

df["Expected_Return"] = df["Expected_Return"].shift(1)


# --------------------------------------------------
# Optimize positions
# --------------------------------------------------

optimized_positions = []

previous_position = 0.0


for _, row in df.iterrows():

    expected_return = row["Expected_Return"]

    # First row has no previous-day signal
    if pd.isna(expected_return):
        position = 0.0

    else:
        position = optimize_position(
            expected_return=expected_return,
            previous_position=previous_position,
            transaction_cost=TRANSACTION_COST,
            position_limit=POSITION_LIMIT,
            turnover_limit=TURNOVER_LIMIT
        )

    optimized_positions.append(position)

    previous_position = position


df["Optimized_Position"] = optimized_positions


# --------------------------------------------------
# Calculate turnover
# --------------------------------------------------

df["Optimized_Turnover"] = (
    df["Optimized_Position"]
    .diff()
    .abs()
    .fillna(0)
)


# --------------------------------------------------
# Calculate transaction cost
# --------------------------------------------------

df["Optimized_Transaction_Cost"] = (
    df["Optimized_Turnover"] * TRANSACTION_COST
)


# --------------------------------------------------
# Calculate optimized strategy return
# --------------------------------------------------

df["Optimized_Gross_Return"] = (
    df["Optimized_Position"]
    * df["Market_Return"]
)


# --------------------------------------------------
# Calculate net return
# --------------------------------------------------

df["Optimized_Net_Return"] = (
    df["Optimized_Gross_Return"]
    - df["Optimized_Transaction_Cost"]
)


df["Optimized_Net_Return"] = (
    df["Optimized_Net_Return"]
    .fillna(0)
)


# --------------------------------------------------
# Portfolio value
# --------------------------------------------------

initial_capital = 100000

df["Optimized_Portfolio_Value"] = (
    initial_capital
    * (1 + df["Optimized_Net_Return"]).cumprod()
)


# --------------------------------------------------
# Performance metrics
# --------------------------------------------------

total_return = (
    df["Optimized_Portfolio_Value"].iloc[-1]
    / initial_capital
    - 1
) * 100


rolling_max = (
    df["Optimized_Portfolio_Value"]
    .cummax()
)

drawdown = (
    df["Optimized_Portfolio_Value"]
    - rolling_max
) / rolling_max

max_drawdown = drawdown.min() * 100


volatility = df["Optimized_Net_Return"].std()

if volatility != 0:
    sharpe = (
        df["Optimized_Net_Return"].mean()
        / volatility
    )
else:
    sharpe = 0


turnover = (
    df["Optimized_Turnover"].sum()
)


total_transaction_cost = (
    df["Optimized_Transaction_Cost"].sum()
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n========== PHASE 2 RESULTS ==========")

print(f"Total Return        : {total_return:.2f}%")
print(f"Maximum Drawdown    : {max_drawdown:.2f}%")
print(f"Sharpe Ratio        : {sharpe:.4f}")
print(f"Total Turnover      : {turnover:.4f}")
print(
    f"Transaction Costs   : "
    f"{total_transaction_cost:.6f}"
)

print(
    f"Final Portfolio     : "
    f"₹{df['Optimized_Portfolio_Value'].iloc[-1]:.2f}"
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)