import pandas as pd
STOP_LOSS = 0.02
TAKE_PROFIT = 0.04

# Load prediction file
signals = pd.read_csv("signals/dcwrnn_signals.csv")

# Generate Buy/Sell signals
signals["Signal"] = 0

signals.loc[signals["Predicted_Close"] > signals["Actual_Close"], "Signal"] = 1
signals.loc[signals["Predicted_Close"] < signals["Actual_Close"], "Signal"] = -1

# Calculate daily market returns
signals["Market_Return"] = signals["Actual_Close"].pct_change()

signals["Stop_Loss_Return"] = signals["Market_Return"]

signals.loc[
    signals["Market_Return"] < -STOP_LOSS,
    "Stop_Loss_Return"
] = -STOP_LOSS

signals["Protected_Return"] = signals["Stop_Loss_Return"]

signals.loc[
    signals["Protected_Return"] > TAKE_PROFIT,
    "Protected_Return"
] = TAKE_PROFIT

signals["Strategy_Return"] = (
    signals["Signal"].shift(1) * signals["Protected_Return"]
)

# Prediction Strength
signals["Prediction_Change"] = (
    (signals["Predicted_Close"] - signals["Actual_Close"])
    / signals["Actual_Close"]
)

# Dynamic Position Size
signals["Position_Size"] = 0.2

signals.loc[
    signals["Prediction_Change"].abs() >= 0.005,
    "Position_Size"
] = 0.5

signals.loc[
    signals["Prediction_Change"].abs() >= 0.01,
    "Position_Size"
] = 1.0



# Transaction cost (0.1%)
transaction_cost = 0.001

# Apply transaction cost whenever signal changes
signals["Trade"] = signals["Signal"].diff().abs()

signals["Transaction_Cost"] = (
    signals["Trade"] * transaction_cost
)

signals["Net_Return"] = (
    signals["Strategy_Return"] * signals["Position_Size"]
    - signals["Transaction_Cost"]
)

print(signals.head(15))
print(signals.head(10))





# Initial investment
initial_capital = 100000

# Fill NaN values
signals["Net_Return"] = signals["Net_Return"].fillna(0)

# Portfolio value over time
signals["Portfolio_Value"] = (
    initial_capital *
    (1 + signals["Net_Return"]).cumprod()
)

print("\nFinal Portfolio Value:")
print(signals["Portfolio_Value"].tail())



import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(
    signals["Portfolio_Value"],
    color="blue",
    linewidth=2
)

plt.title("Portfolio Value Over Time")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value (₹)")
plt.grid(True)

plt.show()

signals.to_csv("signals/backtest_results.csv", index=False)

print("Backtest completed successfully.")
print("Results saved to signals/backtest_results.csv")


initial_capital = 100000

total_return = (
    (signals["Portfolio_Value"].iloc[-1] - initial_capital)
    / initial_capital
) * 100

print("\n========== PERFORMANCE ==========")
print(f"Total Return : {total_return:.2f}%")


rolling_max = signals["Portfolio_Value"].cummax()

drawdown = (
    signals["Portfolio_Value"] - rolling_max
) / rolling_max

max_drawdown = drawdown.min() * 100

print(f"Maximum Drawdown : {max_drawdown:.2f}%")



wins = signals["Net_Return"] > 0

win_rate = wins.mean() * 100

print(f"Win Rate : {win_rate:.2f}%")


avg_return = signals["Net_Return"].mean() * 100

print(f"Average Return per Trade : {avg_return:.4f}%")


volatility = signals["Net_Return"].std() * 100

print(f"Volatility : {volatility:.4f}%")


sharpe = (
    signals["Net_Return"].mean()
    / signals["Net_Return"].std()
)

print(f"Sharpe Ratio : {sharpe:.4f}")




print(
    signals[
        [
            "Actual_Close",
            "Predicted_Close",
            "Prediction_Change",
            "Position_Size"
        ]
    ].head(10)
)


print(signals.columns)



print(signals[
    ["Market_Return",
     "Strategy_Return",
     "Position_Size",
     "Net_Return"]
].head(10))




print(
    signals[
        signals["Market_Return"] < -STOP_LOSS
    ][["Market_Return", "Stop_Loss_Return"]]
)

