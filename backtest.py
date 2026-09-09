# import pandas as pd
# STOP_LOSS = 0.02
# TAKE_PROFIT = 0.04

# # Load prediction file
# signals = pd.read_csv("signals/dcwrnn_signals.csv")

# # Generate Buy/Sell signals
# signals["Signal"] = 0

# signals.loc[signals["Predicted_Close"] > signals["Actual_Close"], "Signal"] = 1
# signals.loc[signals["Predicted_Close"] < signals["Actual_Close"], "Signal"] = -1

# # Calculate daily market returns
# signals["Market_Return"] = signals["Actual_Close"].pct_change()

# signals["Stop_Loss_Return"] = signals["Market_Return"]

# signals.loc[
#     signals["Market_Return"] < -STOP_LOSS,
#     "Stop_Loss_Return"
# ] = -STOP_LOSS

# signals["Protected_Return"] = signals["Stop_Loss_Return"]

# signals.loc[
#     signals["Protected_Return"] > TAKE_PROFIT,
#     "Protected_Return"
# ] = TAKE_PROFIT

# signals["Strategy_Return"] = (
#     signals["Signal"].shift(1) * signals["Protected_Return"]
# )

# # Prediction Strength
# signals["Prediction_Change"] = (
#     (signals["Predicted_Close"] - signals["Actual_Close"])
#     / signals["Actual_Close"]
# )

# # Dynamic Position Size
# signals["Position_Size"] = 0.2

# signals.loc[
#     signals["Prediction_Change"].abs() >= 0.005,
#     "Position_Size"
# ] = 0.5

# signals.loc[
#     signals["Prediction_Change"].abs() >= 0.01,
#     "Position_Size"
# ] = 1.0



# # Transaction cost (0.1%)
# transaction_cost = 0.001


# # Slippage cost (0.05%)
# slippage = 0.0005

# # Apply transaction cost whenever signal changes
# signals["Trade"] = (
#     signals["Signal"] != signals["Signal"].shift(1)
# ).astype(int)

# signals["Transaction_Cost"] = (
#     signals["Trade"] * transaction_cost
# )

# signals["Slippage_Cost"] = (
#     signals["Trade"] * slippage
# )

# signals["Net_Return"] = (
#     signals["Strategy_Return"] * signals["Position_Size"]
#     - signals["Transaction_Cost"]
#     - signals["Slippage_Cost"]
# )

# print(signals.head(15))
# print(signals.head(10))





# # Initial investment
# initial_capital = 100000

# # Fill NaN values
# signals["Net_Return"] = signals["Net_Return"].fillna(0)

# # Portfolio value over time
# signals["Portfolio_Value"] = (
#     initial_capital *
#     (1 + signals["Net_Return"]).cumprod()
# )

# print("\nFinal Portfolio Value:")
# print(signals["Portfolio_Value"].tail())



# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,6))

# plt.plot(
#     signals["Portfolio_Value"],
#     color="blue",
#     linewidth=2
# )

# plt.title("Portfolio Value Over Time")
# plt.xlabel("Trading Days")
# plt.ylabel("Portfolio Value (₹)")
# plt.grid(True)

# plt.show()

# signals.to_csv("signals/backtest_results.csv", index=False)

# print("Backtest completed successfully.")
# print("Results saved to signals/backtest_results.csv")


# initial_capital = 100000

# total_return = (
#     (signals["Portfolio_Value"].iloc[-1] - initial_capital)
#     / initial_capital
# ) * 100

# print("\n========== PERFORMANCE ==========")
# print(f"Total Return : {total_return:.2f}%")


# rolling_max = signals["Portfolio_Value"].cummax()

# drawdown = (
#     signals["Portfolio_Value"] - rolling_max
# ) / rolling_max

# max_drawdown = drawdown.min() * 100

# print(f"Maximum Drawdown : {max_drawdown:.2f}%")



# wins = signals["Net_Return"] > 0

# win_rate = wins.mean() * 100

# print(f"Win Rate : {win_rate:.2f}%")


# avg_return = signals["Net_Return"].mean() * 100

# print(f"Average Return per Trade : {avg_return:.4f}%")


# volatility = signals["Net_Return"].std() * 100

# print(f"Volatility : {volatility:.4f}%")


# sharpe = (
#     signals["Net_Return"].mean()
#     / signals["Net_Return"].std()
# )

# print(f"Sharpe Ratio : {sharpe:.4f}")




# print(
#     signals[
#         [
#             "Actual_Close",
#             "Predicted_Close",
#             "Prediction_Change",
#             "Position_Size"
#         ]
#     ].head(10)
# )


# print(signals.columns)



# print(signals[
# [
# "Market_Return",
# "Stop_Loss_Return",
# "Protected_Return",
# "Strategy_Return",
# "Net_Return"
# ]
# ].head(10))




# print(
#     signals[
#         signals["Market_Return"] < -STOP_LOSS
#     ][["Market_Return", "Stop_Loss_Return"]]
# )


# print(
#     signals[
#         [
#             "Trade",
#             "Transaction_Cost",
#             "Slippage_Cost",
#             "Net_Return"
#         ]
#     ].head(10)
# )



# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # =========================
# # PARAMETERS
# # =========================
# STOP_LOSS = 0.02
# TAKE_PROFIT = 0.04

# TRANSACTION_COST = 0.001   # 0.1%
# SLIPPAGE = 0.0005          # 0.05%

# INITIAL_CAPITAL = 100000


# # =========================
# # 1. LOAD PREDICTIONS
# # =========================
# signals = pd.read_csv("signals/dcwrnn_signals.csv")

# signals["Date"] = pd.to_datetime(signals["Date"])
# signals = signals.sort_values("Date").reset_index(drop=True)


# # =========================
# # 2. MARKET RETURN
# # =========================
# signals["Market_Return"] = signals["Actual_Close"].pct_change()


# # =========================
# # 3. FORECAST RETURN
# # =========================
# # Prediction for day t is compared with
# # the CLOSE available at the end of day t-1.

# signals["Forecast_Return"] = (
#     signals["Predicted_Close"] /
#     signals["Actual_Close"].shift(1)
# ) - 1


# # =========================
# # 4. TRADING SIGNAL
# # =========================
# # +1 = BUY
# # -1 = SELL
# #  0 = HOLD

# signals["Signal"] = 0

# signals.loc[
#     signals["Forecast_Return"] > 0,
#     "Signal"
# ] = 1

# signals.loc[
#     signals["Forecast_Return"] < 0,
#     "Signal"
# ] = -1


# # =========================
# # 5. STRATEGY RETURN
# # =========================
# # Prediction made for day t is used
# # for the return of day t.

# signals["Strategy_Return"] = (
#     signals["Signal"] *
#     signals["Market_Return"]
# )


# # =========================
# # 6. STOP LOSS / TAKE PROFIT
# # =========================
# # Apply limits AFTER considering
# # whether the position is BUY or SELL.

# signals["Protected_Return"] = signals["Strategy_Return"].clip(
#     lower=-STOP_LOSS,
#     upper=TAKE_PROFIT
# )


# # =========================
# # 7. POSITION SIZING
# # =========================
# # Strength of prediction is based on
# # information available at decision time.

# signals["Prediction_Strength"] = (
#     signals["Forecast_Return"].abs()
# )

# signals["Position_Size"] = 0.2

# signals.loc[
#     signals["Prediction_Strength"] >= 0.005,
#     "Position_Size"
# ] = 0.5

# signals.loc[
#     signals["Prediction_Strength"] >= 0.01,
#     "Position_Size"
# ] = 1.0


# # =========================
# # 8. ACTUAL POSITION
# # =========================
# signals["Position"] = (
#     signals["Signal"] *
#     signals["Position_Size"]
# )


# # =========================
# # 9. TURNOVER
# # =========================
# # Cost should depend on how much
# # the position changes.

# signals["Previous_Position"] = (
#     signals["Position"].shift(1).fillna(0)
# )

# signals["Turnover"] = (
#     signals["Position"] -
#     signals["Previous_Position"]
# ).abs()


# # =========================
# # 10. TRANSACTION COST
# # =========================
# signals["Transaction_Cost"] = (
#     signals["Turnover"] *
#     TRANSACTION_COST
# )


# # =========================
# # 11. SLIPPAGE COST
# # =========================
# signals["Slippage_Cost"] = (
#     signals["Turnover"] *
#     SLIPPAGE
# )


# # =========================
# # 12. NET RETURN
# # =========================
# signals["Net_Return"] = (
#     signals["Protected_Return"] *
#     signals["Position_Size"]
#     - signals["Transaction_Cost"]
#     - signals["Slippage_Cost"]
# )


# # First row has no previous market return,
# # therefore no trading return.

# signals.loc[0, "Net_Return"] = 0


# # =========================
# # 13. PORTFOLIO VALUE
# # =========================
# signals["Portfolio_Value"] = (
#     INITIAL_CAPITAL *
#     (1 + signals["Net_Return"].fillna(0)).cumprod()
# )


# # =========================
# # 14. TOTAL RETURN
# # =========================
# total_return = (
#     (signals["Portfolio_Value"].iloc[-1] /
#      INITIAL_CAPITAL) - 1
# ) * 100


# # =========================
# # 15. MAXIMUM DRAWDOWN
# # =========================
# rolling_max = signals["Portfolio_Value"].cummax()

# drawdown = (
#     signals["Portfolio_Value"] -
#     rolling_max
# ) / rolling_max

# max_drawdown = drawdown.min() * 100


# # =========================
# # 16. WIN RATE
# # =========================
# positive_returns = signals.loc[
#     signals["Net_Return"] != 0,
#     "Net_Return"
# ]

# win_rate = (
#     (positive_returns > 0).mean() * 100
# )


# # =========================
# # 17. VOLATILITY
# # =========================
# daily_volatility = (
#     signals["Net_Return"].std()
# )

# annualized_volatility = (
#     daily_volatility * np.sqrt(252) * 100
# )


# # =========================
# # 18. ANNUALIZED SHARPE RATIO
# # =========================
# if daily_volatility != 0:

#     sharpe_ratio = (
#         signals["Net_Return"].mean() /
#         daily_volatility
#     ) * np.sqrt(252)

# else:
#     sharpe_ratio = 0


# # =========================
# # 19. SAVE RESULTS
# # =========================
# os.makedirs("signals", exist_ok=True)

# signals.to_csv(
#     "signals/backtest_results.csv",
#     index=False
# )


# # =========================
# # 20. SAVE GRAPH
# # =========================
# os.makedirs("graphs", exist_ok=True)

# plt.figure(figsize=(12, 6))

# plt.plot(
#     signals["Date"],
#     signals["Portfolio_Value"]
# )

# plt.axhline(
#     INITIAL_CAPITAL,
#     linestyle="--"
# )

# plt.xlabel("Date")
# plt.ylabel("Portfolio Value (₹)")
# plt.title("DCWRNN Strategy Portfolio Value")

# plt.grid(True)

# plt.tight_layout()

# plt.savefig(
#     "graphs/dcwrnn_portfolio_value.png",
#     dpi=300
# )

# plt.show()


# # =========================
# # 21. PRINT RESULTS
# # =========================
# print("\n========== BACKTEST RESULTS ==========")

# print(
#     f"Initial Capital       : ₹{INITIAL_CAPITAL:,.2f}"
# )

# print(
#     f"Final Portfolio Value : ₹{signals['Portfolio_Value'].iloc[-1]:,.2f}"
# )

# print(
#     f"Total Return          : {total_return:.2f}%"
# )

# print(
#     f"Maximum Drawdown      : {max_drawdown:.2f}%"
# )

# print(
#     f"Win Rate              : {win_rate:.2f}%"
# )

# print(
#     f"Annualized Volatility : {annualized_volatility:.2f}%"
# )

# print(
#     f"Annualized Sharpe     : {sharpe_ratio:.2f}"
# )

# print("\nResults saved to:")
# print("signals/backtest_results.csv")

# print("\nGraph saved to:")
# print("graphs/dcwrnn_portfolio_value.png")




# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # =========================
# # CONFIG (defaults — override per-call)
# # =========================
# DEFAULT_TRANSACTION_COST_BPS = 10   # 10 bps = 0.10%
# DEFAULT_SLIPPAGE_BPS = 5            # 5 bps  = 0.05%
# DEFAULT_STOP_LOSS = 0.02            # -2%
# DEFAULT_TAKE_PROFIT = 0.04          # +4%
# INITIAL_CAPITAL = 100000


# def load_signals(path="signals/dcwrnn_signals.csv"):
#     """Load the frozen DCWRNN signal series (Date, Actual_Close, Predicted_Close)."""
#     df = pd.read_csv(path)
#     df["Date"] = pd.to_datetime(df["Date"])
#     df = df.sort_values("Date").reset_index(drop=True)
#     return df


# def run_backtest(
#     signals_df,
#     cost_bps=DEFAULT_TRANSACTION_COST_BPS,
#     slippage_bps=DEFAULT_SLIPPAGE_BPS,
#     stop_loss=DEFAULT_STOP_LOSS,      # set to None to disable
#     take_profit=DEFAULT_TAKE_PROFIT,  # set to None to disable
#     position_sizing=True,             # False = always full size (1.0)
#     initial_capital=INITIAL_CAPITAL,
# ):
#     """
#     Event-driven backtest engine.

#     Parameters
#     ----------
#     signals_df : DataFrame with Date, Actual_Close, Predicted_Close
#     cost_bps, slippage_bps : trading frictions in basis points (1 bps = 0.01%)
#     stop_loss, take_profit : fractional return caps applied to the signed
#         position return; pass None to disable either
#     position_sizing : if True, scale position 0.2 / 0.5 / 1.0 by prediction
#         strength (matches thesis rule); if False, always trade full size —
#         use this to reproduce the "thesis-simple" long/short rule.

#     Returns
#     -------
#     (results_df, metrics_dict)
#     """
#     df = signals_df.copy()

#     transaction_cost = cost_bps / 10000
#     slippage = slippage_bps / 10000

#     # ---- Market return ----
#     df["Market_Return"] = df["Actual_Close"].pct_change()

#     # ---- Forecast return (no look-ahead: Predicted_Close[t] vs Actual_Close[t-1]) ----
#     df["Forecast_Return"] = (
#         df["Predicted_Close"] / df["Actual_Close"].shift(1)
#     ) - 1

#     # ---- Signal ----
#     df["Signal"] = 0
#     df.loc[df["Forecast_Return"] > 0, "Signal"] = 1
#     df.loc[df["Forecast_Return"] < 0, "Signal"] = -1

#     # ---- Raw strategy return (signed) ----
#     df["Strategy_Return"] = df["Signal"] * df["Market_Return"]

#     # ---- Stop-loss / take-profit on the signed return ----
#     if stop_loss is not None or take_profit is not None:
#         lower = -stop_loss if stop_loss is not None else -np.inf
#         upper = take_profit if take_profit is not None else np.inf
#         df["Protected_Return"] = df["Strategy_Return"].clip(lower=lower, upper=upper)
#     else:
#         df["Protected_Return"] = df["Strategy_Return"]

#     # ---- Position sizing ----
#     df["Prediction_Strength"] = df["Forecast_Return"].abs()

#     if position_sizing:
#         df["Position_Size"] = 0.2
#         df.loc[df["Prediction_Strength"] >= 0.005, "Position_Size"] = 0.5
#         df.loc[df["Prediction_Strength"] >= 0.01, "Position_Size"] = 1.0
#     else:
#         df["Position_Size"] = 1.0

#     df["Position"] = df["Signal"] * df["Position_Size"]

#     # ---- Turnover-based costs ----
#     df["Previous_Position"] = df["Position"].shift(1).fillna(0)
#     df["Turnover"] = (df["Position"] - df["Previous_Position"]).abs()

#     df["Transaction_Cost"] = df["Turnover"] * transaction_cost
#     df["Slippage_Cost"] = df["Turnover"] * slippage

#     # ---- Net return ----
#     df["Net_Return"] = (
#         df["Protected_Return"] * df["Position_Size"]
#         - df["Transaction_Cost"]
#         - df["Slippage_Cost"]
#     )
#     df.loc[0, "Net_Return"] = 0  # first row: no prior close to trade against

#     # ---- Portfolio value ----
#     df["Portfolio_Value"] = initial_capital * (1 + df["Net_Return"].fillna(0)).cumprod()

#     # ---- Metrics ----
#     total_return_pct = (df["Portfolio_Value"].iloc[-1] / initial_capital - 1) * 100

#     rolling_max = df["Portfolio_Value"].cummax()
#     drawdown = (df["Portfolio_Value"] - rolling_max) / rolling_max
#     max_drawdown_pct = drawdown.min() * 100

#     traded_returns = df.loc[df["Net_Return"] != 0, "Net_Return"]
#     win_rate_pct = (traded_returns > 0).mean() * 100 if len(traded_returns) else 0.0

#     daily_vol = df["Net_Return"].std()
#     annualized_vol_pct = daily_vol * np.sqrt(252) * 100

#     sharpe_ratio = (
#         (df["Net_Return"].mean() / daily_vol) * np.sqrt(252)
#         if daily_vol not in (0, np.nan) else 0.0
#     )

#     metrics = {
#         "cost_bps": cost_bps,
#         "slippage_bps": slippage_bps,
#         "final_portfolio_value": df["Portfolio_Value"].iloc[-1],
#         "total_return_pct": total_return_pct,
#         "max_drawdown_pct": max_drawdown_pct,
#         "win_rate_pct": win_rate_pct,
#         "annualized_volatility_pct": annualized_vol_pct,
#         "sharpe_ratio": sharpe_ratio,
#     }

#     return df, metrics


# def print_metrics(label, metrics):
#     print(f"\n========== {label} ==========")
#     print(f"Cost / Slippage        : {metrics['cost_bps']} bps / {metrics['slippage_bps']} bps")
#     print(f"Final Portfolio Value  : ₹{metrics['final_portfolio_value']:,.2f}")
#     print(f"Total Return           : {metrics['total_return_pct']:.2f}%")
#     print(f"Maximum Drawdown       : {metrics['max_drawdown_pct']:.2f}%")
#     print(f"Win Rate               : {metrics['win_rate_pct']:.2f}%")
#     print(f"Annualized Volatility  : {metrics['annualized_volatility_pct']:.2f}%")
#     print(f"Annualized Sharpe      : {metrics['sharpe_ratio']:.2f}")


# if __name__ == "__main__":
#     os.makedirs("signals", exist_ok=True)
#     os.makedirs("graphs", exist_ok=True)

#     signals = load_signals("signals/dcwrnn_signals.csv")

#     # ---- Run 1: thesis-simple, zero-cost reference (no frictions, no sizing, no SL/TP) ----
#     raw_df, raw_metrics = run_backtest(
#         signals,
#         cost_bps=0,
#         slippage_bps=0,
#         stop_loss=None,
#         take_profit=None,
#         position_sizing=False,
#     )
#     print_metrics("RAW / ZERO-COST (thesis-simple long/short)", raw_metrics)

#     # ---- Run 2: realistic-cost version (full rule: sizing + SL/TP + costs) ----
#     real_df, real_metrics = run_backtest(
#         signals,
#         cost_bps=DEFAULT_TRANSACTION_COST_BPS,
#         slippage_bps=DEFAULT_SLIPPAGE_BPS,
#         stop_loss=DEFAULT_STOP_LOSS,
#         take_profit=DEFAULT_TAKE_PROFIT,
#         position_sizing=True,
#     )
#     print_metrics("REALISTIC-COST (full rule)", real_metrics)

#     # ---- Decay: how much of the raw edge disappears under realistic frictions ----
#     raw_ret = raw_metrics["total_return_pct"]
#     real_ret = real_metrics["total_return_pct"]

#     if raw_ret != 0:
#         decay_pct = (raw_ret - real_ret) / raw_ret * 100
#         print(f"\n========== COST DECAY ==========")
#         print(f"Raw return            : {raw_ret:.2f}%")
#         print(f"Realistic-cost return : {real_ret:.2f}%")
#         print(f"Decay                 : {decay_pct:.2f}% of the raw edge lost to frictions")
#     else:
#         print("\nRaw return is 0% — decay percentage is undefined; report absolute gap instead.")
#         print(f"Absolute gap: {raw_ret - real_ret:.2f} percentage points")

#     # ---- Save realistic-cost results (this is the version Member B builds on) ----
#     real_df.to_csv("signals/backtest_results.csv", index=False)

#     # ---- Comparison plot: raw vs realistic portfolio value ----
#     plt.figure(figsize=(12, 6))
#     plt.plot(raw_df["Date"], raw_df["Portfolio_Value"], label="Raw / zero-cost", linestyle="--")
#     plt.plot(real_df["Date"], real_df["Portfolio_Value"], label="Realistic-cost", linewidth=2)
#     plt.axhline(INITIAL_CAPITAL, color="gray", linestyle=":")
#     plt.xlabel("Date")
#     plt.ylabel("Portfolio Value (₹)")
#     plt.title("DCWRNN Strategy: Raw vs Realistic-Cost Portfolio Value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("graphs/dcwrnn_portfolio_value_comparison.png", dpi=300)
#     plt.show()

#     print("\nResults saved to: signals/backtest_results.csv")
#     print("Graph saved to  : graphs/dcwrnn_portfolio_value_comparison.png")




import os
from config import BacktestConfig
from engine import load_signals, run_backtest, run_sensitivity_sweep, buy_and_hold_metrics
from reporting import (
    plot_equity_curves, plot_drawdown, plot_sensitivity_heatmap,
    metrics_to_dataframe, save_results_table,
)

os.makedirs("graphs", exist_ok=True)
os.makedirs("signals", exist_ok=True)
os.makedirs("tables", exist_ok=True)

signals = load_signals("signals/dcwrnn_signals.csv")

raw_cfg = BacktestConfig(cost_bps=0, slippage_bps=0, stop_loss=None, take_profit=None,
                          position_sizing=False, label="thesis_simple_zero_cost")
real_cfg = BacktestConfig(cost_bps=10, slippage_bps=5, stop_loss=0.02, take_profit=0.04,
                           position_sizing=True, label="realistic_cost_full_rule")

raw_df, raw_m = run_backtest(signals, raw_cfg)
real_df, real_m = run_backtest(signals, real_cfg)

# --- Safe decay calculation ---
raw_ret = raw_m["annual_return"]
real_ret = real_m["annual_return"]
gap_pp = (raw_ret - real_ret) * 100

print(f"\nRaw (thesis-simple) annual return : {raw_ret*100:.2f}%")
print(f"Realistic-cost annual return       : {real_ret*100:.2f}%")
print(f"Absolute gap                       : {gap_pp:.2f} percentage points")

if abs(raw_ret) > 0.01:
    decay = (raw_ret - real_ret) / raw_ret * 100
    print(f"Decay from frictions                : {decay:.2f}% of raw edge lost")
else:
    print("Decay %: not meaningful here — raw return is too close to zero "
          "(report the percentage-point gap above instead).")

# --- Directional accuracy diagnostics ---
traded = raw_df[raw_df["Signal"] != 0]
directional_accuracy = (traded["Signal"] * traded["Market_Return"] > 0).mean() * 100
pct_long = (raw_df["Signal"] == 1).mean() * 100
pct_short = (raw_df["Signal"] == -1).mean() * 100
pct_flat = (raw_df["Signal"] == 0).mean() * 100

print(f"\nDirectional accuracy (Signal vs. actual next-day move): {directional_accuracy:.2f}%")
print(f"Days long: {pct_long:.1f}% | short: {pct_short:.1f}% | flat: {pct_flat:.1f}%")

for direction, label in [(1, "Long"), (-1, "Short")]:
    subset = raw_df[raw_df["Signal"] == direction]
    if len(subset) > 0:
        acc = (subset["Signal"] * subset["Market_Return"] > 0).mean() * 100
        print(f"{label} accuracy ({len(subset)} days): {acc:.2f}%")

# --- Results table: raw vs. realistic-cost vs. buy-and-hold ---
bh_m = buy_and_hold_metrics(signals, real_cfg.initial_capital)
results_table = metrics_to_dataframe(raw_m, real_m, bh_m)
save_results_table(results_table, "tables/phase1_results.csv", "tables/phase1_results.tex",
                    caption="Phase 1 backtest results: raw vs. realistic-cost vs. buy-and-hold")

# --- Cost sensitivity sweep ---
sweep = run_sensitivity_sweep(
    signals,
    cost_bps_grid=[0, 5, 10, 15, 20, 30],
    slippage_bps_grid=[0, 5, 10, 15],
    base_cfg=real_cfg,
)
sweep.to_csv("tables/cost_sensitivity_sweep.csv", index=False)

# --- Plots ---
plot_equity_curves(raw_df, real_df, "graphs/equity_curve_comparison.png")
plot_drawdown(real_df, "graphs/drawdown_realistic.png")
plot_sensitivity_heatmap(sweep, "sharpe", "graphs/sensitivity_sharpe.png",
                          "Sharpe Ratio Sensitivity to Trading Frictions")
plot_sensitivity_heatmap(sweep, "annual_return", "graphs/sensitivity_return.png",
                          "Annualized Return Sensitivity to Trading Frictions")

real_df.to_csv("signals/backtest_results.csv", index=False)
print("\nAll figures saved to graphs/, tables saved to tables/.")