import cvxpy as cp
import numpy as np
import pandas as pd
from config import BacktestConfig
from metrics import compute_all_metrics


def optimize_positions(signals_df, cfg: BacktestConfig, max_position=1.0, max_turnover=0.5):
    df = signals_df.copy()
    df["Market_Return"] = df["Actual_Close"].pct_change()
    df["Forecast_Return"] = df["Predicted_Close"] / df["Actual_Close"].shift(1) - 1
    df = df.iloc[1:].reset_index(drop=True)

    n = len(df)
    signal = df["Forecast_Return"].values
    cost = (cfg.cost_bps + cfg.slippage_bps) / 10000

    w = cp.Variable(n)
    w_prev = cp.hstack([0, w[:-1]])
    turnover = cp.abs(w - w_prev)

    objective = cp.Maximize(signal @ w - cp.sum(turnover) * cost)
    constraints = [w <= max_position, w >= -max_position, turnover <= max_turnover]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    df["Position"] = w.value
    return df


def backtest_optimized(df, cfg: BacktestConfig):
    df = df.copy()
    df["Previous_Position"] = df["Position"].shift(1).fillna(0)
    df["Turnover"] = (df["Position"] - df["Previous_Position"]).abs()
    cost = cfg.cost_bps / 10000
    slip = cfg.slippage_bps / 10000
    df["Transaction_Cost"] = df["Turnover"] * cost
    df["Slippage_Cost"] = df["Turnover"] * slip
    df["Net_Return"] = (
        df["Position"] * df["Market_Return"]
        - df["Transaction_Cost"] - df["Slippage_Cost"]
    )
    df.loc[df.index[0], "Net_Return"] = 0.0
    df["Portfolio_Value"] = cfg.initial_capital * (1 + df["Net_Return"].fillna(0)).cumprod()
    m = compute_all_metrics(df, cfg.cost_bps, cfg.slippage_bps)
    m["label"] = "convex_optimized"
    return df, m