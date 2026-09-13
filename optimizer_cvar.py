import cvxpy as cp
import numpy as np
import pandas as pd
from config import BacktestConfig
from metrics import compute_all_metrics


def optimize_positions_cvar(signals_df, cfg: BacktestConfig, alpha=0.95, cvar_limit=0.006,
                             max_position=1.0, max_turnover=0.5):
    df = signals_df.copy()
    df["Market_Return"] = df["Actual_Close"].pct_change()
    df["Forecast_Return"] = df["Predicted_Close"] / df["Actual_Close"].shift(1) - 1
    df = df.iloc[1:].reset_index(drop=True)

    n = len(df)
    signal = df["Forecast_Return"].values
    cost = (cfg.cost_bps + cfg.slippage_bps) / 10000

    # Trailing realized volatility (no look-ahead — only uses PAST data at each day)
    rolling_vol = df["Market_Return"].rolling(window=20, min_periods=5).std().shift(1)
    rolling_vol = rolling_vol.fillna(df["Market_Return"].std())
    risk_proxy = rolling_vol.values

    w = cp.Variable(n)
    w_prev = cp.hstack([0, w[:-1]])
    turnover = cp.abs(w - w_prev)

    # Rockafellar-Uryasev CVaR linearization
    zeta = cp.Variable()              # VaR threshold
    z = cp.Variable(n)                # auxiliary loss variables
    port_return = cp.multiply(w, -risk_proxy)   # trailing-vol-scaled risk proxy

    cvar_expr = zeta + cp.sum(z) / ((1 - alpha) * n)

    objective = cp.Maximize(signal @ w - cp.sum(turnover) * cost)
    constraints = [
        w <= max_position, w >= -max_position,
        turnover <= max_turnover,
        z >= 0,
        z >= -port_return - zeta,
        cvar_expr <= cvar_limit,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        raise RuntimeError("CVaR optimizer did not converge — try relaxing cvar_limit or max_turnover")

    df["Position"] = w.value
    return df


def backtest_cvar(df, cfg: BacktestConfig):
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
    m["label"] = "cvar_constrained"
    return df, m