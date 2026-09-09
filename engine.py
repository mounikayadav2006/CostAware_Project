import numpy as np
import pandas as pd
from config import BacktestConfig
from metrics import compute_all_metrics, annualized_return, max_drawdown, annualized_volatility, sharpe_ratio


def load_signals(path="signals/dcwrnn_signals.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    assert df["Actual_Close"].notna().all(), "NaNs found in Actual_Close"
    assert df["Predicted_Close"].notna().all(), "NaNs found in Predicted_Close"
    return df


def run_backtest(signals_df, cfg: BacktestConfig):
    df = signals_df.copy()
    cost = cfg.cost_bps / 10000
    slip = cfg.slippage_bps / 10000

    df["Market_Return"] = df["Actual_Close"].pct_change()
    df["Forecast_Return"] = df["Predicted_Close"] / df["Actual_Close"].shift(1) - 1

    df["Signal"] = 0
    df.loc[df["Forecast_Return"] > 0, "Signal"] = 1
    df.loc[df["Forecast_Return"] < 0, "Signal"] = -1
    df.loc[0, "Signal"] = 0

    df["Strategy_Return"] = df["Signal"] * df["Market_Return"]

    lower = -cfg.stop_loss if cfg.stop_loss is not None else -np.inf
    upper = cfg.take_profit if cfg.take_profit is not None else np.inf
    df["Protected_Return"] = df["Strategy_Return"].clip(lower=lower, upper=upper)

    df["Prediction_Strength"] = df["Forecast_Return"].abs()
    if cfg.position_sizing:
        df["Position_Size"] = 0.2
        df.loc[df["Prediction_Strength"] >= 0.005, "Position_Size"] = 0.5
        df.loc[df["Prediction_Strength"] >= 0.01, "Position_Size"] = 1.0
    else:
        df["Position_Size"] = 1.0
    df.loc[0, "Position_Size"] = 0.0

    df["Position"] = df["Signal"] * df["Position_Size"]
    df["Previous_Position"] = df["Position"].shift(1).fillna(0)
    df["Turnover"] = (df["Position"] - df["Previous_Position"]).abs()

    df["Transaction_Cost"] = df["Turnover"] * cost
    df["Slippage_Cost"] = df["Turnover"] * slip

    df["Net_Return"] = (
        df["Protected_Return"] * df["Position_Size"]
        - df["Transaction_Cost"]
        - df["Slippage_Cost"]
    )
    df.loc[0, "Net_Return"] = 0.0

    df["Portfolio_Value"] = cfg.initial_capital * (1 + df["Net_Return"].fillna(0)).cumprod()

    metrics = compute_all_metrics(df, cfg.cost_bps, cfg.slippage_bps)
    metrics["label"] = cfg.label
    return df, metrics


def run_sensitivity_sweep(signals_df, cost_bps_grid, slippage_bps_grid, base_cfg: BacktestConfig):
    rows = []
    for c in cost_bps_grid:
        for s in slippage_bps_grid:
            cfg = BacktestConfig(
                cost_bps=c, slippage_bps=s,
                stop_loss=base_cfg.stop_loss, take_profit=base_cfg.take_profit,
                position_sizing=base_cfg.position_sizing,
                initial_capital=base_cfg.initial_capital,
                label=f"c{c}_s{s}",
            )
            _, m = run_backtest(signals_df, cfg)
            rows.append(m)
    return pd.DataFrame(rows)


def buy_and_hold_metrics(signals_df, initial_capital):
    df = signals_df.copy()
    df["Portfolio_Value"] = initial_capital * (df["Actual_Close"] / df["Actual_Close"].iloc[0])
    df["Net_Return"] = df["Actual_Close"].pct_change().fillna(0)
    return {
        "label": "buy_and_hold",
        "cost_bps": 0, "slippage_bps": 0,
        "n_days": len(df) - 1,
        "final_portfolio_value": df["Portfolio_Value"].iloc[-1],
        "total_return_pct": (df["Portfolio_Value"].iloc[-1] / initial_capital - 1) * 100,
        "annual_return": annualized_return(df["Portfolio_Value"]),
        "max_drawdown": max_drawdown(df["Portfolio_Value"]),
        "annualized_volatility": annualized_volatility(df["Net_Return"]),
        "sharpe": sharpe_ratio(df["Net_Return"]),
        "sortino": None, "calmar": None, "omega": None, "win_rate_pct": None,
    }