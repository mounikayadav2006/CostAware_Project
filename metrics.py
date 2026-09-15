import numpy as np
from config import TRADING_DAYS_PER_YEAR


def annualized_return(portfolio_values):
    n = len(portfolio_values) - 1
    if n <= 0:
        return 0.0
    total_growth = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
    if total_growth <= 0:
        return -1.0
    return total_growth ** (TRADING_DAYS_PER_YEAR / n) - 1


def max_drawdown(portfolio_values):
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max
    return abs(drawdown.min())  # positive magnitude


def annualized_volatility(returns):
    std = returns.std()
    return 0.0 if np.isnan(std) else std * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns, risk_free=0.0):
    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns, threshold=0.0):
    downside = returns[returns < threshold]
    downside_dev = downside.std()
    if downside_dev == 0 or np.isnan(downside_dev):
        return 0.0
    excess_mean = returns.mean() - threshold
    return (excess_mean / downside_dev) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calmar_ratio(returns, portfolio_values):
    ann_ret = annualized_return(portfolio_values)
    mdd = max_drawdown(portfolio_values)
    return 0.0 if mdd == 0 else ann_ret / mdd


def omega_ratio(returns, threshold=0.0):
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    total_losses = losses.sum()
    if total_losses == 0:
        return np.inf if gains.sum() > 0 else 0.0
    return gains.sum() / total_losses


def win_rate(returns, positions):
    """Only counts days where an actual position was held (Position != 0)."""
    traded = returns[positions != 0]
    return 0.0 if len(traded) == 0 else (traded > 0).mean()


def turnover(positions):
    """
    Average daily turnover: mean absolute change in position size.
    Higher turnover = more trading activity = more real-world cost.
    """
    position_changes = positions.diff().abs()
    return position_changes.mean()


def compute_all_metrics(df, cost_bps, slippage_bps):
    returns = df["Net_Return"].iloc[1:]  # drop row 0 (no signal yet)
    portfolio_values = df["Portfolio_Value"]
    positions = df["Position"].iloc[1:]

    return {
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "n_days": len(df) - 1,
        "final_portfolio_value": portfolio_values.iloc[-1],
        "total_return_pct": (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100,
        "annual_return": annualized_return(portfolio_values),
        "max_drawdown": max_drawdown(portfolio_values),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns, portfolio_values),
        "omega": omega_ratio(returns),
        "win_rate_pct": win_rate(returns, positions) * 100,
        "turnover": turnover(positions),
    }