from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 100_000

@dataclass
class BacktestConfig:
    cost_bps: float = 10.0        # transaction cost, basis points (10 bps = 0.10%)
    slippage_bps: float = 5.0     # slippage, basis points
    stop_loss: float | None = 0.02
    take_profit: float | None = 0.04
    position_sizing: bool = True  # False => always full size (thesis-simple rule)
    initial_capital: float = INITIAL_CAPITAL
    label: str = "run"