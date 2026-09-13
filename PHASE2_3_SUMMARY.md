# Phase 2 & 3 Summary — Member B (Anjali)

## Phase 2: Convex Position Optimizer

**Objective:** Replace Member A's naive threshold-based position sizing (100% long/short based on forecast direction) with a convex-optimized position size, maximizing expected return net of transaction cost.

**Method:** Formulated as a single convex optimization problem in `cvxpy` across the full ~300-day backtest period, using the DCWRNN forecast signal as the return estimate, with turnover-based cost penalty, position caps, and turnover limits.

**Result:**

| Metric | Naive | Convex Optimized |
|---|---|---|
| Total Return | -1.99% | +5.64% |
| Sharpe | -0.26 | +0.48 |
| Sortino | -0.33 | +0.68 |
| Calmar | -0.21 | +0.63 |
| Max Drawdown | 7.79% | 7.48% |

**Milestone met:** Optimized sizing beats naive on net-of-cost return, across every risk-adjusted metric.

**Robustness check:** Cost-sensitivity sweep across 5–25 bps cost/slippage assumptions confirms the edge holds up comfortably under realistic Nifty 50 trading costs, only turning negative under pessimistic extreme assumptions.

---

## Phase 3: CVaR Tail-Risk Constraint

**Objective:** Add a Rockafellar-Uryasev CVaR constraint to control tail risk, addressing a real ~8% Nifty decline present in our test data (Feb 27 – Mar 13, 2026).

**Iteration 1 (forecast-based CVaR):** Constrained the model's own forecast-return tail risk. Result: return dropped, but drawdown did **not** improve — because the model's forecast errors, not just position sizing, drive real tail losses. Documented as a legitimate limitation.

**Iteration 2 (realized-volatility-based CVaR — final version):** Constrained tail risk using a 20-day rolling window of *past* realized market volatility (no look-ahead bias — only uses information available at each decision point). This successfully anticipates volatility clustering ahead of real market shocks.

**Result:**

| Metric | Convex Optimized (Phase 2) | CVaR-Constrained (Phase 3) |
|---|---|---|
| Total Return | 5.64% | **6.01%** |
| Sharpe | 0.48 | **0.60** |
| Sortino | 0.68 | **0.87** |
| Calmar | 0.63 | **1.06** |
| Max Drawdown | 7.48% | **4.76%** |
| Volatility | 10.90% | **8.89%** |

**Milestone met:** Lower max drawdown (36% reduction) at better Sharpe — exceeds the plan's success criterion.

**Key evidence — drawdown chart:** At both major crash points in the test data (April 2025 and March 2026), CVaR-constrained drawdown stayed around -4.3%, while the unconstrained optimizer fell to roughly -7%.

**Key finding to highlight:** A CVaR constraint is only as protective as the risk signal it's built on. Constraining forecast-based risk did not translate into real protection since the model's forecast errors are themselves a source of tail risk. Switching to realized historical volatility — a legitimate, leak-free proxy for market turbulence — successfully anticipated and reduced real drawdowns without sacrificing return.

---

## Files
- `optimizer.py`, `run_optimizer.py`, `make_phase2_plots.py` — Phase 2
- `optimizer_cvar.py`, `run_optimizer_cvar.py`, `make_phase3_plots.py` — Phase 3
- `outputs/phase2_naive_vs_optimized.csv`, `outputs/phase3_three_way_comparison.csv`
- `graphs/phase2_*.png`, `graphs/phase3_*.png`