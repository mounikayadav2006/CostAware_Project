import cvxpy as cp
import numpy as np

# Example historical/scenario returns
returns = np.array([
    0.02,
    0.01,
    -0.01,
    0.015,
    -0.02,
    -0.03,
    0.01,
    -0.05,
    0.005,
    -0.08
])

# Confidence level
alpha = 0.95

# Portfolio position
w = cp.Variable()

# VaR threshold
z = cp.Variable()

# Excess loss above VaR
u = cp.Variable(len(returns), nonneg=True)

# Loss for each scenario
loss = -returns * w

# Rockafellar-Uryasev CVaR formulation
constraints = [
    u >= loss - z,
    w >= -1,
    w <= 1
]

cvar = z + (1 / ((1 - alpha) * len(returns))) * cp.sum(u)

# Minimize CVaR
problem = cp.Problem(
    cp.Minimize(cvar),
    constraints
)

problem.solve(solver="CLARABEL")

print("========== CVaR TEST ==========")
print(f"Status       : {problem.status}")
print(f"Optimal Position : {w.value:.4f}")
print(f"VaR (z)      : {z.value:.4f}")
print(f"CVaR 95%     : {cvar.value:.4f}")