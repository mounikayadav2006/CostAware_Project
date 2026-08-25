import cvxpy as cp


def optimize_position(
    expected_return,
    previous_position,
    transaction_cost=0.001,
    position_limit=1.0,
    turnover_limit=0.20
):
    """
    Cost-aware convex portfolio optimizer.

    Parameters:
        expected_return: Expected return from the DCWRNN prediction.
        previous_position: Position held on the previous trading day.
        transaction_cost: Cost per unit of position turnover.
        position_limit: Maximum absolute portfolio position.
        turnover_limit: Maximum allowed daily position change.

    Returns:
        Optimized portfolio position.
    """

    # Decision variable
    position = cp.Variable()

    # Turnover
    turnover = cp.abs(position - previous_position)

    # Objective:
    # maximize expected return while penalizing trading costs
    objective = cp.Maximize(
        expected_return * position
        - transaction_cost * turnover
    )

    # Constraints
    constraints = [
        position >= -position_limit,
        position <= position_limit,
        turnover <= turnover_limit
    ]

    # Create optimization problem
    problem = cp.Problem(objective, constraints)

    # Solve
    problem.solve(solver=cp.CLARABEL)

    # If solver fails, keep previous position
    if position.value is None:
        return previous_position

    return float(position.value)