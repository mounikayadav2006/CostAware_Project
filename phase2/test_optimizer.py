from optimizer import optimize_position


previous_position = 0.0

expected_returns = [
    0.01,
    0.008,
    -0.005,
    -0.012,
    0.015
]

for expected_return in expected_returns:

    position = optimize_position(
        expected_return=expected_return,
        previous_position=previous_position
    )

    print(
        f"Expected Return: {expected_return:.4f} "
        f"→ Position: {position:.4f}"
    )

    previous_position = position
    