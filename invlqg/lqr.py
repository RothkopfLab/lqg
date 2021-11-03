from jax import numpy as jnp

from invlqg.riccati import solve_discrete_riccati


def control_law(A, B, Q, R, T):
    S = solve_discrete_riccati(A, B, Q, R, T)

    L = jnp.linalg.inv(B.T @ S @ B + R) @ B.T @ S @ A

    return L
