from jax import numpy as jnp

from lqg.riccati import discrete_riccati


def control_law(A, B, Q, R, T):
    S = discrete_riccati(A, B, Q, R, T)

    L = jnp.linalg.inv(B.T @ S @ B + R) @ B.T @ S @ A

    return L