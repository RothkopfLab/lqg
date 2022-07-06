from jax import numpy as jnp

from lqg.riccati import solve_discrete_riccati


def control_law(A, B, Q, R, T):
    """ LQR control law after T time steps

    Args:
        A (jnp.array): state transition matrix
        B (jnp.array): control matrix
        Q (jnp.array): control costs
        R (jnp.array): action costs
        T (jnp.array): time steps

    Returns:
        jnp.array: LQR controller
    """
    S = solve_discrete_riccati(A, B, Q, R, T)

    L = jnp.linalg.inv(B.T @ S @ B + R) @ B.T @ S @ A

    return L
