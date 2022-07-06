import jax.numpy as jnp
from jax import numpy as jnp
from jax.lax import scan


def solve_discrete_riccati(A, B, Q, R, T):
    def riccati_iter(S, t):
        S = A.T @ (S - S @ B @ jnp.linalg.inv(B.T @ S @ B + R) @ B.T @ S) @ A + Q
        return S, S

    S, _ = scan(riccati_iter, Q, jnp.arange(T))

    return S


def kalman_gain(A, C, V, W, T):
    """ Compute Kalman gain after T time steps

    Args:
        A (jnp.array): state transition matrix
        C (jnp.array): observation matrix
        V (jnp.array): state transition noise covariance
        W (jnp.array): observation noise covariance
        T (int): number of time steps

    Returns:
        jnp.array: Kalman gain (final time step)
    """
    P = solve_discrete_riccati(A.T, C.T, V, W, T)

    S = C @ P @ C.T + W
    K = P @ C.T @ jnp.linalg.inv(S)

    return K


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
