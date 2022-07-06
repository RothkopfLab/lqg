import jax.numpy as jnp
from jax.lax import scan


def solve_discrete_riccati(A, B, Q, R, T):
    def riccati_iter(S, t):
        S = A.T @ (S - S @ B @ jnp.linalg.inv(B.T @ S @ B + R) @ B.T @ S) @ A + Q
        return S, S

    S, _ = scan(riccati_iter, Q, jnp.arange(T))

    return S
