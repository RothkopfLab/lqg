import jax.numpy as jnp
# from fax import implicit
from jax.lax import scan


# def solve_dare(A, B, Q, R):
#     def _make_riccati_operator(params):
#         A, B, Q, R = params
#
#         def _riccati_operator(P):
#             X = R + B.T @ P.T @ B
#             Y = B.T @ P @ A
#             return (A.T @ P @ A) - ((A.T @ P @ B) @ jnp.linalg.solve(X, Y)) + Q
#
#         return _riccati_operator
#
#     solution = implicit.two_phase_solve(_make_riccati_operator, jnp.eye(A.shape[0]), (A, B, Q, R))
#     return solution


def discrete_riccati(A, B, Q, R, T):
    def riccati_iter(S, t):
        S = A.T @ (S - S @ B @ jnp.linalg.inv(B.T @ S @ B + R) @ B.T @ S) @ A + Q
        return S, S

    S, _ = scan(riccati_iter, Q, jnp.arange(T))

    # alternative method, does not seem to be faster though
    # S = solve_dare(A, B, Q, R)

    return S