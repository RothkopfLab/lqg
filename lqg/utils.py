from jax import vmap, numpy as jnp

from lqg.spec import LQGSpec

# quadratic_form takes arrays A (n, d, m) and B (n, n) and computes A[:, i, :].T @ B @ A[:, i, :]
# for all i along A's second axis in a vectorized way and returns an array (d, m, m)
quadratic_form = vmap(lambda A, B: A.T @ B @ A, in_axes=(1, None))

# quadratic_form_t takes arrays A (m, d, n) and B (n, n) and computes A[:, i, :] @ B @ A[:, i, :].T
# for all i along A's second axis in a vectorized way and returns an array (d, m, m)
quadratic_form_t = vmap(lambda A, B: A @ B @ A.T, in_axes=(1, None))

# bilinear_form takes arrays A (n, d, m), B (n, n) and C (n, d, o)
# and computes A[:, i, :].T @ B @ C[:, i, :]
# for all i along A and C's second axis in a vectorized way and returns an array (d, m, o)
bilinear_form = vmap(lambda A, B, C: A.T @ B @ C, in_axes=(1, None, 1))

# bilinear_form_t takes arrays A (m, d, n), B (n, n) and C (o, d, n)
# and computes A[:, i, :] @ B @ C[:, i, :].T
# for all i along A and C's second axis in a vectorized way and returns an array (d, m, o)
bilinear_form_t = vmap(lambda A, B, C: A @ B @ C.T, in_axes=(1, None, 1))


def time_stack(A: jnp.ndarray, T: int):
    return jnp.stack((A,) * T)


def time_stack_spec(A: jnp.ndarray,
                    B: jnp.ndarray,
                    F: jnp.ndarray,
                    V: jnp.ndarray,
                    W: jnp.ndarray,
                    Q: jnp.ndarray,
                    R: jnp.ndarray,
                    T: int) -> LQGSpec:
    A = time_stack(A, T)
    B = time_stack(B, T)
    F = time_stack(F, T)
    V = time_stack(V, T)
    W = time_stack(W, T)
    Q = time_stack(Q, T)
    R = time_stack(R, T)

    state_dim = Q.shape[1]
    action_dim = R.shape[1]
    obs_dim = W.shape[1]

    q = jnp.zeros((T, state_dim))
    Qf = Q[-1]
    qf = q[-1]
    P = jnp.zeros((T, action_dim, state_dim))
    r = jnp.zeros((T, action_dim))
    Cx = jnp.zeros((T, state_dim, V.shape[-1], state_dim))
    Cu = jnp.zeros((T, state_dim, V.shape[-1], action_dim))
    D = jnp.zeros((T, obs_dim, W.shape[-1], state_dim))

    return LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, q=q, Qf=Qf, qf=qf, P=P, r=r, Cx=Cx, Cu=Cu, D=D)
