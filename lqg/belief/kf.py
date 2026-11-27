from jax import numpy as jnp, lax

from lqg.spec import LQGSpec


def forward(spec: LQGSpec, Sigma0: jnp.ndarray) -> jnp.ndarray:
    """Compute Kalman gains with a forward pass.

    The belief is then updated as

    ```
    x_pred    = A[t] xhat[t] + B[t] u[t]
    xhat[t+1] = x_pred + K[t] (y[t+1] - F[t] x_pred)
    ```

    Args:
        spec (LQGSpec): problem specification, of which ``A``, ``F``, ``V``
            and ``W`` are used
        Sigma0 (jnp.ndarray): initial covariance of the state estimate, of
            shape (xdim, xdim)

    Returns:
        jnp.ndarray: Kalman gains of shape (T, xdim, ydim)
    """

    def loop(P, step):
        A, F, V, W = step

        P = A @ P @ A.T + V @ V.T
        G = F @ P @ F.T + W @ W.T
        K = P @ F.T @ jnp.linalg.inv(G)

        P = (jnp.eye(P.shape[0]) - K @ F) @ P

        return P, K

    _, K = lax.scan(loop, Sigma0, (spec.A, spec.F, spec.V, spec.W))

    return K
