from jax import numpy as jnp, lax

from lqg.spec import LQGSpec


def forward(spec: LQGSpec, Sigma0: jnp.ndarray) -> jnp.ndarray:
    def loop(P, step):
        A, F, V, W = step

        P = A @ P @ A.T + V @ V.T
        G = F @ P @ F.T + W @ W.T
        K = P @ F.T @ jnp.linalg.inv(G)
        
        P = (jnp.eye(P.shape[0]) - K @ F) @ P

        return P, K

    _, K = lax.scan(loop, Sigma0,
                    (spec.A, spec.F, spec.V, spec.W))

    return K
