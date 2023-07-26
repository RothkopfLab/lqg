from typing import NamedTuple

from jax import numpy as jnp, lax

from lqg.spec import LQGSpec


class Gains(NamedTuple):
    """LQR control gains"""

    L: jnp.ndarray
    l: jnp.ndarray
    H: jnp.ndarray = None


def backward(spec: LQGSpec, eps: float = 1e-8) -> Gains:
    def loop(carry, step):
        S, s = carry

        Q, q, P, R, r, A, B = step

        H = R + B.T @ S @ B
        G = P + B.T @ S @ A
        g = r + B.T @ s

        # Deal with negative eigenvals of H, see section 5.4.1 of Li's PhD thesis
        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        S = Q + A.T @ S @ A + L.T @ H @ L + L.T @ G + G.T @ L
        s = q + A.T @ s + G.T @ l + L.T @ H @ l + L.T @ g

        return (S, s), (L, l, Ht)

    _, (L, l, H) = lax.scan(loop, (spec.Qf, spec.qf),
                            (spec.Q, spec.q, spec.P, spec.R, spec.r, spec.A, spec.B),
                            reverse=True)

    return Gains(L=L, l=l, H=H)
