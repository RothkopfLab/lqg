from jax import lax, numpy as jnp

from lqg.control import lqr
from lqg.spec import LQGSpec
from lqg.utils import quadratic_form, bilinear_form


def backward(spec: LQGSpec, eps: float = 1e-8) -> lqr.Gains:
    def loop(carry, step):
        S, s = carry

        Q, q, P, R, r, A, B, V, Cx, Cu = step

        H = R + B.T @ S @ B + quadratic_form(Cu, S).sum(axis=0)
        G = P + B.T @ S @ A + bilinear_form(Cu, S, Cx).sum(axis=0)
        g = r + B.T @ s + bilinear_form(Cu, S, V).sum(axis=0)

        # Deal with negative eigenvals of H, see section 5.4.1 of Li's PhD thesis
        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        Sn = Q + A.T @ S @ A + L.T @ H @ L + L.T @ G + G.T @ L + quadratic_form(Cx, S).sum(axis=0)
        sn = q + A.T @ s + G.T @ l + L.T @ H @ l + L.T @ g + bilinear_form(Cx, S, V).sum(axis=0)

        return (Sn, sn), (L, l, Ht)

    _, (L, l, H) = lax.scan(loop, (spec.Qf, spec.qf),
                            (spec.Q, spec.q, spec.P, spec.R, spec.r, spec.A, spec.B, spec.V, spec.Cx, spec.Cu),
                            reverse=True)

    return lqr.Gains(L=L, l=l, H=H)
