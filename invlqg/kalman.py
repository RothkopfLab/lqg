import numpyro.distributions as dist
import jax.numpy as jnp
from jax.lax import scan
import numpy as np

from invlqg.riccati import solve_discrete_riccati


def kalman_gain(A, C, V, W, T):
    P = solve_discrete_riccati(A.T, C.T, V, W, T)

    S = C @ P @ C.T + W
    K = P @ C.T @ jnp.linalg.inv(S)

    return K


class KalmanFilter:
    def __init__(self, A, C, V, W):
        self.A = A
        self.C = C
        self.V = V
        self.W = W

    @property
    def xdim(self):
        return self.A.shape[0] * 2

    def K(self, T):
        K = kalman_gain(self.A, self.C, self.V @ self.V.T, self.W @ self.W.T, T=T)
        return K

    def simulate(self, n=1, T=100, seed=0):
        np.random.seed(seed)

        V = self.V
        W = self.W
        K = self.K(T)

        xs = []
        xhats = []

        xi = jnp.zeros((n, self.A.shape[0]))
        xhat = jnp.zeros((n, self.A.shape[0]))

        for t in range(T):
            xi = xi @ self.A.T + np.random.normal(size=(n, self.A.shape[0])) @ V.T

            yi = xi @ self.C.T + np.random.normal(size=(n, self.C.shape[0])) @ W.T

            xhat = xhat @ self.A.T + (yi - xhat @ self.A.T @ self.C.T) @ K.T

            xs.append(xi)
            xhats.append(xhat)

        x = jnp.stack(xs)
        xhat = jnp.stack(xhats)

        x = jnp.concatenate([x, xhat], axis=2)

        return jnp.concatenate([x[..., ::2], x[..., 1::2]], axis=2)

    def simulate_given_x(self, x, seed=0, x0=None, xhat0=None):
        T, n = x.shape

        np.random.seed(seed)

        A = np.array(self.A)

        V = np.array(self.V)
        W = np.array(self.W)
        K = np.array(self.K(T))

        xs = []
        xhats = []

        xi = np.zeros((n, A.shape[0])) if x0 is None else x0
        xhat = np.zeros((n, A.shape[0])) if xhat0 is None else xhat0

        xs.append(xi)
        xhats.append(xhat)

        for t in range(1, T):
            xi = xi @ A.T + np.random.normal(size=(n, A.shape[0])) @ V.T
            xi[:, 0] = x[t]

            yi = xi @ self.C.T + np.random.normal(size=(n, self.C.shape[0])) @ W.T

            xhat = xhat @ self.A.T + (yi - xhat @ self.A.T @ self.C.T) @ K.T

            xs.append(xi)
            xhats.append(xhat)

        x = jnp.stack(xs)
        xhat = jnp.stack(xhats)

        x = jnp.concatenate([x, xhat], axis=2)

        return jnp.concatenate([x[..., ::2], x[..., 1::2]], axis=2)

    def conditional_distribution(self, x):
        T, n, d = x.shape
        idx = list(range(d))
        swapdim = idx[::2] + idx[1::2]

        V = self.V
        W = self.W

        K = self.K(T)

        F = jnp.vstack(
            [jnp.hstack([self.A, jnp.zeros_like(self.A)]),
             jnp.hstack([K @ self.C @ self.A, self.A - K @ self.C @ self.A])])

        F = F[:, swapdim]
        F = F[swapdim, :]

        # TODO: this throws errors for the model with velocities..
        G = jnp.vstack([jnp.hstack([V, jnp.zeros_like(V)]), jnp.hstack([K @ self.C @ V, K @ W])])

        G = G[:, swapdim]
        G = G[swapdim, :]

        mu = jnp.zeros((n, d))
        Sigma = G @ G.T

        def f(carry, xt):
            mu, Sigma = carry

            mu = mu @ F.T + (xt - mu) @ jnp.linalg.inv(Sigma).T @ (F @ Sigma).T

            Sigma = F @ Sigma @ F.T + G @ G.T - (F @ Sigma) @ jnp.linalg.inv(Sigma) @ (Sigma @ F.T)
            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(f, (mu, Sigma), x)

        return dist.MultivariateNormal(mu.transpose((1, 0, 2)), Sigma)

    def log_likelihood(self, x):
        return self.conditional_distribution(x[:-1]).log_prob(x[1:].transpose((1, 0, 2)))
