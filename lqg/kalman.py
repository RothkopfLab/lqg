import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

from lqg.belief import kf
from lqg.spec import LQGSpec


class KalmanFilter:
    def __init__(self, dynamics: LQGSpec):
        self.dynamics = dynamics

    @property
    def T(self):
        """ Length of trajectory

        Returns:
            int: number of time steps
        """
        return self.dynamics.A.shape[0]

    @property
    def xdim(self):
        return self.dynamics.A.shape[1] * 2

    def simulate(self, rng_key, n=1, x0=None, xhat0=None):
        """ Simulate n trials

        Args:
            rng_key (jax.random.PRNGKey): random number generator key
            n (int): number of trials
            T (int): number of time steps
            x0 (jnp.array): initial state
            xhat0 (jnp.array): initial belief

        Returns:
            jnp.array (T, n, d)
        """

        # compute Kalman gain
        K = kf.forward(self.dynamics, Sigma0=self.dynamics.V[0] @ self.dynamics.V[0].T)

        def simulate_trial(rng_key, x0=None, xhat0=None):
            """ Simulate a single trial

            Args:
                rng_key (jax.random.PRNGKey): random number generator key
                T (int): number of time steps
                x0 (jnp.array): initial state
                xhat0 (jnp.array): initial belief

            Returns:
                jnp.array, jnp.array, jnp.array, jnp.array: x (states), x_hat (estimates), y, u
            """

            x0 = jnp.zeros(self.dynamics.A.shape[1]) if x0 is None else x0
            xhat0 = jnp.zeros(self.dynamics.A.shape[1]) if xhat0 is None else xhat0

            # generate standard normal noise terms
            rng_key, subkey = random.split(rng_key)
            epsilon = random.normal(subkey, shape=(self.T, x0.shape[0]))
            rng_key, subkey = random.split(rng_key)
            eta = random.normal(subkey, shape=(self.T, x0.shape[0]))

            def loop(carry, t):
                x, x_hat = carry

                # generate observation
                y = self.dynamics.F[t] @ x + self.dynamics.W[t] @ eta[t]

                # update agent's belief
                x_pred = self.dynamics.A[t] @ x_hat
                x_hat = x_pred + K[t] @ (y - self.dynamics.F[t] @ x_pred)

                # apply dynamics
                x = self.dynamics.A[t] @ x + self.dynamics.V[t] @ epsilon[t]

                return (x, x_hat), (x, x_hat, y)

            _, (x, x_hat, y) = scan(loop, (x0, xhat0), jnp.arange(1, self.T))

            x = jnp.vstack((x0, x))
            x_hat = jnp.vstack((xhat0, x_hat))

            # stack up state and estimate
            # so that the data have the same structure as the control models
            x = jnp.hstack((x, x_hat))

            # rearrange dimension so that we always have the state
            # and then the corresponding estimate (interleaved)
            return jnp.hstack([x[..., ::2], x[..., 1::2]])

        # simulate n trials
        x = vmap(lambda key: simulate_trial(key, x0=x0, xhat0=xhat0))(random.split(rng_key, num=n))

        return x

    def conditional_moments(self, x):
        T, d = x.shape
        idx = list(range(d))
        swapdim = idx[::2] + idx[1::2]

        K = kf.forward(self.dynamics, Sigma0=self.dynamics.V[0] @ self.dynamics.V[0].T)

        # joint dynamics
        F = jnp.concatenate([
            jnp.concatenate([self.dynamics.A,
                             jnp.zeros_like(self.dynamics.A)], axis=-1),
            jnp.concatenate([K @ self.dynamics.F,
                             self.dynamics.A - K @ self.dynamics.F], axis=-1)],
            axis=-2)

        F = F[:, :, swapdim]
        F = F[:, swapdim, :]

        # joint noise covariance Cholesky factor
        G = jnp.concatenate([
            jnp.concatenate([self.dynamics.V,
                             jnp.zeros_like(self.dynamics.V)], axis=-1),
            jnp.concatenate([jnp.zeros_like(self.dynamics.A),
                             K @ self.dynamics.W], axis=-1)],
            axis=-2)

        G = G[:, :, swapdim]
        G = G[:, swapdim, :]

        mu = x[0]
        Sigma = G[0] @ G[0].T

        def scan_fn(carry, step):
            mu, Sigma = carry
            F, G, x = step

            # conditioning and marginalizing
            mu = F @ mu + (F @ Sigma) @ jnp.linalg.solve(Sigma, x - mu)

            Sigma = F @ Sigma @ F.T + G @ G.T - (F @ Sigma) @ jnp.linalg.solve(Sigma, Sigma @ F.T)
            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(scan_fn, (mu, Sigma), (F, G, x))

        return mu, Sigma

    def conditional_distribution(self, x):
        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = vmap(self.conditional_moments)(x)

        return dist.MultivariateNormal(mu, Sigma)

    def log_likelihood(self, x):
        return self.conditional_distribution(x[:-1]).log_prob(x[1:])
