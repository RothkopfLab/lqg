import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

from lqg.riccati import kalman_gain
from lqg.model import Dynamics


class KalmanFilter:
    def __init__(self, dynamics: Dynamics):
        self.dynamics = dynamics

    @property
    def xdim(self):
        return self.dynamics.A.shape[0] * 2

    def K(self, T):
        K = kalman_gain(self.dynamics.A, self.dynamics.C,
                        self.dynamics.V @ self.dynamics.V.T,
                        self.dynamics.W @ self.dynamics.W.T, T=T)
        return K

    def simulate(self, rng_key, n=1, T=100, x0=None, xhat0=None):
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
        K = self.K(T)

        def simulate_trial(rng_key, T=100, x0=None, xhat0=None):
            """ Simulate a single trial

            Args:
                rng_key (jax.random.PRNGKey): random number generator key
                T (int): number of time steps
                x0 (jnp.array): initial state
                xhat0 (jnp.array): initial belief

            Returns:
                jnp.array, jnp.array, jnp.array, jnp.array: x (states), x_hat (estimates), y, u
            """

            x0 = jnp.zeros(self.dynamics.A.shape[0]) if x0 is None else x0
            xhat0 = jnp.zeros(self.dynamics.A.shape[0]) if xhat0 is None else xhat0

            # generate standard normal noise terms
            rng_key, subkey = random.split(rng_key)
            epsilon = random.normal(subkey, shape=(T, x0.shape[0]))
            rng_key, subkey = random.split(rng_key)
            eta = random.normal(subkey, shape=(T, x0.shape[0]))

            def loop(carry, t):
                x, x_hat = carry

                # apply dynamics
                x = self.dynamics.A @ x + self.dynamics.V @ epsilon[t]

                # generate observation
                y = self.dynamics.C @ x + self.dynamics.W @ eta[t]

                # update agent's belief
                x_pred = self.dynamics.A @ x_hat
                x_hat = x_pred + K @ (y - self.dynamics.C @ x_pred)

                return (x, x_hat), (x, x_hat, y)

            _, (x, x_hat, y) = scan(loop, (x0, xhat0), jnp.arange(1, T))

            x = jnp.vstack((x0, x))
            x_hat = jnp.vstack((xhat0, x_hat))

            # stack up state and estimate
            # so that the data have the same structure as the control models
            x = jnp.hstack((x, x_hat))

            # rearrange dimension so that we always have the state
            # and then the corresponding estimate (interleaved)
            return jnp.hstack([x[..., ::2], x[..., 1::2]])

        # simulate n trials
        x = vmap(lambda key: simulate_trial(key, T=T, x0=x0, xhat0=xhat0),
                 out_axes=1)(random.split(rng_key, num=n))

        return x

    def conditional_distribution(self, x):
        T, n, d = x.shape
        idx = list(range(d))
        swapdim = idx[::2] + idx[1::2]

        V = self.dynamics.V
        W = self.dynamics.W

        K = self.K(T)

        F = jnp.vstack(
            [jnp.hstack([self.dynamics.A, jnp.zeros_like(self.dynamics.A)]),
             jnp.hstack([K @ self.dynamics.C @ self.dynamics.A,
                         self.dynamics.A - K @ self.dynamics.C @ self.dynamics.A])])

        F = F[:, swapdim]
        F = F[swapdim, :]

        # TODO: this throws errors for the model with velocities..
        G = jnp.vstack([jnp.hstack([V, jnp.zeros_like(V)]), jnp.hstack([K @ self.dynamics.C @ V, K @ W])])

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
