from jax import random, vmap, numpy as jnp
from jax.lax import scan

from lqg.control import glqr
from lqg.belief import kf
from lqg.lqg import System
from lqg.spec import LQGSpec
from lqg.utils import quadratic_form_t


class SignalDependentNoiseSystem(System):
    def __init__(self, actor: LQGSpec, dynamics: LQGSpec):
        super().__init__(actor, dynamics)

    def simulate(self, rng_key, n=1, x0=None, xhat0=None, Sigma0=None, return_all=False):
        """ Simulate n trials

        Args:
            rng_key (jax.random.PRNGKey): random number generator key
            n (int): number of trials
            T (int): number of time steps
            x0 (jnp.array): initial state
            xhat0 (jnp.array): initial belief
            return_all (bool): return estimates, controls and observations as well

        Returns:
            jnp.array (T, n, d)
        """

        Sigma0 = self.actor.V[0] @ self.actor.V[0].T if Sigma0 is None else Sigma0

        gains = glqr.backward(self.actor)
        K = kf.forward(self.actor, Sigma0=Sigma0)

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

            x0 = jnp.zeros(self.xdim) if x0 is None else x0
            xhat0 = jnp.zeros(self.bdim) if xhat0 is None else xhat0

            # generate standard normal noise terms
            key1, key2, key3, key4 = random.split(rng_key, 4)
            noise_x = random.normal(key1, shape=(self.T, self.dynamics.V.shape[-1]))
            noise_y = random.normal(key2, shape=(self.T, self.dynamics.W.shape[-1]))
            noise_Cu = random.normal(key3, (self.T, self.dynamics.Cu.shape[-2],))
            noise_Cx = random.normal(key4, (self.T, self.dynamics.Cx.shape[-2],))

            def loop(carry, t):
                x, x_hat = carry

                # generate observation
                y = self.dynamics.F[t] @ x + self.dynamics.W[t] @ noise_y[t]

                # update agent's belief
                x_pred = self.actor.A[t] @ x_hat + self.actor.B[t] @ (gains.L[t] @ x_hat + gains.l[t])
                x_hat = x_pred + K[t] @ (y - self.actor.F[t] @ x_hat)

                # compute control based on agent's current belief
                u = gains.L[t] @ x_hat + gains.l[t]

                # apply dynamics
                x = (self.dynamics.A[t] @ x + self.dynamics.B[t] @ u + self.dynamics.V[t] @ noise_x[t]
                     + self.dynamics.Cx[t] @ x @ noise_Cx[t] + self.dynamics.Cu[t] @ u @ noise_Cu[t])

                return (x, x_hat), (x, x_hat, y, u)

            _, (x, x_hat, y, u) = scan(loop, (x0, xhat0), jnp.arange(self.T))

            return (jnp.vstack([x0, x]),
                    jnp.vstack([xhat0, x_hat]),
                    jnp.vstack([y, self.dynamics.F[-1] @ x[-1] + self.dynamics.W[-1] @ noise_y[-1]]),
                    u)

        # simulate n trials
        x, x_hat, y, u = vmap(lambda key: simulate_trial(key, x0=x0, xhat0=xhat0))(random.split(rng_key, num=n))

        if return_all:
            return x, x_hat, y, u
        else:
            return x

    def conditional_moments(self, x, Sigma0=None):
        """ Conditional distribution p(x | theta)

        Args:
            self: LQG
            x: time series of shape T (time steps), n (trials), d (dimensionality)

        Returns:
            numpyro.distributions.MultivariateNormal
        """
        T, obs_dim = x.shape
        xdim = self.dynamics.A.shape[1]
        bdim = self.actor.A.shape[1]

        # compute control and estimator gains
        gains = glqr.backward(self.actor)
        K = kf.forward(self.actor, Sigma0=self.actor.V[0] @ self.actor.V[0].T if Sigma0 is None else Sigma0)

        # initialize p(x_t, xhat_t | x_{1:t-1})
        # TODO: initialization should not always be zero for the unobserved dims
        mu = jnp.hstack([x[0], jnp.zeros(xdim - obs_dim + bdim)])
        # TODO: is there a smarter way to initialize Sigma?
        #  For example, with zeros in the covariances, we get problems
        Sigma = jnp.eye(xdim * 2) * 1e-1

        def scan_fn(carry, t):
            mu, Sigma = carry

            # joint dynamics
            F = jnp.concatenate([
                jnp.concatenate([self.dynamics.A[t],
                                 self.dynamics.B[t] @ gains.L[t]],
                                axis=-1),
                jnp.concatenate([K[t] @ self.dynamics.F[t],
                                 self.actor.A[t] + self.actor.B[t] @ gains.L[t] - K[t] @ self.actor.F[t]],
                                axis=-1)],
                axis=-2)

            # joint state-independent noise covariance Cholesky factor
            G = jnp.concatenate([
                jnp.concatenate([self.dynamics.V[t],
                                 jnp.zeros_like(self.dynamics.F[t].T)],
                                axis=-1),
                jnp.concatenate([jnp.zeros_like(self.actor.V[t]),
                                 K[t] @ self.dynamics.W[t]],
                                axis=-1)],
                axis=-2)

            # signal-dependent noise matrix
            M = jnp.concatenate([jnp.concatenate([self.dynamics.Cx[t],
                                                  self.dynamics.Cu[t] @ gains.L[t]], axis=-1),
                                 jnp.concatenate([jnp.zeros((self.dynamics.A[t].shape[0],
                                                             self.dynamics.Cx[t].shape[1],
                                                             self.dynamics.A[t].shape[0])),
                                                  jnp.zeros((self.dynamics.A[t].shape[0],
                                                             self.dynamics.Cu[t].shape[1],
                                                             self.dynamics.A[t].shape[0]))], axis=-1)],
                                axis=-3)

            # conditioning: p(xhat_t | x_1:t)
            Sxh = Sigma[:obs_dim, obs_dim:]
            Shh = Sigma[obs_dim:, obs_dim:]
            Sxx = Sigma[:obs_dim, :obs_dim]
            Shx = Sigma[obs_dim:, :obs_dim]
            mu_x = mu[:obs_dim]
            mu_ba = mu[obs_dim:] + Shx @ jnp.linalg.solve(Sxx, x[t] - mu_x)
            Sigma_ba = Shh - Shx @ jnp.linalg.solve(Sxx, Sxh)

            # updating: p(x_t+1, xhat_t+1 | x_1:t)
            mu = F[:, :obs_dim] @ x[t] + F[:, obs_dim:] @ mu_ba
            rsm_xh = Sigma_ba + jnp.outer(mu_ba, mu_ba)
            TM = quadratic_form_t(M[..., :obs_dim], jnp.outer(x[t], x[t])).sum(axis=0)
            TMh = quadratic_form_t(M[..., obs_dim:], rsm_xh).sum(axis=0)
            Sigma = TM + TMh + F[:, obs_dim:] @ Sigma_ba @ F[:, obs_dim:].T + G @ G.T

            # for numerical stability, add small diagonal term
            Sigma += jnp.eye(Sigma.shape[-1]) * 1e-7  # TODO: can we do this in a more principled way?

            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(scan_fn, (mu, Sigma), jnp.arange(T - 1))
        return mu, Sigma
