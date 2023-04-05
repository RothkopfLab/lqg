import jax.numpy as jnp
import numpyro.distributions as dist
from jax import vmap, random
from jax.lax import scan

from lqg.spec import LQGSpec
from lqg.control import lqr
from lqg.belief import kf


class System:
    def __init__(self, actor: LQGSpec, dynamics: LQGSpec):
        self.actor = actor
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
        """ State dimensionality

        Returns:
            int: dimensionality of state
        """
        return self.dynamics.A.shape[1]

    @property
    def ydim(self):
        """ Observation dimensionality

        Returns:
            int: dimensionality of observation
        """
        return self.dynamics.F.shape[1]

    @property
    def bdim(self):
        """ Belief dimensionality

        Returns:
            int: dimensionality of belief
        """
        return self.actor.A.shape[1]

    @property
    def udim(self):
        """ Action dimensionality

        Returns:
            int: dimensionality of action
        """
        return self.dynamics.B.shape[2]

    def simulate(self, rng_key, n=1, x0=None, xhat0=None, return_all=False):
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

        gains = lqr.backward(self.actor)
        K = kf.forward(self.actor, Sigma0=self.actor.V[0] @ self.actor.V[0].T)

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
            rng_key, subkey = random.split(rng_key)
            epsilon = random.normal(subkey, shape=(self.T, self.xdim))
            rng_key, subkey = random.split(rng_key)
            eta = random.normal(subkey, shape=(self.T, self.ydim))

            def loop(carry, t):
                x, x_hat = carry

                # generate observation
                y = self.dynamics.F[t] @ x + self.dynamics.W[t] @ eta[t]

                # update agent's belief
                x_pred = self.actor.A[t] @ x_hat + self.actor.B[t] @ (gains.L[t] @ x_hat + gains.l[t])
                x_hat = x_pred + K[t] @ (y - self.actor.F[t] @ x_hat)

                # compute control based on agent's current belief
                u = gains.L[t] @ x_hat + gains.l[t]

                # apply dynamics
                x = self.dynamics.A[t] @ x + self.dynamics.B[t] @ u + self.dynamics.V[t] @ epsilon[t]

                return (x, x_hat), (x, x_hat, y, u)

            _, (x, x_hat, y, u) = scan(loop, (x0, xhat0), jnp.arange(1, self.T))

            return (jnp.vstack([x0, x]),
                    jnp.vstack([xhat0, x_hat]),
                    jnp.vstack([y, self.dynamics.F[-1] @ x[-1] + self.dynamics.W[-1] @ eta[-1]]),
                    u)

        # simulate n trials
        x, x_hat, y, u = vmap(lambda key: simulate_trial(key, x0=x0, xhat0=xhat0),
                              out_axes=1)(random.split(rng_key, num=n))

        if return_all:
            return x, x_hat, y, u
        else:
            return x

    def conditional_moments(self, x, mu0=None):
        """ Conditional distribution p(x | theta)

                Args:
                    self: LQG
                    x: time series of shape T (time steps), n (trials), d (dimensionality)

                Returns:
                    numpyro.distributions.MultivariateNormal
                """
        T, n, d = x.shape

        # compute control and estimator gains
        L = self.actor.L(T)
        K = self.actor.K(T)

        # set up joint dynamical system for state and belief
        # p(x_t, xhat_t | x_{t-1}, xhat_{t-1})
        F = jnp.vstack(
            [jnp.hstack([self.dynamics.A,
                         -self.dynamics.B @ L]),
             jnp.hstack([K @ self.dynamics.C @ self.dynamics.A,
                         self.actor.A - self.actor.B @ L - K @ self.actor.C @ self.actor.A])])

        G = jnp.vstack([jnp.hstack([self.dynamics.V, jnp.zeros_like(self.dynamics.C.T)]),
                        jnp.hstack([K @ self.dynamics.C @ self.dynamics.V, K @ self.dynamics.W])])

        # initialize p(x_t, xhat_t | x_{1:t-1})
        mu = jnp.zeros((n, self.dynamics.A.shape[0] + self.actor.A.shape[0])) if mu0 is None else mu0
        Sigma = G @ G.T

        def f(carry, xt):
            mu, Sigma = carry

            # condition on observed state x_t
            mu = mu @ F.T + (xt - mu[:, :d]) @ jnp.linalg.inv(Sigma[:d, :d]).T @ (F @ Sigma)[:, :d].T

            Sigma = F @ Sigma @ F.T + G @ G.T - (F @ Sigma)[:, :d] @ jnp.linalg.inv(Sigma[:d, :d]) @ (Sigma @ F.T)[:d,
                                                                                                     :]
            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(f, (mu, Sigma), x)
        return mu, Sigma

    def conditional_distribution(self, x):
        T, n, d = x.shape

        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = self.conditional_moments(x)

        # marginalize out xhat by using only those entries of mu and Sigma that correspond to x
        return dist.MultivariateNormal(mu[:, :, :d], Sigma[:, jnp.newaxis, :d, :d])

    def log_likelihood(self, x):
        # log likelihood of the states at time t+1 given all previous states up to time t
        return self.conditional_distribution(x[:-1]).log_prob(x[1:])

    def belief_tracking_distribution(self, x):
        d = self.xdim

        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = self.conditional_moments(x)

        # return those elements of mu and Sigma that correspond to xhat
        return dist.MultivariateNormal(mu[:, :, d:], Sigma[:, jnp.newaxis, d:, d:])
