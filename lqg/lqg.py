import jax.numpy as jnp
import numpyro.distributions as dist
from jax import vmap, random, Array
from jax.lax import scan

from lqg.spec import LQGSpec
from lqg.control import lqr
from lqg.belief import kf
from lqg.utils import time_stack_spec


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

        gains = lqr.backward(self.actor)
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

            _, (x, x_hat, y, u) = scan(loop, (x0, xhat0), jnp.arange(self.T))

            return (jnp.vstack([x0, x]),
                    jnp.vstack([xhat0, x_hat]),
                    jnp.vstack([y, self.dynamics.F[-1] @ x[-1] + self.dynamics.W[-1] @ eta[-1]]),
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
        gains = lqr.backward(self.actor)
        K = kf.forward(self.actor, Sigma0=self.actor.V[0] @ self.actor.V[0].T if Sigma0 is None else Sigma0)

        # set up joint dynamical system for state and belief
        # p(x_t, xhat_t | x_{t-1}, xhat_{t-1})

        # joint dynamics
        F = jnp.concatenate([
            jnp.concatenate([self.dynamics.A,
                             self.dynamics.B @ gains.L], axis=-1),
            jnp.concatenate([K @ self.dynamics.F,
                             self.actor.A + self.actor.B @ gains.L - K @ self.actor.F], axis=-1)],
            axis=-2)

        # joint noise covariance Cholesky factor
        G = jnp.concatenate([
            jnp.concatenate([self.dynamics.V,
                             jnp.zeros((self.T, self.dynamics.A.shape[1], self.dynamics.W.shape[2]))], axis=-1),
            jnp.concatenate([jnp.zeros((self.T, self.actor.A.shape[1], self.dynamics.A.shape[2])),
                             K @ self.dynamics.W], axis=-1)],
            axis=-2)

        # initialize p(x_t, xhat_t | x_{1:t-1})
        # TODO: initialization should not always be zero for the unobserved dims
        mu0 = jnp.hstack([x[0], jnp.zeros(xdim - obs_dim + bdim)])
        Sigma0 = G[0] @ G[0].T

        def scan_fn(carry, step):
            mu, Sigma = carry
            F, G, x = step

            # conditioning and marginalizing
            mu = F @ mu + (F @ Sigma)[:, :obs_dim] @ jnp.linalg.solve(Sigma[:obs_dim, :obs_dim], x - mu[:obs_dim])

            Sigma = F @ Sigma @ F.T + G @ G.T - (F @ Sigma)[:, :obs_dim] @ jnp.linalg.solve(Sigma[:obs_dim, :obs_dim],
                                                                                            (Sigma @ F.T)[:obs_dim, :])
            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(scan_fn, (mu0, Sigma0), (F, G, x[:-1]))
        # return jnp.vstack((mu0, mu)), jnp.vstack((Sigma0[jnp.newaxis], Sigma))
        return mu, Sigma

    def conditional_distribution(self, x, Sigma0=None):
        n, T, d = x.shape

        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = vmap(self.conditional_moments, in_axes=(0, None))(x, Sigma0)

        # marginalize out xhat by using only those entries of mu and Sigma that correspond to x
        return dist.MultivariateNormal(mu[:, :, :d], Sigma[:, :, :d, :d])

    def log_likelihood(self, x, Sigma0=None):
        # log likelihood of the states at time t+1 given all previous states up to time t
        return self.conditional_distribution(x, Sigma0=Sigma0).log_prob(x[:, 1:])

    def belief_tracking_distribution(self, x, Sigma0=None):
        d = self.xdim

        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = vmap(self.conditional_moments, in_axes=(0, None))(x, Sigma0)

        # return those elements of mu and Sigma that correspond to xhat
        return dist.MultivariateNormal(mu[:, :, d:], Sigma[:, :, d:, d:])
    
    def _repr_latex_(self) -> str:
        """
        Produces a Latex representation of the system.
        Can be used in jupyter notebooks via `display(system)`.

        Returns:
            str: Latex representation.
        """
        def bmatrix(arr: Array) -> str:
            """
            Produces a Latex bmatrix string representation of a given array.
            Adapted from https://stackoverflow.com/a/17131750.

            Args:
                arr (Array): Array to represent as bmatrix.

            Returns:
                str: Latex bmatrix representation.
            """
            assert jnp.ndim(arr) == 2 

            lines = (
                jnp.array_str(arr, max_line_width=jnp.inf, precision=4)
                .replace("[", "")
                .replace("]", "")
                .splitlines()
            )

            rows = ["\\begin{bmatrix}"]
            rows += [" " + " & ".join(l.split()) + "\\\\" for l in lines]
            rows += ["\\end{bmatrix}"]
            bmatrix = "".join(rows)
            return bmatrix

        mats_dynamics = [
            self.dynamics.A,
            self.dynamics.B,
            self.dynamics.F,
            self.dynamics.V,
            self.dynamics.W,
            # self.dynamics.Q,
            # self.dynamics.R
        ]
        mats_actor = [
            self.actor.A,
            self.actor.B,
            self.actor.F,
            self.actor.V,
            self.actor.W,
            self.actor.Q,
            self.actor.R,
        ]
        mat_names = ["A", "B", "F", "V", "W", "Q", "R"]

        latex_str = "\\begin{align*} \\text{Dynamics:}"
        for mat, mat_name in zip(mats_dynamics, mat_names):
            latex_str += f" &&{mat_name} = {bmatrix(mat[0])}"

        latex_str += "\\\\"

        latex_str += "\\text{Actor:}"
        for mat, mat_name in zip(mats_actor, mat_names):
            latex_str += f" &&{mat_name} = {bmatrix(mat[0])}"

        latex_str += "\\end{align*}"
        return latex_str


def Dynamics(A, B, F, V, W, T=1000):
    xdim = A.shape[0]
    udim = B.shape[1]

    return time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=jnp.zeros((xdim, xdim)), R=jnp.zeros((udim, udim)), T=T - 1)


def Actor(A, B, F, V, W, Q, R, T=1000):
    return time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T - 1)


class LQG(System):
    def __init__(self, A, B, F, V, W, Q, R, T=1000):
        spec = time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T - 1)

        super().__init__(actor=spec, dynamics=spec)
