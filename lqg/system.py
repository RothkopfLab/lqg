import jax.numpy as jnp
import numpyro.distributions as dist
from jax import vmap, random, Array
from jax.lax import scan

from lqg.spec import LQGSpec
from lqg.control import lqr
from lqg.belief import kf
from lqg.utils import time_stack_spec


class System:
    """An agent acting in an environment, both described as linear-Gaussian systems.

    The environment (``dynamics``) evolves as

    ```
    x[t+1] = A x[t] + B u[t] + V eps[t],    eps[t] ~ N(0, I)
    y[t+1] = F x[t+1] + W eta[t],         eta[t] ~ N(0, I)
    ```

    The agent (``actor``) has its own internal model of the environment, with
    its own matrices ``A, B, F, V, W`` and a quadratic cost (``Q, R``). It
    computes optimal control gains for that cost (LQR) and tracks its belief
    ``xhat`` about the state with a Kalman filter. Its actions
    ``u[t] = L[t] xhat[t] + k[t]`` then drive the true dynamics.

    When the agent's model matches the environment, use [LQG][lqg.system.LQG].
    Keeping the two separate allows models of agents with subjective or
    mismatched beliefs about the environment.

    Shapes in this class use these names:

    - ``n``: number of trials
    - ``T``: number of time steps, so trajectories have ``T + 1`` states
    - ``xdim``: state dimensionality of the environment
    - ``bdim``: state dimensionality of the agent's internal model
    - ``ydim``: observation dimensionality
    - ``udim``: action dimensionality

    Args:
        actor (LQGSpec): the agent's internal model and cost function, e.g. from
            [Actor][lqg.system.Actor].
        dynamics (LQGSpec): the true dynamics of the environment, e.g. from
            [Dynamics][lqg.system.Dynamics]. Its cost terms are ignored.
    """

    def __init__(self, actor: LQGSpec, dynamics: LQGSpec):
        self.actor = actor
        self.dynamics = dynamics

    @property
    def T(self):
        """Length of trajectory

        Returns:
            int: number of time steps
        """
        return self.dynamics.A.shape[0]

    @property
    def xdim(self):
        """State dimensionality

        Returns:
            int: dimensionality of state
        """
        return self.dynamics.A.shape[1]

    @property
    def ydim(self):
        """Observation dimensionality

        Returns:
            int: dimensionality of observation
        """
        return self.dynamics.F.shape[1]

    @property
    def bdim(self):
        """Belief dimensionality

        Returns:
            int: dimensionality of belief
        """
        return self.actor.A.shape[1]

    @property
    def udim(self):
        """Action dimensionality

        Returns:
            int: dimensionality of action
        """
        return self.dynamics.B.shape[2]

    def simulate(
        self, rng_key, n=1, x0=None, xhat0=None, Sigma0=None, return_all=False
    ):
        """Simulate n trials

        Args:
            rng_key (jax.random.PRNGKey): random number generator key
            n (int): number of trials
            x0 (jnp.array, optional): initial state of shape (xdim,). Defaults
                to zeros.
            xhat0 (jnp.array, optional): initial belief of shape (bdim,).
                Defaults to zeros.
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim), used to compute the Kalman gains. Defaults to
                ``V[0] @ V[0].T`` of the actor.
            return_all (bool): return beliefs, observations and actions as well

        Returns:
            jnp.array or tuple: If ``return_all`` is False, the simulated states
                ``x`` of shape (n, T+1, xdim), including the initial state.
                Otherwise, a tuple ``(x, xhat, y, u)`` with the states, the
                beliefs of shape (n, T+1, bdim), the observations ``y[1:]`` of
                shape (n, T, ydim) and the actions of shape (n, T, udim).
        """

        Sigma0 = self.actor.V[0] @ self.actor.V[0].T if Sigma0 is None else Sigma0

        gains = lqr.backward(self.actor)
        K = kf.forward(self.actor, Sigma0=Sigma0)

        def simulate_trial(rng_key, x0=None, xhat0=None):
            """Simulate a single trial

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

                # compute control based on agent's current belief
                u = gains.L[t] @ x_hat + gains.k[t]

                # apply dynamics
                x = (
                    self.dynamics.A[t] @ x
                    + self.dynamics.B[t] @ u
                    + self.dynamics.V[t] @ epsilon[t]
                )

                # generate observation
                y = self.dynamics.F[t] @ x + self.dynamics.W[t] @ eta[t]

                # update agent's belief
                x_pred = self.actor.A[t] @ x_hat + self.actor.B[t] @ u
                x_hat = x_pred + K[t] @ (y - self.actor.F[t] @ x_pred)

                return (x, x_hat), (x, x_hat, y, u)

            _, (x, x_hat, y, u) = scan(loop, (x0, xhat0), jnp.arange(self.T))

            return (jnp.vstack([x0, x]), jnp.vstack([xhat0, x_hat]), y, u)

        # simulate n trials
        x, x_hat, y, u = vmap(lambda key: simulate_trial(key, x0=x0, xhat0=xhat0))(
            random.split(rng_key, num=n)
        )

        if return_all:
            return x, x_hat, y, u
        else:
            return x

    def simulate_actions(
        self, x, action_dim_mask, rng_key, xhat0=None, Sigma0=None, return_all=False
    ):
        """Simulate cursor actions given pre-existing experimental target trajectories.

        Args:
        The target dimensions of the state are taken from the data at each time
        step, while the dimensions selected by ``action_dim_mask`` are simulated
        from the agent's actions.

        Args:
            x (jnp.array): Experimental state data of shape (n, T+1, xdim)
            action_dim_mask (jnp.array): Boolean mask of shape (xdim,) that
                selects the state dimensions to simulate (e.g. cursor position)
            rng_key (jax.random.PRNGKey): Random number generator key
            xhat0 (jnp.array, optional): Initial belief of shape (bdim,).
                Defaults to zeros.
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim). Defaults to ``V[0] @ V[0].T`` of the actor.
            return_all (bool): Whether to return beliefs, observations and
                actions as well

        Returns:
            jnp.array or tuple: If ``return_all`` is False, the states of shape
                (n, T+1, xdim). Otherwise, a tuple ``(x, xhat, y, u)`` with the
                same shapes as in [simulate][lqg.system.System.simulate].
        """
        Sigma0 = self.actor.V[0] @ self.actor.V[0].T if Sigma0 is None else Sigma0

        gains = lqr.backward(self.actor)
        K = kf.forward(self.actor, Sigma0=Sigma0)

        def simulate_trial_actions(x_single, action_dim_mask, rng_key, xhat0=None):
            # 1. Initial state at t=0 comes directly from experimental data x_single[0]
            x0 = x_single[0]
            xhat0 = jnp.zeros(self.bdim) if xhat0 is None else xhat0

            # Generate standard normal noise terms
            rng_key, subkey = random.split(rng_key)
            epsilon = random.normal(subkey, shape=(self.T, self.xdim))
            rng_key, subkey = random.split(rng_key)
            eta = random.normal(subkey, shape=(self.T, self.ydim))

            def loop(carry, t):
                x_curr, x_hat = carry  # x_curr is a 1D state vector of shape (xdim,)

                # Compute control based on agent's current belief
                u = gains.L[t] @ x_hat + gains.k[t]

                # Apply full dynamics transition: A[t] @ x_curr carries state & physics forward
                x_next_sim = (
                    self.dynamics.A[t] @ x_curr
                    + self.dynamics.B[t] @ u
                    + self.dynamics.V[t] @ epsilon[t]
                )

                # Splicing: Take fixed experimental target state for t+1,
                # but inject newly simulated cursor/action states
                x_next = (
                    x_single[t + 1].at[action_dim_mask].set(x_next_sim[action_dim_mask])
                )

                # Generate observation based on updated state x_next
                y = self.dynamics.F[t] @ x_next + self.dynamics.W[t] @ eta[t]

                # Update agent's belief
                x_pred = self.actor.A[t] @ x_hat + self.actor.B[t] @ u
                x_hat = x_pred + K[t] @ (y - self.actor.F[t] @ x_pred)

                return (x_next, x_hat), (x_next, x_hat, y, u)

            # Run scan across T time steps
            _, (x_steps, x_hat, y, u) = scan(loop, (x0, xhat0), jnp.arange(self.T))

            # Combine initial state x0 with simulated sequence -> shape (T+1, xdim)
            x_full = jnp.vstack([x0, x_steps])

            return x_full, jnp.vstack([xhat0, x_hat]), y, u

        # Map across trials and random keys
        keys = random.split(rng_key, num=x.shape[0])
        x_sim, x_hat, y, u = vmap(
            lambda x_i, key: simulate_trial_actions(
                x_i, action_dim_mask, key, xhat0=xhat0
            )
        )(x, keys)

        if return_all:
            return x_sim, x_hat, y, u
        else:
            return x_sim

    def conditional_moments(self, x, x0=None, xhat0=None, Sigma0=None):
        """Moments of the joint state and belief, given past observed states.

        For each time step ``t``, computes the mean and covariance of the
        Gaussian ``p(x[t+1], xhat[t+1] | x[0:t+1])`` for a single trial, where
        the agent's belief ``xhat`` is marginalized out over the past.

        Args:
            x (jnp.array): one trial of observed states, of shape (T+1, d). If
                ``d < xdim``, the first ``d`` state dimensions are observed.
            x0 (jnp.array, optional): initial state. Defaults to ``x[0]``.
            xhat0 (jnp.array, optional): initial belief. Defaults to ``x0``.
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim), used to compute the Kalman gains. Defaults to
                ``V[0] @ V[0].T`` of the actor.

        Returns:
            tuple: ``(mu, Sigma)``, the means of shape (T, xdim + bdim) and the
                covariances of shape (T, xdim + bdim, xdim + bdim). The first
                ``xdim`` entries are the state, the rest the belief.
        """
        T, obs_dim = x.shape
        xdim = self.dynamics.A.shape[1]
        bdim = self.actor.A.shape[1]

        # compute control and estimator gains
        gains = lqr.backward(self.actor)
        K = kf.forward(
            self.actor,
            Sigma0=self.actor.V[0] @ self.actor.V[0].T if Sigma0 is None else Sigma0,
        )

        # set up joint dynamical system for state and belief
        # p(x_t, xhat_t | x_{t-1}, xhat_{t-1})

        # joint dynamics
        F = jnp.concatenate(
            [
                jnp.concatenate([self.dynamics.A, self.dynamics.B @ gains.L], axis=-1),
                jnp.concatenate(
                    [
                        K @ self.dynamics.F @ self.dynamics.A,
                        self.actor.A
                        + self.actor.B @ gains.L
                        - K @ self.actor.F @ self.actor.A
                        + K
                        @ (
                            self.dynamics.F @ self.dynamics.B
                            - self.actor.F @ self.actor.B
                        )
                        @ gains.L,
                    ],
                    axis=-1,
                ),
            ],
            axis=-2,
        )

        # joint noise covariance Cholesky factor
        G = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        self.dynamics.V,
                        jnp.zeros(
                            (self.T, self.dynamics.A.shape[1], self.dynamics.W.shape[2])
                        ),
                    ],
                    axis=-1,
                ),
                jnp.concatenate(
                    [K @ self.dynamics.F @ self.dynamics.V, K @ self.dynamics.W],
                    axis=-1,
                ),
            ],
            axis=-2,
        )

        # initialize p(x_t, xhat_t | x_{1:t-1})
        if x0 is None:
            x0 = x[0]
        if xhat0 is None:
            xhat0 = x0
        mu0 = jnp.hstack(
            [x0, jnp.zeros(xdim - x0.shape[0]), xhat0, jnp.zeros(bdim - xhat0.shape[0])]
        )
        Sigma0 = G[0] @ G[0].T

        def scan_fn(carry, step):
            mu, Sigma = carry
            F, G, x = step

            # conditioning and marginalizing
            mu = F @ mu + (F @ Sigma)[:, :obs_dim] @ jnp.linalg.solve(
                Sigma[:obs_dim, :obs_dim], x - mu[:obs_dim]
            )

            Sigma = (
                F @ Sigma @ F.T
                + G @ G.T
                - (F @ Sigma)[:, :obs_dim]
                @ jnp.linalg.solve(
                    Sigma[:obs_dim, :obs_dim], (Sigma @ F.T)[:obs_dim, :]
                )
            )
            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(scan_fn, (mu0, Sigma0), (F, G, x[:-1]))
        # return jnp.vstack((mu0, mu)), jnp.vstack((Sigma0[jnp.newaxis], Sigma))
        return mu, Sigma

    def conditional_distribution(self, x, x0=None, xhat0=None, Sigma0=None):
        """Distribution of each state given all previous states.

        This is ``p(x[t+1] | x[0:t+1])`` for every trial and time step, with the
        agent's beliefs marginalized out. See
        [conditional_moments][lqg.system.System.conditional_moments].

        Args:
            x (jnp.array): observed states of shape (n, T+1, d)
            x0 (jnp.array, optional): initial states of shape (n, d). Defaults
                to ``x[:, 0]``.
            xhat0 (jnp.array, optional): initial beliefs of shape (n, bdim).
                Defaults to ``x0``.
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim)

        Returns:
            numpyro.distributions.Distribution: a multivariate normal with batch
                shape (n,) and event shape (T, d), to be evaluated on
                ``x[:, 1:]``.
        """
        n, T, d = x.shape

        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = vmap(self.conditional_moments, in_axes=(0, 0, 0, None))(
            x, x0, xhat0, Sigma0
        )

        # marginalize out xhat by using only those entries of mu and Sigma that correspond to x
        return dist.MultivariateNormal(mu[:, :, :d], Sigma[:, :, :d, :d]).to_event(1)

    def log_likelihood(self, x, x0=None, xhat0=None, Sigma0=None):
        """Log likelihood of observed trajectories.

        Evaluates ``log p(x[1:] | x[0])``, factorized over time steps as in
        [conditional_distribution][lqg.system.System.conditional_distribution].

        Args:
            x (jnp.array): observed states of shape (n, T+1, d)
            x0 (jnp.array, optional): initial states of shape (n, d)
            xhat0 (jnp.array, optional): initial beliefs of shape (n, bdim)
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim)

        Returns:
            jnp.array: log likelihood of each trial, of shape (n,)
        """
        return self.conditional_distribution(
            x, x0=x0, xhat0=xhat0, Sigma0=Sigma0
        ).log_prob(x[:, 1:])

    def belief_tracking_distribution(self, x, Sigma0=None):
        """Distribution of the agent's beliefs given the observed states.

        This is ``p(xhat[t+1] | x[0:t+1])`` for every trial and time step.

        Args:
            x (jnp.array): observed states of shape (n, T+1, xdim)
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim)

        Returns:
            numpyro.distributions.MultivariateNormal: a distribution with batch
                shape (n, T) and event shape (bdim,)
        """
        d = self.xdim

        # compute p(x_{t+1}, xhat_{t+1} | x_{1:t})
        mu, Sigma = vmap(
            lambda xi: self.conditional_moments(xi, Sigma0=Sigma0), in_axes=0
        )(x)

        # return those elements of mu and Sigma that correspond to xhat
        return dist.MultivariateNormal(mu[:, :, d:], Sigma[:, :, d:, d:])

    def to_numpyro(self, Sigma0=None, xdim=None):
        """Wrap the system as a numpyro distribution over trajectories.

        The distribution can be used with ``numpyro.sample`` in a model, e.g.
        ``numpyro.sample("x", system.to_numpyro(), obs=x)``.

        Args:
            Sigma0 (jnp.array, optional): initial belief covariance of shape
                (bdim, bdim)
            xdim (int, optional): number of observed state dimensions. Defaults
                to all of them.

        Returns:
            numpyro.distributions.Distribution: a distribution with event shape
                (T+1, xdim)
        """
        return NumpyroLQG(self, Sigma0=Sigma0, xdim=xdim)

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
            rows += [" " + " & ".join(line.split()) + "\\\\" for line in lines]
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
    """Specification of an environment with time-invariant dynamics.

    ```
    x[t+1] = A x[t] + B u[t] + V eps[t],    eps[t] ~ N(0, I)
    y[t+1] = F x[t+1] + W eta[t],         eta[t] ~ N(0, I)
    ```

    Args:
        A (jnp.array): state transition matrix of shape (xdim, xdim)
        B (jnp.array): control matrix of shape (xdim, udim)
        F (jnp.array): observation matrix of shape (ydim, xdim)
        V (jnp.array): process noise Cholesky factor of shape (xdim, xdim)
        W (jnp.array): observation noise Cholesky factor of shape (ydim, ydim)
        T (int): number of time steps

    Returns:
        LQGSpec: the matrices repeated over ``T`` time steps, with zero costs
    """
    xdim = A.shape[0]
    udim = B.shape[1]

    return time_stack_spec(
        A=A,
        B=B,
        F=F,
        V=V,
        W=W,
        Q=jnp.zeros((xdim, xdim)),
        R=jnp.zeros((udim, udim)),
        T=T,
    )


def Actor(A, B, F, V, W, Q, R, T=1000):
    """Specification of an agent's internal model and cost function.

    The agent assumes the same linear-Gaussian dynamics as in
    [Dynamics][lqg.system.Dynamics] and minimizes the cost
    ``sum_t x[t]^T Q x[t] + u[t]^T R u[t]`` (with constant factors dropped).

    Args:
        A (jnp.array): state transition matrix of shape (bdim, bdim)
        B (jnp.array): control matrix of shape (bdim, udim)
        F (jnp.array): observation matrix of shape (ydim, bdim)
        V (jnp.array): process noise Cholesky factor of shape (bdim, bdim)
        W (jnp.array): observation noise Cholesky factor of shape (ydim, ydim)
        Q (jnp.array): state cost matrix of shape (bdim, bdim)
        R (jnp.array): action cost matrix of shape (udim, udim)
        T (int): number of time steps

    Returns:
        LQGSpec: the matrices repeated over ``T`` time steps. The final state
            cost is ``Q``, and all linear and cross terms are zero.
    """
    return time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)


class LQG(System):
    """A system in which the agent's internal model matches the environment.

    The same matrices are used for the actor and the dynamics. See
    [Actor][lqg.system.Actor] for their meaning and shapes.
    """

    def __init__(self, A, B, F, V, W, Q, R, T=1000):
        """Create an LQG system from time-invariant matrices.

        Args:
            A (jnp.array): state transition matrix of shape (xdim, xdim)
            B (jnp.array): control matrix of shape (xdim, udim)
            F (jnp.array): observation matrix of shape (ydim, xdim)
            V (jnp.array): process noise Cholesky factor of shape (xdim, xdim)
            W (jnp.array): observation noise Cholesky factor of shape
                (ydim, ydim)
            Q (jnp.array): state cost matrix of shape (xdim, xdim)
            R (jnp.array): action cost matrix of shape (udim, udim)
            T (int): number of time steps
        """
        spec = time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


class NumpyroLQG(dist.Distribution):
    """A [System][lqg.system.System] as a numpyro distribution over trajectories.

    Usually created with [to_numpyro][lqg.system.System.to_numpyro].
    """

    def __init__(self, system: System, xdim=None, Sigma0=None):
        self.system = system
        self.Sigma0 = Sigma0
        xdim = system.xdim if xdim is None else xdim

        super().__init__(
            event_shape=(system.T + 1, xdim),
            batch_shape=(),
        )

    def log_prob(self, x):
        return self.system.log_likelihood(x, Sigma0=self.Sigma0)

    def sample(self, key, sample_shape=()):
        if len(sample_shape) == 0:
            return self.system.simulate(key, n=1, Sigma0=self.Sigma0)[0]
        else:
            return self.system.simulate(key, n=sample_shape[0], Sigma0=self.Sigma0)
