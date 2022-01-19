import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.lax import scan

from lqg.kalman import kalman_gain
from lqg.lqr import control_law


class Dynamics:
    def __init__(self, A, B, C, V):
        self.A = A
        self.B = B
        self.C = C

        self.V = V

    def simulate(self, n=1, T=100, seed=0):
        np.random.seed(seed)
        V = self.V

        xs = []

        xi = jnp.zeros((n, self.A.shape[0]))

        for t in range(0, T):
            xi = xi @ self.A.T + np.random.normal(size=(n, self.A.shape[0])) @ V.T

            xs.append(xi)

        return jnp.stack(xs)


class Actor:
    def __init__(self, A, B, C, V, W, Q, R):
        self.A = A
        self.B = B
        self.C = C

        self.V = V
        self.W = W

        self.Q = Q
        self.R = R

    def L(self, T=100):
        return control_law(self.A, self.B, self.Q, self.R, T=T)

    def K(self, T=100):
        K = kalman_gain(self.A, self.C, self.V @ self.V.T, self.W @ self.W.T, T=T)
        return K


class System:
    def __init__(self, actor, dynamics):
        self.actor = actor
        self.dynamics = dynamics

    @property
    def xdim(self):
        return self.dynamics.A.shape[0]

    @property
    def udim(self):
        return self.dynamics.B.shape[1]

    def simulate(self, n=1, T=100, seed=0, x0=None, xhat0=None, return_all=False):
        np.random.seed(seed)

        A = np.array(self.dynamics.A)
        B = np.array(self.dynamics.B)
        C = np.array(self.dynamics.C)

        L = np.array(self.actor.L(T=T))
        K = np.array(self.actor.K(T=T))

        W = np.array(self.actor.W)
        V = np.array(self.dynamics.V)

        A_act = np.array(self.actor.A)
        B_act = np.array(self.actor.B)
        C_act = np.array(self.actor.C)

        xi = np.zeros((n, self.dynamics.A.shape[0])) if x0 is None else x0
        xhat = np.zeros((n, self.actor.A.shape[0])) if xhat0 is None else xhat0
        # xhat = np.zeros((n, self.actor.A.shape[0]))
        # xi = np.zeros((n, self.dynamics.A.shape[0]))
        # xi = np.array(np.tile(self.x0, (n, 1)))
        # xhat = np.array(np.tile(self.xhat0, (n, 1)))
        u = np.zeros((n, self.actor.B.shape[1])) if xhat0 is None else (-xhat0 @ L.T)

        xs = []

        if return_all:
            xhats = []
            ys = [] # [xi @ C.T + np.random.normal(size=(n, C.shape[0])) @ W.T]
            us = []

        for t in range(T):


            xi = xi @ A.T + u @ B.T + np.random.normal(size=(n, A.shape[0])) @ V.T
            # xi = random.multivariate_normal(subkey, xi @ self.A.T + u @ self.B.T, V)

            yi = xi @ C.T + np.random.normal(size=(n, C.shape[0])) @ W.T
            # yi = random.multivariate_normal(subkey, xi @ self.C.T, W)

            x_pred = xhat @ A_act.T + u @ B_act.T

            xhat = x_pred + (yi - x_pred @ C_act.T) @ K.T

            u = - xhat @ L.T

            xs.append(xi)

            if return_all:
                xhats.append(xhat)
                ys.append(yi)
                us.append(u)

        if return_all:
            return np.stack(xs), np.stack(xhats), np.stack(ys), np.stack(us)
        else:
            return np.stack(xs)

    def simulate_given_x(self, x, seed=0, x0=None, xhat0=None, return_all=False):

        T, n = x.shape

        np.random.seed(seed)

        A = np.array(self.dynamics.A)
        B = np.array(self.dynamics.B)
        C = np.array(self.dynamics.C)

        L = np.array(self.actor.L(T=T))
        K = np.array(self.actor.K(T=T))

        W = np.array(self.actor.W)
        V = np.array(self.dynamics.V)

        A_act = np.array(self.actor.A)
        B_act = np.array(self.actor.B)
        C_act = np.array(self.actor.C)

        xs = []

        xi = np.zeros((n, self.dynamics.A.shape[0])) if x0 is None else x0
        xhat = np.zeros((n, self.actor.A.shape[0])) if xhat0 is None else xhat0
        # xhat = np.zeros((n, self.actor.A.shape[0]))
        # xi = np.zeros((n, self.dynamics.A.shape[0]))
        # xi = np.array(np.tile(self.x0, (n, 1)))
        # xhat = np.array(np.tile(self.xhat0, (n, 1)))

        xs.append(xi)

        if return_all:
            xhats = [xhat]

        for t in range(1, T):
            u = - xhat @ L.T

            xi = xi @ A.T + u @ B.T + np.random.normal(size=(n, A.shape[0])) @ V.T
            xi[:, 0] = x[t]
            # xi = random.multivariate_normal(subkey, xi @ self.A.T + u @ self.B.T, V)

            yi = xi @ C.T + np.random.normal(size=(n, C.shape[0])) @ W.T
            # yi = random.multivariate_normal(subkey, xi @ self.C.T, W)

            x_pred = xhat @ A_act.T + u @ B_act.T

            xhat = x_pred + (yi - x_pred @ C_act.T) @ K.T

            xs.append(xi)
            if return_all:
                xhats.append(xhat)

        if return_all:
            return jnp.stack(xs), jnp.stack(xhats)
        else:
            return jnp.stack(xs)

    def conditional_moments(self, x, mu0=None):
        """ Conditional distribution p(x | theta)

                Args:
                    self: LQG
                    x: time series of shape T (time steps), n (trials), d (dimensionality)

                Returns:
                    numpyro.distributions.MultivariateNormal
                """
        T, n, d = x.shape

        # xdim = self.dynamics.A.shape[0]
        # xhatdim = self.actor.A.shape[0]

        L = self.actor.L(T)

        K = self.actor.K(T)

        F = jnp.vstack(
            [jnp.hstack([self.dynamics.A,
                         -self.dynamics.B @ L]),
             jnp.hstack([K @ self.dynamics.C @ self.dynamics.A,
                         self.actor.A - self.actor.B @ L - K @ self.actor.C @ self.actor.A])])

        # G = jnp.vstack([jnp.hstack([V, jnp.zeros((d, d // 2)), jnp.zeros((d, d))]),
        #                 jnp.hstack([K @ self.C @ V, K @ W, self.E - K @ self.C @ self.E])])

        G = jnp.vstack([jnp.hstack([self.dynamics.V, jnp.zeros_like(self.dynamics.C.T)]),
                        jnp.hstack([K @ self.dynamics.C @ self.dynamics.V, K @ self.actor.W])])

        # mu = jnp.hstack((x[0], x[0]))
        mu = jnp.zeros((n, self.dynamics.A.shape[0] + self.actor.A.shape[0])) if mu0 is None else mu0
        Sigma = G @ G.T

        # Sigma += jnp.eye(Sigma.shape[0]) * 1e-7

        # return (x[0] - mu[:, :d]) # @ jnp.linalg.inv(Sigma[:d, :d]).T @ (F @ Sigma)[:, :d].T

        def f(carry, xt):
            mu, Sigma = carry

            mu = mu @ F.T + (xt - mu[:, :d]) @ jnp.linalg.inv(Sigma[:d, :d]).T @ (F @ Sigma)[:, :d].T

            Sigma = F @ Sigma @ F.T + G @ G.T - (F @ Sigma)[:, :d] @ jnp.linalg.inv(Sigma[:d, :d]) @ (Sigma @ F.T)[:d,
                                                                                                     :]
            return (mu, Sigma), (mu, Sigma)

        _, (mu, Sigma) = scan(f, (mu, Sigma), x)
        return mu, Sigma

    def conditional_distribution(self, x):
        T, n, d = x.shape
        mu, Sigma = self.conditional_moments(x)
        return dist.MultivariateNormal(mu[:, :, :d].transpose((1, 0, 2)), Sigma[:, :d, :d])

    def log_likelihood(self, x):
        return self.conditional_distribution(x[:-1]).log_prob(x[1:].transpose((1, 0, 2)))

    def mean_given_x(self, x):
        T, n, d = x.shape

        A = np.array(self.dynamics.A)
        B = np.array(self.dynamics.B)
        C = np.array(self.dynamics.C)

        L = np.array(self.actor.L(T=T))
        K = np.array(self.actor.K(T=T))

        A_act = np.array(self.actor.A)
        B_act = np.array(self.actor.B)
        C_act = np.array(self.actor.C)

        xs = []

        xi = x[0]  # np.zeros((n, self.dynamics.A.shape[0])) if x0 is None else x0
        xi = np.nan_to_num(xi)

        # TODO: this is a problem for models in which the subjective dimensionality is different
        xhat = x[0]  # np.zeros((n, self.actor.A.shape[0])) if xhat0 is None else xhat0
        xhat = np.nan_to_num(xhat)

        xs.append(xi)

        for t in range(1, T):
            u = - xhat @ L.T

            xi = xi @ A.T + u @ B.T
            xi[~np.isnan(x[t])] = x[t][~np.isnan(x[t])]
            # xi[:, 0:-2:2] = x[t]

            x_pred = xhat @ A_act.T + u @ B_act.T

            xhat = x_pred + (xi @ C.T - x_pred @ C_act.T) @ K.T

            xs.append(xi)

        return np.stack(xs)


class LQG(System):
    def __init__(self, A, B, C, V, W, Q, R):
        dynamics = Dynamics(A, B, C, V)
        actor = Actor(A, B, C, V, W, Q, R)
        super().__init__(actor=actor, dynamics=dynamics)
