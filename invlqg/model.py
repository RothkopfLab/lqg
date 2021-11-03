from collections import namedtuple

import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.lax import scan

from invlqg.kalman import kalman_gain
from invlqg.lqr import control_law

Dynamics = namedtuple("Dynamics", ["A", "B", "H", "V"])
Actor = namedtuple("Actor", ["A", "B", "V", "H", "W", "Q", "R"])
SystemSpec = namedtuple("SystemSpec", ["dynamics", "actor"])


def compute_kalman_gain(actor, T):
    return kalman_gain(A=actor.A, H=actor.H, V=actor.V @ actor.V.T, W=actor.W @ actor.W.T, T=T)


def compute_control_law(actor, T):
    return control_law(A=actor.A, B=actor.B, Q=actor.Q, R=actor.R, T=T)


def conditional_moments(system, x):
    T, n, d = x.shape

    L = compute_control_law(system.actor, T)

    K = compute_kalman_gain(system.actor, T)

    F = jnp.vstack(
        [jnp.hstack([system.dynamics.A,
                     -system.dynamics.B @ L]),
         jnp.hstack([K @ system.dynamics.H @ system.dynamics.A,
                     system.actor.A - system.actor.B @ L - K @ system.actor.H @ system.actor.A])])

    G = jnp.vstack([jnp.hstack([system.dynamics.V, jnp.zeros_like(system.dynamics.H.T)]),
                    jnp.hstack([K @ system.dynamics.H @ system.dynamics.V, K @ system.actor.W])])

    mu = jnp.zeros((n, system.dynamics.A.shape[0] + system.actor.A.shape[0]))  # if mu0 is None else mu0
    Sigma = G @ G.T

    def f(carry, xt):
        mu, Sigma = carry

        mu = mu @ F.T + (xt - mu[:, :d]) @ jnp.linalg.inv(Sigma[:d, :d]).T @ (F @ Sigma)[:, :d].T

        Sigma = F @ Sigma @ F.T + G @ G.T - (F @ Sigma)[:, :d] @ jnp.linalg.inv(Sigma[:d, :d]) @ (Sigma @ F.T)[:d,
                                                                                                 :]
        return (mu, Sigma), (mu, Sigma)

    _, (mu, Sigma) = scan(f, (mu, Sigma), x)
    return mu, Sigma


def conditional_distribution(system, x):
    T, n, d = x.shape
    mu, Sigma = conditional_moments(system, x)
    return dist.MultivariateNormal(mu[:, :, :d].transpose((1, 0, 2)), Sigma[:, :d, :d])


def log_likelihood(system, x):
    return conditional_distribution(system, x[:-1]).log_prob(x[1:].transpose((1, 0, 2)))


def simulate(system, n=1, T=100, seed=0, x0=None, xhat0=None, return_all=False):
    np.random.seed(seed)

    A = np.array(system.dynamics.A)
    B = np.array(system.dynamics.B)
    H = np.array(system.dynamics.H)

    L = np.array(compute_control_law(system.actor, T))
    K = np.array(compute_kalman_gain(system.actor, T))

    W = np.array(system.actor.W)
    V = np.array(system.dynamics.V)

    A_act = np.array(system.actor.A)
    B_act = np.array(system.actor.B)
    H_act = np.array(system.actor.H)

    xi = np.zeros((n, system.dynamics.A.shape[0])) if x0 is None else x0
    xhat = np.zeros((n, system.actor.A.shape[0])) if xhat0 is None else xhat0
    u = np.zeros((n, system.actor.B.shape[1])) if xhat0 is None else (-xhat0 @ L.T)

    xs = []

    if return_all:
        xhats = []
        ys = []  # [xi @ H.T + np.random.normal(size=(n, H.shape[0])) @ W.T]
        us = []

    for t in range(T):

        xi = xi @ A.T + u @ B.T + np.random.normal(size=(n, A.shape[0])) @ V.T
        # xi = random.multivariate_normal(subkey, xi @ system.A.T + u @ system.B.T, V)

        yi = xi @ H.T + np.random.normal(size=(n, H.shape[0])) @ W.T
        # yi = random.multivariate_normal(subkey, xi @ system.H.T, W)

        x_pred = xhat @ A_act.T + u @ B_act.T

        xhat = x_pred + (yi - x_pred @ H_act.T) @ K.T

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
        H = np.array(self.dynamics.H)

        L = np.array(compute_control_law(self.actor, T))
        K = np.array(compute_kalman_gain(self.actor, T))

        W = np.array(self.actor.W)
        V = np.array(self.dynamics.V)

        A_act = np.array(self.actor.A)
        B_act = np.array(self.actor.B)
        H_act = np.array(self.actor.H)

        xi = np.zeros((n, self.dynamics.A.shape[0])) if x0 is None else x0
        xhat = np.zeros((n, self.actor.A.shape[0])) if xhat0 is None else xhat0
        u = np.zeros((n, self.actor.B.shape[1])) if xhat0 is None else (-xhat0 @ L.T)

        xs = []

        if return_all:
            xhats = []
            ys = []  # [xi @ H.T + np.random.normal(size=(n, H.shape[0])) @ W.T]
            us = []

        for t in range(T):

            xi = xi @ A.T + u @ B.T + np.random.normal(size=(n, A.shape[0])) @ V.T
            # xi = random.multivariate_normal(subkey, xi @ system.A.T + u @ system.B.T, V)

            yi = xi @ H.T + np.random.normal(size=(n, H.shape[0])) @ W.T
            # yi = random.multivariate_normal(subkey, xi @ system.H.T, W)

            x_pred = xhat @ A_act.T + u @ B_act.T

            xhat = x_pred + (yi - x_pred @ H_act.T) @ K.T

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

    def conditional_moments(self, x, mu0=None):
        """ Conditional distribution p(x | theta)

                Args:
                    self: LQG
                    x: time series of shape T (time steps), n (trials), d (dimensionality)

                Returns:
                    numpyro.distributions.MultivariateNormal
                """
        T, n, d = x.shape

        L = compute_control_law(self.actor, T)

        K = compute_kalman_gain(self.actor, T)

        F = jnp.vstack(
            [jnp.hstack([self.dynamics.A,
                         -self.dynamics.B @ L]),
             jnp.hstack([K @ self.dynamics.H @ self.dynamics.A,
                         self.actor.A - self.actor.B @ L - K @ self.actor.H @ self.actor.A])])

        G = jnp.vstack([jnp.hstack([self.dynamics.V, jnp.zeros_like(self.dynamics.H.T)]),
                        jnp.hstack([K @ self.dynamics.H @ self.dynamics.V, K @ self.actor.W])])

        mu = jnp.zeros((n, self.dynamics.A.shape[0] + self.actor.A.shape[0])) if mu0 is None else mu0
        Sigma = G @ G.T

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


class LQG(System):
    def __init__(self, A, B, H, V, W, Q, R):
        dynamics = Dynamics(A, B, H, V)
        actor = Actor(A, B, H, V, W, Q, R)
        super().__init__(actor=actor, dynamics=dynamics)
