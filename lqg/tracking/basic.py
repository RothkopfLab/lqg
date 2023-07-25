from jax import numpy as jnp
from jax.scipy import linalg

from lqg.lqg import System, Actor


class TrackingTask(System):
    def __init__(self, dim=1, process_noise=1.0, action_variability=0.5,
                 sigma_target=6.0, sigma_cursor=6.0, action_cost=1.0, dt=1. / 60., T=1000):
        self.dim = dim
        self.process_noise = process_noise
        # dimensionality
        d = 2 * dim
        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.], [10.]])] * dim)

        # observation model
        F = jnp.eye(d)

        # noise model
        V = jnp.diag(jnp.array([process_noise, action_variability] * dim))
        W = jnp.diag(jnp.array([sigma_target, sigma_cursor] * dim))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1., -1.], [-1., 1.]])] * dim)
        R = jnp.eye(B.shape[1]) * action_cost

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


class BoundedActor(TrackingTask):
    def __init__(self, process_noise=1.0, action_variability=0.5,
                 sigma_target=6.0, sigma_cursor=6.0, action_cost=1.0, dt=1. / 60, T=1000):
        super().__init__(dim=1, process_noise=process_noise,
                         action_variability=action_variability,
                         sigma_target=sigma_target, sigma_cursor=sigma_cursor, action_cost=action_cost, dt=dt, T=T)


class OptimalActor(TrackingTask):
    def __init__(self, process_noise=1.0, action_variability=0.5,
                 sigma_target=6.0, sigma_cursor=6.0, dt=1. / 60, T=1000):
        super().__init__(dim=1, process_noise=process_noise,
                         action_variability=action_variability,
                         sigma_target=sigma_target, sigma_cursor=sigma_cursor, action_cost=0., dt=dt, T=T)
