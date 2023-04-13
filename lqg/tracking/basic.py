from jax import numpy as jnp
from jax.scipy import linalg

from lqg.lqg import System
from lqg.spec import LQGSpec


class TrackingTask(System):
    def __init__(self, dim=1, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, c=1.0, dt=1. / 60., T=1000):
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
        V = jnp.diag(jnp.array([process_noise, motor_noise] * dim))
        W = jnp.diag(jnp.array([sigma, prop_noise] * dim))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1., -1.], [-1., 1.]])] * dim)
        R = jnp.eye(B.shape[1]) * c

        A = jnp.stack((A,) * T)
        B = jnp.stack((B,) * T)
        F = jnp.stack((F,) * T)
        V = jnp.stack((V,) * T)
        W = jnp.stack((W,) * T)
        Q = jnp.stack((Q,) * T)
        R = jnp.stack((R,) * T)

        spec = LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=spec, dynamics=spec)


class BoundedActor(TrackingTask):
    def __init__(self, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, c=1.0, dt=1. / 60, T=1000):
        super().__init__(dim=1, process_noise=process_noise,
                         motor_noise=motor_noise,
                         sigma=sigma, prop_noise=prop_noise, c=c, dt=dt, T=T)


class OptimalActor(TrackingTask):
    def __init__(self, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, dt=1. / 60, T=1000):
        super().__init__(dim=1, process_noise=process_noise,
                         motor_noise=motor_noise,
                         sigma=sigma, prop_noise=prop_noise, c=0., dt=dt, T=T)
