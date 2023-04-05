from jax import numpy as jnp

from lqg.kalman import KalmanFilter
from lqg.spec import LQGSpec


class IdealObserver(KalmanFilter):
    def __init__(self, dim=1, process_noise=1., sigma=6., dt=1. / 60., T=1000):
        A = jnp.eye(dim)
        F = jnp.eye(dim)
        V = jnp.eye(dim) * process_noise
        W = jnp.eye(dim) * sigma

        A = jnp.stack((A,) * T)
        B = jnp.zeros((T, dim, dim))
        F = jnp.stack((F,) * T)
        V = jnp.stack((V,) * T)
        W = jnp.stack((W,) * T)
        Q = jnp.zeros((T, dim, dim))
        R = jnp.zeros((T, dim, dim))

        super().__init__(LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R))
