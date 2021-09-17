from jax import numpy as jnp

from lqg.kalman import KalmanFilter


class TrackingFilter(KalmanFilter):
    def __init__(self, dim=1, process_noise=1., sigma=6., dt=1. / 60.):
        A = jnp.eye(dim)
        C = jnp.eye(dim)
        V = jnp.eye(dim) * process_noise
        W = jnp.eye(dim) * sigma
        super().__init__(A, C, V, W)


class TwoDimTrackingFilter(KalmanFilter):
    def __init__(self, process_noise=1., sigma_v=6., sigma_h=6., dt=1. / 60.):
        A = jnp.eye(2)
        C = jnp.eye(2)
        V = jnp.eye(2) * process_noise
        W = jnp.diag(jnp.array([sigma_h, sigma_v]))
        super().__init__(A, C, V, W)
