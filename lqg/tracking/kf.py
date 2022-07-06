from jax import numpy as jnp

from lqg.kalman import KalmanFilter
from lqg.model import Dynamics


class TrackingFilter(KalmanFilter):
    def __init__(self, dim=1, process_noise=1., sigma=6., dt=1. / 60.):
        A = jnp.eye(dim)
        C = jnp.eye(dim)
        V = jnp.eye(dim) * process_noise
        W = jnp.eye(dim) * sigma

        super().__init__(Dynamics(A, None, C, V, W))


class TwoDimTrackingFilter(KalmanFilter):
    def __init__(self, process_noise=1., sigma_v=6., sigma_h=6., dt=1. / 60.):
        A = jnp.eye(2)
        C = jnp.eye(2)
        V = jnp.eye(2) * process_noise
        W = jnp.diag(jnp.array([sigma_h, sigma_v]))
        super().__init__(Dynamics(A, None, C, V, W))


if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt
    import arviz as az

    from lqg.infer import infer

    kf = TrackingFilter()

    x = kf.simulate(random.PRNGKey(113), n=20, T=500)

    plt.plot(x[:, 0, 0])
    plt.plot(x[:, 0, 1])
    plt.show()

    mcmc = infer(x, model=TrackingFilter, num_samples=5_000, num_warmup=2_000)

    data = az.convert_to_inference_data(mcmc)
    az.plot_trace(data)
    plt.show()
