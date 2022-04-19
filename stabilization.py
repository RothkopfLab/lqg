from jax import numpy as jnp
from jax.scipy import linalg

from lqg.model import System, Dynamics, Actor


class Stabilization(System):
    def __init__(self, dim=2, process_noise=1.0, motor_noise=0.5,
                 sigma_h=6.0, sigma_v=6.0, c=1.0, dt=1. / 60.):
        self.process_noise = process_noise

        # dynamics model
        A = jnp.array([[1., dt],
                       [0., 1.]])
        B = dt * jnp.array([[0.], [10.]])

        # observation model
        C = jnp.array([[1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([process_noise, motor_noise]))
        W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = jnp.diag(jnp.array([1., 0]))
        R = jnp.eye(B.shape[1]) * c

        # stack up dimensions
        A = linalg.block_diag(*[A] * dim)
        B = linalg.block_diag(*[B] * dim)
        C = linalg.block_diag(*[C] * dim)
        V = linalg.block_diag(*[V] * dim)
        W = linalg.block_diag(*[W] * dim)  # TODO: correlations
        Q = linalg.block_diag(*[Q] * dim)
        R = linalg.block_diag(*[R] * dim)

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


def autocorr(x):
    result = jnp.correlate(x, x, mode='full')
    return result[result.size // 2:]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import arviz as az

    from lqg.ccg import xcorr
    from lqg.infer import infer

    for sigma in [6.0, 60.0]:
        model = Stabilization(c=1., motor_noise=0.6, sigma=sigma)

        x = model.simulate(n=25, T=500)
        # plt.plot(x[:, 0, 0])
        # plt.title(jnp.mean(x[..., 0] ** 2))
        # plt.show()

        lags, correls = xcorr(x[..., 0].T, x[..., 0].T, maxlags=400)
        start = lags.size // 2
        plt.plot(lags[start:], correls.mean(axis=0)[start:])

    plt.axhline(0, color="gray", linestyle="--")
    plt.show()

    mcmc = infer(x, model=Stabilization, num_samples=5_000, num_warmup=2_000)

    data = az.convert_to_inference_data(mcmc)
    az.plot_pair(data, kind="hexbin")
    plt.show()
