from jax import numpy as jnp
from jax.scipy import linalg

from lqg.model import System, Dynamics, Actor


class Independent3DModel(System):
    def __init__(self, process_noise=1.0, motor_noise=0.5, sigma_h=5., sigma_v=5., sigma_z=10.,
                 c=1.0, dt=1. / 60.):
        dim = 3

        # dimensionality
        d = 2 * dim

        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.], [10.]])] * dim)

        # observation model
        C = linalg.block_diag(*[jnp.array([[1., -1.]])] * dim)

        # noise model
        V = jnp.diag(jnp.array([process_noise, motor_noise] * dim))
        W = jnp.diag(jnp.array([sigma_h, sigma_v, sigma_z]))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1., -1.], [-1., 1.]])] * dim)
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class Independent3DVelocityModel(System):
    def __init__(self, process_noise=1.0, motor_noise=0.5, vel_noise=1., sigma_h=30., sigma_v=30., sigma_z=60.,
                 prop_noise=10., c=1.0, dt=1. / 60.):
        dim = 3

        # dynamics model
        A = jnp.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., dt, 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., dt, 0.],
                       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 1., 0., 0., dt],
                       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        B = dt * jnp.array([[0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.],
                            [10., 0., 0.],
                            [0., 10., 0.],
                            [0., 0., 10.]])

        # observation model
        C = jnp.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0., 1., 0., 0., 0.]])

        # noise model
        V = jnp.diag(jnp.array([process_noise, motor_noise] * dim + [vel_noise] * dim))
        W = jnp.diag(jnp.array([sigma_h, prop_noise, sigma_v, prop_noise, sigma_z, prop_noise]))

        # cost function
        Q = linalg.block_diag(*([jnp.array([[1., -1.], [-1., 1.]])] * dim + [jnp.zeros((dim, dim))]))
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)