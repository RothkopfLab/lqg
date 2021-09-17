from jax import numpy as jnp
from jax.scipy import linalg

from lqg.model import System, Dynamics, Actor


class DimModel(System):
    def __init__(self, dim=3, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, c=1.0, dt=1. / 60.):
        self.dim = dim
        self.process_noise = process_noise
        # dimensionality
        d = 2 * dim
        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.], [10.]])] * dim)

        # observation model
        C = jnp.eye(d)

        # noise model
        V = jnp.diag(jnp.array([process_noise, motor_noise] * dim))
        W = jnp.diag(jnp.array([sigma, prop_noise] * dim))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1., -1.], [-1., 1.]])] * dim)
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class OneDimModel(DimModel):
    def __init__(self, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, c=1.0, dt=1. / 60):
        super().__init__(dim=1, process_noise=process_noise,
                         motor_noise=motor_noise,
                         sigma=sigma, prop_noise=prop_noise, c=c, dt=dt)


class NoiseFreeModel(DimModel):
    def __init__(self, process_noise=1.0,
                 sigma=6.0, prop_noise=6.0, c=1.0, dt=1. / 60):
        super().__init__(dim=1, process_noise=process_noise,
                         motor_noise=1e-4,
                         sigma=sigma, prop_noise=prop_noise, c=c, dt=dt)


class CostlessModel(DimModel):
    def __init__(self, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, dt=1. / 60):
        super().__init__(dim=1, process_noise=process_noise,
                         motor_noise=motor_noise,
                         sigma=sigma, prop_noise=prop_noise, c=0, dt=dt)


class TwoDimModel(DimModel):
    def __init__(self, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, prop_noise=6.0, c=1.0, dt=1. / 60):
        super().__init__(dim=2, process_noise=process_noise,
                         motor_noise=motor_noise,
                         sigma=sigma, prop_noise=prop_noise, c=c, dt=dt)


class DiffModel(System):
    def __init__(self, dim=1, process_noise=1.0, motor_noise=0.5,
                 sigma=6.0, c=1.0, dt=1. / 60.):
        self.dim = dim
        self.process_noise = process_noise

        # dynamics model
        A = jnp.eye(2 * dim)
        B = dt * linalg.block_diag(*[jnp.array([[0.], [10.]])] * dim)

        # observation model
        C = linalg.block_diag(*[jnp.array([[1., -1.]])] * dim)

        # noise model
        V = jnp.diag(jnp.array([process_noise, motor_noise] * dim))
        W = jnp.diag(jnp.array([sigma] * dim))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1., -1.], [-1., 1.]])] * dim)
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class VelocityModel(System):
    def __init__(self, process_noise=1.0, c=1., motor_noise=0.5, prop_noise=3.0, sigma=6.0, dt=1. / 60.):
        A = linalg.block_diag(*[jnp.array([[1.]]), jnp.array([[1., dt], [0., 1.]])])
        B = jnp.array([[0.], [0.], [10. * dt]])

        C = jnp.array([[1., 0, 0.], [0., 1., 0.]])

        # V = linalg.block_diag(*[jnp.array([[process_noise]]),
        #                         jnp.array([[1 / 3 * dt ** 3, 1 / 2 * dt**2], [1 / 2 * dt**2, dt]]) * motor_noise])
        V = jnp.diag(jnp.array([process_noise, motor_noise, 0.]))
        W = jnp.diag(jnp.array([sigma, prop_noise]))

        # cost function
        Q = jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., c]])
        R = jnp.eye(B.shape[1]) * 0.

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class VelocityDiffModel(System):
    def __init__(self, process_noise=1.0, c=1., motor_noise=0.5, vel_noise=1., sigma=6.0, dt=1. / 60.):
        A = linalg.block_diag(*[jnp.array([[1.]]), jnp.array([[1., dt], [0., 1.]])])
        B = jnp.array([[0.], [0.], [10. * dt]])

        C = jnp.array([[1., -1., 0.]])

        # V = linalg.block_diag(*[jnp.array([[process_noise]]),
        #                         jnp.array([[1 / 3 * dt ** 2, 1 / 2 * dt], [1 / 2 * dt, 1.]]) * motor_noise])
        V = jnp.diag(jnp.array([process_noise, motor_noise, vel_noise]))
        W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., c]])
        R = jnp.eye(B.shape[1]) * 0.

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)
