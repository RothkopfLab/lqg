from itertools import chain
from jax import numpy as jnp
from jax.scipy import linalg

from lqg.glqg import SignalDependentNoiseSystem
from lqg.spec import LQGSpec
from lqg.utils import time_stack


def glqg_tracking_matrices(process_noise, sigma_target, sigma_cursor, action_variability, action_cost, dt=1. / 60.):
    m = 1.
    tau = 0.04

    A = jnp.array([[1., 0., 0., 0., 0.],
                   [0., 1., dt, 0., 0.],
                   [0., 0., 1., dt / m, 0.],
                   [0., 0., 0., 1 - dt / tau, dt / tau],
                   [0., 0., 0., 0., 1 - dt / tau]])
    B = jnp.array([[0.], [0.], [0.], [0.], [dt / tau]])
    V = jnp.array([[process_noise, 0., 0.],
                   [0., action_variability, 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    H = jnp.eye(2, 5)
    W = jnp.diag(jnp.array([sigma_target, sigma_cursor]))
    Q = jnp.array([[1., -1., 0., 0., 0.],
                   [-1., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])
    R = jnp.array([[action_cost]]) * dt ** 2
    return A, B, V, H, W, Q, R


def swap_dims(d, dim):
    idx = list(range(d))
    obs_dims = [idx[(d // dim) * i:((d // dim) * i + 2)] for i in range(dim)]
    unobs_dims = [idx[((d // dim) * i + 2):(d // dim) * (i + 1)] for i in range(dim)]
    dims = list(chain(*(obs_dims + unobs_dims)))
    return dims


class SignalDependentNoiseActor(SignalDependentNoiseSystem):
    def __init__(self, dim=1, process_noise=1.0, action_variability=0.5, signal_dep_noise=0.,
                 sigma_target=6.0, sigma_cursor=6.0, action_cost=1.0, dt=1. / 60., T=1000):
        system = glqg_tracking_matrices(process_noise=process_noise, sigma_target=sigma_target,
                                        sigma_cursor=sigma_cursor,
                                        action_variability=action_variability, action_cost=action_cost, dt=dt)

        A, B, V, F, W, Q, R = (linalg.block_diag(*(M,) * dim) for M in system)

        dims = swap_dims(A.shape[0], dim)

        A = A[dims, :][:, dims]
        B = B[dims, :]
        V = V[dims, :]
        F = F[:, dims]
        Q = Q[dims, :][:, dims]

        C = jnp.array([0., 0., 1.])
        Cu = jnp.stack(
            [B @ linalg.block_diag(*(signal_dep_noise * int(i == j) * C for i in range(dim))) for j in range(dim)],
            axis=-1)

        spec = LQGSpec(A=time_stack(A, T - 1),
                       B=time_stack(B, T - 1),
                       F=time_stack(F, T - 1),
                       V=time_stack(V, T - 1),
                       W=time_stack(W, T - 1),
                       Cu=time_stack(Cu, T - 1),
                       Q=time_stack(Q, T - 1),
                       R=time_stack(R, T - 1))

        super().__init__(actor=spec, dynamics=spec)
