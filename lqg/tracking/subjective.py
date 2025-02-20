from itertools import chain
from jax import numpy as jnp
from jax.scipy import linalg

from lqg.lqg import System, Actor, Dynamics

def swap_dims(d, dim):
    idx = list(range(d))
    obs_dims = [idx[(d // dim) * i : ((d // dim) * i + 2)] for i in range(dim)]
    unobs_dims = [idx[((d // dim) * i + 2) : (d // dim) * (i + 1)] for i in range(dim)]
    dims = list(chain(*(obs_dims + unobs_dims)))
    return dims


class SubjectiveActor(System):
    def __init__(self, dim=1, process_noise=1., action_cost=1., action_variability=0.5, subj_noise=1., subj_vel_noise=.5,
                 sigma_target=6., sigma_cursor=6., dt=1. / 60, T=1000):
        A = jnp.eye(2 * dim)
        B = linalg.block_diag(*[jnp.array([[0.], [10. * dt]])] * dim)
        F = jnp.eye(2 * dim)

        V = linalg.block_diag(*[jnp.diag(jnp.array([process_noise, action_variability]))] * dim)
        W = linalg.block_diag(*[jnp.diag(jnp.array([sigma_target, sigma_cursor]))] * dim)

        dyn = Dynamics(A=A, B=B, F=F, V=V, W=W, T=T)

        A = linalg.block_diag(*[jnp.array([[1., 0., dt], [0., 1., 0.], [0., 0., 1.]])] * dim)
        B = linalg.block_diag(*[jnp.array([[0.], [10. * dt], [0.]])] * dim)
        F = linalg.block_diag(*[jnp.array([[1., 0., 0.],
                       [0., 1., 0.]])] * dim)

        V = linalg.block_diag(*[jnp.diag(jnp.array([subj_noise, action_variability, subj_vel_noise]))] * dim)

        Q = linalg.block_diag(*[jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]])] * dim)
        R = jnp.eye(B.shape[1]) * action_cost

        dims = swap_dims(A.shape[0], dim)

        A = A[dims, :][:, dims]
        B = B[dims, :]
        V = V[dims, :]
        F = F[:, dims]
        Q = Q[dims, :][:, dims]

        act = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=act, dynamics=dyn)
