from itertools import chain
from jax import numpy as jnp
from jax.scipy import linalg

from lqg.system import System, Actor, Dynamics


def swap_dims(d, dim):
    idx = list(range(d))
    obs_dims = [idx[(d // dim) * i : ((d // dim) * i + 2)] for i in range(dim)]
    unobs_dims = [idx[((d // dim) * i + 2) : (d // dim) * (i + 1)] for i in range(dim)]
    dims = list(chain(*(obs_dims + unobs_dims)))
    return dims


class SubjectiveActor(System):
    """Bounded actor with a subjective model of the target dynamics.

    The true dynamics are the same as in
    [BoundedActor][lqg.tracking.basic.BoundedActor]. The actor, however,
    assumes that the target has an additional latent velocity, i.e. it
    believes in the model

    ```
    target[t+1]     = target[t] + dt * target_vel[t] + subj_noise * eps
    target_vel[t+1] = target_vel[t] + subj_vel_noise * eps
    ```

    The actor's state is ``[target, cursor]`` for each spatial dimension,
    followed by ``target_vel`` for each spatial dimension.

    Args:
        dim (int): number of spatial dimensions (e.g. 2 for tracking in x and y)
        process_noise (float): standard deviation of the target's random walk
        action_variability (float): standard deviation of the noise on the
            cursor's movement
        action_cost (float): weight of the action cost relative to the distance
            cost
        subj_noise (float): standard deviation of the target noise that the
            actor assumes
        subj_vel_noise (float): standard deviation of the target velocity noise
            that the actor assumes
        sigma_target (float): standard deviation of the target observation noise
        sigma_cursor (float): standard deviation of the cursor observation noise
        dt (float): duration of a time step in seconds
        T (int): number of time steps
    """

    def __init__(
        self,
        dim=1,
        process_noise=1.0,
        action_cost=1.0,
        action_variability=0.5,
        subj_noise=1.0,
        subj_vel_noise=0.5,
        sigma_target=6.0,
        sigma_cursor=6.0,
        dt=1.0 / 60,
        T=1000,
    ):
        A = jnp.eye(2 * dim)
        B = linalg.block_diag(*[jnp.array([[0.0], [1.0 * dt]])] * dim)
        F = jnp.eye(2 * dim)

        V = linalg.block_diag(
            *[jnp.diag(jnp.array([process_noise, action_variability]))] * dim
        )
        W = linalg.block_diag(
            *[jnp.diag(jnp.array([sigma_target, sigma_cursor]))] * dim
        )

        dyn = Dynamics(A=A, B=B, F=F, V=V, W=W, T=T)

        A = linalg.block_diag(
            *[jnp.array([[1.0, 0.0, dt], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])] * dim
        )
        B = linalg.block_diag(*[jnp.array([[0.0], [1.0 * dt], [0.0]])] * dim)
        F = linalg.block_diag(*[jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])] * dim)

        V = linalg.block_diag(
            *[jnp.diag(jnp.array([subj_noise, action_variability, subj_vel_noise]))]
            * dim
        )

        Q = linalg.block_diag(
            *[jnp.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])] * dim
        )
        R = jnp.eye(B.shape[1]) * action_cost

        dims = swap_dims(A.shape[0], dim)

        A = A[dims, :][:, dims]
        B = B[dims, :]
        V = V[dims, :]
        F = F[:, dims]
        Q = Q[dims, :][:, dims]

        act = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=act, dynamics=dyn)
