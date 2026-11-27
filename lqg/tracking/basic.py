from jax import numpy as jnp
from jax.scipy import linalg

from lqg.system import System, Actor


class TrackingTask(System):
    """Continuous tracking of a target with a cursor whose velocity is controlled.

    For each spatial dimension, the state is ``[target, cursor]``. The target
    follows a Gaussian random walk, the action sets the cursor's velocity, and
    both positions are observed with independent Gaussian noise. The cost is
    the squared distance between target and cursor plus
    ``action_cost * u^2``. The agent's internal model matches the dynamics.

    Args:
        dim (int): number of spatial dimensions (e.g. 2 for tracking in x and y)
        process_noise (float): standard deviation of the target's random walk
        action_variability (float): standard deviation of the noise on the
            cursor's movement
        sigma_target (float): standard deviation of the target observation noise
        sigma_cursor (float): standard deviation of the cursor observation noise
        action_cost (float): weight of the action cost relative to the distance
            cost
        dt (float): duration of a time step in seconds
        T (int): number of time steps
    """

    def __init__(
        self,
        dim=1,
        process_noise=1.0,
        action_variability=0.5,
        sigma_target=6.0,
        sigma_cursor=6.0,
        action_cost=1.0,
        dt=1.0 / 60.0,
        T=1000,
    ):
        self.dim = dim
        self.process_noise = process_noise
        # dimensionality
        d = 2 * dim
        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.0], [1.0]])] * dim)

        # observation model
        F = jnp.eye(d)

        # noise model
        V = jnp.diag(jnp.array([process_noise, action_variability] * dim))
        W = jnp.diag(jnp.array([sigma_target, sigma_cursor] * dim))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1.0, -1.0], [-1.0, 1.0]])] * dim)
        R = jnp.eye(B.shape[1]) * action_cost

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


class BoundedActor(TrackingTask):
    """Boundedly rational actor in a tracking task, with an action cost.

    The same model as [TrackingTask][lqg.tracking.basic.TrackingTask]; the
    name emphasizes that the action cost makes the actor deviate from
    optimal tracking. See there for the arguments.
    """

    def __init__(
        self,
        dim=1,
        process_noise=1.0,
        action_variability=0.5,
        sigma_target=6.0,
        sigma_cursor=6.0,
        action_cost=1.0,
        dt=1.0 / 60,
        T=1000,
    ):
        super().__init__(
            dim=dim,
            process_noise=process_noise,
            action_variability=action_variability,
            sigma_target=sigma_target,
            sigma_cursor=sigma_cursor,
            action_cost=action_cost,
            dt=dt,
            T=T,
        )


class OptimalActor(TrackingTask):
    """Actor in a tracking task with a negligible action cost.

    The same model as [BoundedActor][lqg.tracking.basic.BoundedActor], but
    with ``action_cost`` fixed to ``1e-3``.

    Args:
        dim (int): number of spatial dimensions (e.g. 2 for tracking in x and y)
        process_noise (float): standard deviation of the target's random walk
        action_variability (float): standard deviation of the noise on the
            cursor's movement
        sigma_target (float): standard deviation of the target observation noise
        sigma_cursor (float): standard deviation of the cursor observation noise
        dt (float): duration of a time step in seconds
        T (int): number of time steps
    """

    def __init__(
        self,
        dim=1,
        process_noise=1.0,
        action_variability=0.5,
        sigma_target=6.0,
        sigma_cursor=6.0,
        dt=1.0 / 60,
        T=1000,
    ):
        super().__init__(
            dim=dim,
            process_noise=process_noise,
            action_variability=action_variability,
            sigma_target=sigma_target,
            sigma_cursor=sigma_cursor,
            action_cost=1e-3,
            dt=dt,
            T=T,
        )


class RelativeObservationBoundedActor(System):
    """Bounded actor that only observes the target position relative to the cursor.

    The same model as [BoundedActor][lqg.tracking.basic.BoundedActor], except
    that the observation is ``target - cursor`` (one per spatial dimension)
    instead of both positions separately.

    Args:
        dim (int): number of spatial dimensions (e.g. 2 for tracking in x and y)
        process_noise (float): standard deviation of the target's random walk
        action_variability (float): standard deviation of the noise on the
            cursor's movement
        sigma (float): standard deviation of the observation noise
        action_cost (float): weight of the action cost relative to the distance
            cost
        dt (float): duration of a time step in seconds
        T (int): number of time steps
    """

    def __init__(
        self,
        dim=1,
        process_noise=1.0,
        action_variability=0.5,
        sigma=6.0,
        action_cost=1.0,
        dt=1.0 / 60.0,
        T=1000,
    ):
        self.dim = dim
        self.process_noise = process_noise

        # dimensionality
        d = 2 * dim

        # dynamics model
        A = jnp.eye(d)
        B = dt * linalg.block_diag(*[jnp.array([[0.0], [1.0]])] * dim)

        # observation model
        F = linalg.block_diag(*[jnp.array([[1.0, -1.0]])] * dim)

        # noise model
        V = jnp.diag(jnp.array([process_noise, action_variability] * dim))
        W = jnp.diag(jnp.array([sigma] * dim))

        # cost function
        Q = linalg.block_diag(*[jnp.array([[1.0, -1.0], [-1.0, 1.0]])] * dim)
        R = jnp.eye(B.shape[1]) * action_cost

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)
