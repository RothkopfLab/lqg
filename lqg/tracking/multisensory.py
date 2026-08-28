import jax.numpy as jnp
from jax.scipy import linalg
from lqg import System
from lqg.spec import LQGSpec
from lqg.tracking.point_mass import point_mass_dynamics_matrices
from lqg.utils import time_stack_spec


def multisensory_delay_system(A, B, V, Fs, Ws, Q, R, delays=None, T=500) -> LQGSpec:
    """
    Create a multisensory delay system with the given parameters.

    Args:
        A (jnp.array): Dynamics matrix
        B (jnp.array): Control matrix
        V (jnp.array): Dynamics noise covariance factor
        Fs (list of jnp.array): Sensory feedback matrices
        Ws (list of jnp.array): Sensory noise covariance factors
        Q (jnp.array): State cost matrix
        R (jnp.array): Control cost matrix
        delays (list of int): Delays for each sensory modality
        T (int): Number of time steps

    Returns:
        A time-stacked system specification (LQGSpec)
    """

    if delays is None:
        delays = [0] * len(Fs)

    d = A.shape[1]

    # get the maximum delay across all sensory modalities to determine how many past states we need to include in the extended state vector
    max_delay = max(delays)

    # stack up the dynamics matrices to work with an extended state vector
    # the extended state vector contains the current state and the past states up to the maximum delay
    # this is described in more detail in Izawa & Shadmehr (2008), eqn (3)
    A = linalg.block_diag(A, jnp.diag(jnp.zeros(d * max_delay))) + jnp.diag(
        jnp.ones(d * max_delay), k=-d
    )

    # stack up the control gain matrix to work with the extended state vector
    B = jnp.vstack([B] + [jnp.zeros_like(B)] * max_delay)

    # stack up the sensory feedback matrices to work with the extended state vector
    # here we apply the appropriate delay to each sensory modality by padding with zeros as needed
    # this is described in more detail in Crevecoeur et al. (2016)
    F = jnp.vstack(
        [
            jnp.hstack(
                [
                    jnp.zeros((F.shape[0], F.shape[1] * delay)),
                    F,
                    jnp.zeros((F.shape[0], F.shape[1] * (max_delay - delay))),
                ]
            )
            for F, delay in zip(Fs, delays)
        ]
    )

    # stack up the dynamics noise covariance factors to work with the extended state vector
    V = linalg.block_diag(V, jnp.diag(jnp.zeros(d * max_delay)))
    # the sensory noise covariance factors are block diagonal, with each block corresponding to a different sensory modality
    W = linalg.block_diag(*Ws)

    # stack up the state cost matrix to work with the extended state vector
    # the cost is only applied to the current state, not the past states,
    # which means that the resulting Q is block diagonal with the original Q in the first block and zeros in the remaining blocks
    Q = linalg.block_diag(Q, *[jnp.zeros_like(Q)] * max_delay)

    # create a time-stacked system specification that can be used with the lqg package to solve for the optimal control policy
    spec = time_stack_spec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

    return spec


class MultisensoryBoundedActor(System):
    def __init__(
        self,
        process_noise=1.0,
        sigmas=None,
        action_variability=0.5,
        action_cost=0.1,
        dt=0.075,
        delays=None,
        T=1000,
    ):

        if sigmas is None:
            sigmas = [1.0, 1.0]
        if delays is None:
            delays = [1, 1]

        A = jnp.eye(2)
        B = dt * jnp.array([[0.0], [1.0]])
        F = jnp.array([[1.0, -1.0]])
        V = jnp.diag(jnp.array([process_noise, action_variability]))
        Q = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
        R = jnp.array([[action_cost]])

        spec = multisensory_delay_system(
            A,
            B,
            V,
            [F for _ in sigmas],
            [jnp.diag(jnp.array([sigma])) for sigma in sigmas],
            Q,
            R,
            delays=delays,
            T=T,
        )
        super().__init__(actor=spec, dynamics=spec)


class MultisensoryPointMassBoundedActor(System):
    def __init__(
        self,
        process_noise=1.0,
        sigmas=None,
        action_variability=0.5,
        action_cost=0.1,
        damping=0.0015,
        m=1.0,
        tau=0.066,
        dt=0.075,
        delays=None,
        T=1000,
    ):

        if delays is None:
            delays = [1, 1]
        if sigmas is None:
            sigmas = [1.0, 1.0]

        A, B, V = point_mass_dynamics_matrices(damping, m, tau, action_variability, dt)

        A = linalg.block_diag(jnp.eye(1), A)
        B = jnp.vstack([jnp.zeros((1, 1)), B])
        V = linalg.block_diag(jnp.diag(jnp.array([process_noise])), V)

        F = jnp.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        Q = 500.0 * jnp.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        R = B.T @ B * jnp.array([[action_cost]])

        spec = multisensory_delay_system(
            A,
            B,
            V,
            [F for _ in sigmas],
            [sigma * jnp.eye(2) for sigma in sigmas],
            Q,
            R,
            delays=delays,
            T=T,
        )
        super().__init__(actor=spec, dynamics=spec)


if __name__ == "__main__":
    from jax import random
    from lqg import xcorr
    import matplotlib.pyplot as plt

    for delays in [[0, 0], [0, 12], [12, 0], [12, 12]]:
        model = MultisensoryPointMassBoundedActor(delays=delays, sigmas=[10.0, 20.0])
        x = model.simulate(rng_key=random.PRNGKey(0), n=20)

        vels = jnp.diff(x, axis=-2)
        lags, correls = xcorr(vels[..., 1], vels[..., 0])
        plt.plot(lags, correls.mean(axis=0), label=f"delays={delays}")
    plt.legend()
    plt.show()
