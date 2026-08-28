import jax.numpy as jnp
from jax.scipy import linalg
from lqg import System
from lqg.spec import LQGSpec
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
