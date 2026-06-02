import jax.numpy as jnp
from jax.scipy import linalg

from lqg.utils import time_stack_spec
from lqg.system import System
from lqg.tracking.point_mass import point_mass_dynamics_matrices


def multisensory_delay_system(A, B, V, Fs, Ws, Q, R, delays=[0, 1], T=500):

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


class RelativeObservationMultisensoryDelayModel(System):
    def __init__(
        self,
        process_noise=1.0,
        sigmas=[1.0, 1.0],
        action_variability=0.5,
        action_cost=0.1,
        damping=0.0015,
        m=1.0,
        tau=0.066,
        dt=1 / 60.0,
        delays=[1, 1],
        T=1000,
    ):

        A, B, V = point_mass_dynamics_matrices(damping, m, tau, action_variability, dt)

        A = linalg.block_diag(jnp.eye(1), A)
        B = jnp.vstack([jnp.zeros((1, 1)), B])
        V = linalg.block_diag(jnp.diag(jnp.array([process_noise])), V)

        Fs = [
            jnp.array([[0.0, 0.0, 1.0, 0.0]]),
            jnp.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        ]
        Ws = [
            jnp.diag(jnp.array([sigmas[0]])),
            jnp.diag(jnp.array([sigmas[1], sigmas[1]])),
        ]
        Q = jnp.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        R = jnp.array([[action_cost]])

        spec = multisensory_delay_system(
            A,
            B,
            V,
            Fs,
            Ws,
            Q,
            R,
            delays=delays,
            T=T,
        )
        super().__init__(actor=spec, dynamics=spec)


class IndependentObservationMultisensoryDelayModel(System):
    def __init__(
        self,
        process_noise=1.0,
        sigma_target=1.0,
        sigmas_cursor=[1.0, 1.0],
        action_variability=0.5,
        action_cost=0.1,
        damping=0.0015,
        m=1.0,
        tau=0.066,
        dt=1 / 60.0,
        delays=[1, 1],
        T=1000,
    ):

        A, B, V = point_mass_dynamics_matrices(damping, m, tau, action_variability, dt)

        A = linalg.block_diag(jnp.eye(1), A)
        B = jnp.vstack([jnp.zeros((1, 1)), B])
        V = linalg.block_diag(jnp.diag(jnp.array([process_noise])), V)

        Fs = [
            jnp.array(
                [[0.0, 0.0, 1.0, 0.0]]
            ),  # the proprioceptive observation contains the cursor velocity, but no information about the target
            jnp.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),  # the visual observation contains the target position, cursor position and cursor velocity
        ]
        Ws = [
            jnp.diag(jnp.array([sigmas_cursor[0]])),
            jnp.diag(jnp.array([sigma_target, sigmas_cursor[1], sigmas_cursor[1]])),
        ]

        Q = jnp.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        R = jnp.array([[action_cost]])

        spec = multisensory_delay_system(
            A,
            B,
            V,
            Fs,
            Ws,
            Q,
            R,
            delays=delays,
            T=T,
        )
        super().__init__(actor=spec, dynamics=spec)


if __name__ == "__main__":
    from jax import random
    import matplotlib.pyplot as plt

    from lqg import xcorr

    delay_vis = 0.15
    delay_prop = 0.075
    delays = [int(delay_prop / (1 / 60.0)), int(delay_vis / (1 / 60.0))]
    print(delays)

    sigma_prop = 10.0
    sigma_target = 10.0

    f, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for sigma in [1.0, 30.0, 100.0]:
        model = RelativeObservationMultisensoryDelayModel(
            delays=delays,
            sigmas=[sigma_prop, jnp.sqrt(sigma_target ** 2 + sigma ** 2)],
            action_cost=1e-3,
        )

        x = model.simulate(random.PRNGKey(0), n=100)

        vels = jnp.diff(x, axis=-2)

        lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=120)

        ax[0].plot(lags, correls.mean(axis=0))
        ax

        model = IndependentObservationMultisensoryDelayModel(
            delays=delays,
            sigma_target=sigma_target,
            sigmas_cursor=[sigma_prop, sigma],
            action_cost=1e-3,
        )

        x = model.simulate(random.PRNGKey(0), n=100)

        vels = jnp.diff(x, axis=-2)

        lags, correls = xcorr(vels[..., 1], vels[..., 0], maxlags=120)

        ax[1].plot(lags, correls.mean(axis=0))

    ax[0].set_xlabel("Lag")
    ax[1].set_xlabel("Lag")
    ax[0].set_ylabel("Correlation")
    plt.show()
