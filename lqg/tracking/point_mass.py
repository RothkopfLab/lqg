import jax.numpy as jnp
from jax.scipy import linalg

from lqg.system import Actor, System


class PointMassBoundedActor(System):
    def __init__(
        self,
        process_noise=1.0,
        action_variability=1e-3,
        sigma_target=6.0,
        action_cost=0.01,
        dt=1.0 / 60.0,
        T=1000,
        m=1.0,
        tau=0.066,
    ):
        A, B, V = point_mass_dynamics_matrices(
            m=m, tau=tau, action_variability=action_variability, dt=dt
        )
        A = linalg.block_diag(jnp.eye(1), A)  # add target position as a constant state
        B = jnp.vstack([jnp.zeros((1, 1)), B])
        V = linalg.block_diag(jnp.diag(jnp.array([process_noise])), V)

        # observation of target position
        F = jnp.eye(1, 4)
        W = jnp.diag(jnp.array([sigma_target]))

        Q = (
            500.0
            * linalg.block_diag(
                *[
                    jnp.array(
                        [
                            [1.0, -1.0, 0.0, 0.0],
                            [-1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    )
                ]
            )
        )  # cost on distance between cursor and target, no cost on velocity or muscle activation
        R = jnp.eye(B.shape[1]) * action_cost * dt

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


def van_loan_discretization(A, G, dt, Qc=None):
    """
    Compute discrete-time process noise covariance Qd.

    Args:
        A: (n, n)
        G: (n, r)
        Qc: (r, r) continuous noise covariance
        dt: scalar

    Returns:
        Qd: (n, n)
    """
    n = A.shape[0]

    if Qc is None:
        Qc = jnp.eye(G.shape[1])

    # Continuous noise covariance mapped into state space
    Q = G @ Qc @ G.T

    # Van Loan matrix
    M = jnp.block([[A, Q], [jnp.zeros_like(A), -A.T]])

    M_exp = linalg.expm(M * dt)

    Qd = M_exp[:n, n:]

    return Qd


def point_mass_dynamics_matrices(m, tau, action_variability, dt):
    # continuous-time dynamics of a point mass with a first-order muscle activation dynamics
    dtt = dt / (tau)
    A = jnp.array(
        [
            [1.0, dt, 0.0],
            [0.0, 1.0, dt / m],
            [0.0, 0.0, 1 - dtt],
        ]
    )
    B = jnp.array([[0.0], [0.0], [dtt]])

    # B scaled by action_variability to represent variability in muscle activation
    V = action_variability * jnp.diag(jnp.array([1e-3, 1e-3, dtt]))

    return A, B, V


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax import random

    from lqg import xcorr
    from lqg.tracking import BoundedActor

    # setup model and simulate data
    dt = 1 / 60.0
    T = int(5 / dt)
    model = PointMassBoundedActor(T=T, action_cost=0.01, action_variability=0.25, dt=dt)
    #
    x = model.simulate(random.PRNGKey(0), x0=jnp.zeros(model.xdim), n=50)

    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, model in enumerate(
        [
            PointMassBoundedActor(
                T=T, action_cost=0.01, action_variability=0.5, tau=0.066
            ),
            PointMassBoundedActor(
                T=T, action_cost=0.01, action_variability=0.5, tau=10.0
            ),
        ]
    ):
        dim_mask = (
            jnp.array([0, 1], dtype=bool)
            if isinstance(model, BoundedActor)
            else jnp.array([0, 1, 1, 1], dtype=bool)
        )

        x_sim = model.simulate_actions(
            x[..., : model.xdim], action_dim_mask=dim_mask, rng_key=random.PRNGKey(0)
        )

        velocities = jnp.diff(x_sim, axis=-2)
        lags, correls = xcorr(velocities[..., 1], velocities[..., 0], maxlags=60)

        # visualize trajectories
        ax[0].plot(jnp.arange(T + 1) * dt, x[0, :, 0])

        ax[0].plot(jnp.arange(T + 1) * dt, x_sim[0, :, 1])
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("position")

        ax[1].plot(lags[60:] * dt, correls.mean(axis=0)[60:])
        ax[1].set_xlabel("lag (s)")
        ax[1].set_ylabel("cross-correlation")

    f.tight_layout()
    plt.show()
