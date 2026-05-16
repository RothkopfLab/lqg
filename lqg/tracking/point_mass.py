import jax.numpy as jnp
from jax.scipy import linalg

from lqg.system import Actor, System


class PointMassBoundedActor(System):
    def __init__(
        self,
        process_noise=1.0,
        action_variability=1e-3,
        sigma_target=6.0,
        sigma_cursor=6.0,
        action_cost=.01,
        dt=1.0 / 60.0,
        T=1000,
        damping=0.1,
        m=1.0,
        tau=0.0015,
    ):
        A, B, V = point_mass_dynamics_matrices(
            damping=damping, m=m, tau=tau, action_variability=action_variability, dt=dt
        )
        A = linalg.block_diag(jnp.eye(1), A)  # add target position as a constant state
        B = jnp.vstack([jnp.zeros((1, 1)), B])
        V = linalg.block_diag(jnp.diag(jnp.array([process_noise])), V)

        F = jnp.eye(3, 4)  # full observation of position, no observation of velocity
        W = jnp.diag(jnp.array([sigma_target, sigma_cursor, sigma_cursor]))

        Q = linalg.block_diag(
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
        )  # cost on distance between cursor and target, no cost on velocity or muscle activation
        R = jnp.eye(B.shape[1]) * action_cost

        spec = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=spec, dynamics=spec)


def discretize_linear_system(A, B, dt):
    """
    Discretize continuous-time system x' = A x + B u
    using zero-order hold (exact method).

    Args:
        A: (n, n) array
        B: (n, m) array
        dt: scalar timestep

    Returns:
        Ad: (n, n) array
        Bd: (n, m) array
    """
    n = A.shape[0]
    m = B.shape[1]

    # Construct block matrix
    M = jnp.zeros((n + m, n + m))
    M = M.at[:n, :n].set(A)
    M = M.at[:n, n:].set(B)

    # Matrix exponential
    M_exp = linalg.expm(M * dt)

    # Extract Ad and Bd
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:]

    return Ad, Bd


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


def point_mass_dynamics_matrices(damping, m, tau, action_variability, dt):
    # continuous-time dynamics of a point mass with viscous damping and a first-order muscle activation dynamics
    A_c = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, -damping / m, 1.0 / m], [0.0, 0.0, -1.0 / tau]]
    )
    B_c = jnp.array([[0.0], [0.0], [1.0 / tau]])

    # discretize dynamics
    A, B = discretize_linear_system(A_c, B_c, dt)
    # discretize noise model using van Loan's method, which accounts for the effect of the control input on the noise covariance (makes fitting more stable)
    V = linalg.cholesky(
        make_psd(van_loan_discretization(A_c, action_variability * B_c, dt))
    )

    return A, B, V


def make_psd(M, eps=1e-6):
    """
    Make a symmetric matrix positive semi-definite by adding a small value to the diagonal.

    Args:
        M: (n, n) array
        eps: scalar, small value to add to the diagonal
    Returns:
        M_psd: (n, n) array, positive semi-definite version of M
    """
    M_sym = (M + M.T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(M_sym)
    eigvals_clipped = jnp.clip(eigvals, min=eps)
    M_psd = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
    return M_psd


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax import random

    # setup model and simulate data
    model = PointMassBoundedActor(
        T=500,
        action_cost=1e-5
    )
    print(model.actor.V[0])

    x = model.simulate(random.PRNGKey(0), n=20)

    # visualize trajectories
    plt.plot(x[0, :, 0])
    plt.plot(x[0, :, 1])
    plt.xlabel("time")
    plt.ylabel("position")
    plt.show()
