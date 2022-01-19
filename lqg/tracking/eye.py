import math

from jax import numpy as jnp
from jax.scipy import linalg

from lqg.model import System, Dynamics, Actor
from lqg.kalman import KalmanFilter


class DampedSpringTrackingFilter(KalmanFilter):
    def __init__(self, process_noise=1., sigma=6., dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        A = jnp.array([[1., 1.], [-k, lmbd]])
        C = jnp.array([[1., 0.]])

        V = jnp.diag(jnp.array([1e-7, sigma_w]))
        W = jnp.diag(jnp.array([sigma]))
        super().__init__(A, C, V, W)


class DampedSpringModel(System):
    def __init__(self, process_noise=.2, sigma=1., c=0.1, motor_noise=0.5,
                 dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        prop_noise = 0.

        # dynamics model
        A = jnp.array([[1., 0., 1.], [0., 1., 0], [-k, 0., lmbd]])

        B = dt * jnp.array([[0.], [10.], [0.]])

        # observation model
        C = jnp.array([[1., 0., 0.], [0., 1., 0]])
        # C = jnp.array([[0., 0., 0.] * delay + [1., -1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, motor_noise, sigma_w]))
        W = jnp.diag(jnp.array([sigma, prop_noise]))
        # W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]]))
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DampedSpringCostlessModel(DampedSpringModel):
    def __init__(self, process_noise=.2, sigma=1., motor_noise=0.5, dt=1. / 60.):
        super().__init__(process_noise=process_noise, sigma=sigma, c=0., motor_noise=motor_noise, dt=dt)


class DampedSpringSubjectiveModel(System):
    def __init__(self, process_noise=.2, sigma=1., c=0.1, motor_noise=0.5, subj_noise=.2, subj_k=.0007, subj_lmbd=0.675,
                 dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        prop_noise = 0.

        # dynamics model
        A = jnp.array([[1., 0., 1.], [0., 1., 0], [-k, 0., lmbd]])
        A_subj = jnp.array([[1., 0., 1.], [0., 1., 0], [-subj_k, 0., subj_lmbd]])

        B = dt * jnp.array([[0.], [10.], [0.]])

        # observation model
        C = jnp.array([[1., 0., 0.], [0., 1., 0]])
        # C = jnp.array([[0., 0., 0.] * delay + [1., -1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, motor_noise, sigma_w]))
        V_subj = jnp.diag(jnp.array([1e-7, motor_noise, subj_noise]))
        W = jnp.diag(jnp.array([sigma, prop_noise]))
        # W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]]))
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = Actor(A=A_subj, B=B, C=C, V=V_subj, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DampedSpringSubjectiveVelocityModel(System):
    def __init__(self, process_noise=.2, sigma=1., c=0.1, motor_noise=0.5, subj_noise=.2, subj_vel_noise=1.,
                 dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        prop_noise = 0.

        # dynamics model
        A = jnp.array([[1., 0., 1.], [0., 1., 0], [-k, 0., lmbd]])
        A_subj = jnp.array([[1., 0., dt], [0., 1., 0], [0., 0., 1.]])

        B = dt * jnp.array([[0.], [10.], [0.]])

        # observation model
        C = jnp.array([[1., 0., 0.], [0., 1., 0]])
        # C = jnp.array([[0., 0., 0.] * delay + [1., -1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, motor_noise, sigma_w]))
        V_subj = jnp.diag(jnp.array([subj_noise, motor_noise, subj_vel_noise]))
        W = jnp.diag(jnp.array([sigma, prop_noise]))
        # W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]]))
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = Actor(A=A_subj, B=B, C=C, V=V_subj, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DampedSpringDiffModel(System):
    def __init__(self, process_noise=.2, sigma=1., c=0.1, motor_noise=0.5,
                 dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        prop_noise = 0.

        # dynamics model
        A = jnp.array([[1., 0., 1.], [0., 1., 0], [-k, 0., lmbd]])

        B = dt * jnp.array([[0.], [10.], [0.]])

        # observation model
        C = jnp.array([[1., -1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, motor_noise, sigma_w]))
        W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]]))
        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DampedSpringTwoDimFullModel(System):
    def __init__(self, process_noise=.2, sigma_h=1., sigma_v=1., motor_noise_h=0.5, motor_noise_v=0.5,
                 c_h=.1, c_v=.1, prop_noise_h=0.5, prop_noise_v=0.5,
                 dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        # dynamics model
        A = jnp.array([[1., 0., 0., 0., 1., 0.],
                       [0., 1., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 1.],
                       [0., 0., 0., 1., 0., 0.],
                       [-k, 0., 0., 0., lmbd, 0.],
                       [0., 0., -k, 0., 0., lmbd]])

        B = dt * jnp.array([[0., 0.], [10., 0.], [0., 0.], [0., 10.], [0., 0.], [0., 0.]])

        # observation model
        C = jnp.eye(4, 6)
        # C = jnp.array([[0., 0., 0.] * delay + [1., -1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, motor_noise_h, 1e-7, motor_noise_v, sigma_w, sigma_w]))
        W = jnp.diag(jnp.array([sigma_h, prop_noise_h, sigma_v, prop_noise_v]))
        # W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0., 0., 0., 0.],
                                         [-1., 1., 0., 0., 0., 0.],
                                         [0., 0., 1., -1., 0., 0.],
                                         [0., 0., -1., 1., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.]]))
        R = jnp.diag(jnp.array([c_h, c_v]))

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DampedSpringTwoDimModel(DampedSpringTwoDimFullModel):
    def __init__(self, process_noise=.2, sigma_h=1., sigma_v=1., motor_noise_h=0.5, motor_noise_v=0.5,
                 c_h=.1, c_v=.1,
                 dt=1. / 60.):
        super().__init__(process_noise=process_noise, sigma_h=sigma_h, sigma_v=sigma_v,
                         motor_noise_h=motor_noise_h, motor_noise_v=motor_noise_v,
                         c_h=c_h, c_v=c_v, prop_noise_h=0., prop_noise_v=0.,
                         dt=dt)


class DampedSpringTwoDimCostlessModel(DampedSpringTwoDimModel):
    def __init__(self, process_noise=.2, sigma_h=1., sigma_v=1., motor_noise_h=0.5, motor_noise_v=0.5,
                 dt=1. / 60.):
        super().__init__(process_noise=process_noise, sigma_h=sigma_h, sigma_v=sigma_v,
                         motor_noise_h=motor_noise_h, motor_noise_v=motor_noise_v,
                         c_h=0., c_v=0.,
                         dt=dt)


class DampedSpringTwoDimSubjectiveModel(System):
    def __init__(self, process_noise=.2, sigma_h=1., sigma_v=1., motor_noise_h=0.5, motor_noise_v=0.5,
                 c_h=.1, c_v=.1, subj_noise=2.,
                 dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        # dynamics model
        A = jnp.array([[1., 0., 0., 0., 1., 0.],
                       [0., 1., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 1.],
                       [0., 0., 0., 1., 0., 0.],
                       [-k, 0., 0., 0., lmbd, 0.],
                       [0., 0., -k, 0., 0., lmbd]])

        B = dt * jnp.array([[0., 0.], [10., 0.], [0., 0.], [0., 10.], [0., 0.], [0., 0.]])

        # observation model
        C = jnp.eye(4, 6)
        # C = jnp.array([[0., 0., 0.] * delay + [1., -1., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, motor_noise_h, 1e-7, motor_noise_v, sigma_w, sigma_w]))
        V_subj = jnp.diag(jnp.array([1e-7, motor_noise_h, 1e-7, motor_noise_v, subj_noise, subj_noise]))
        W = jnp.diag(jnp.array([sigma_h, 0., sigma_v, 0.]))
        # W = jnp.diag(jnp.array([sigma]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0., 0., 0., 0.],
                                         [-1., 1., 0., 0., 0., 0.],
                                         [0., 0., 1., -1., 0., 0.],
                                         [0., 0., -1., 1., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.]]))
        R = jnp.diag(jnp.array([c_h, c_v]))

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = Actor(A=A, B=B, C=C, V=V_subj, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DampedSpringVelocityModel(System):
    def __init__(self, process_noise=.2, sigma=1., c=0.01, motor_noise=0.1, dt=1. / 60.):
        lmbd = 0.675
        k = 0.0007
        sigma_w = process_noise

        At = jnp.array([[1., 1], [-k, lmbd]])

        tau1 = 224 / 1000.
        tau2 = 13 / 1000.

        # compute system dynamics and cost matrices
        A0 = jnp.array([[0., 1.], [-1 / (tau1 * tau2), -(tau1 + tau2) / (tau1 * tau2)]])
        Ar = linalg.expm(dt * A0)

        A = linalg.block_diag(At, Ar)

        idx = list(range(A.shape[0]))
        swapdim = idx[::2] + idx[1::2]

        A = A[:, swapdim]
        A = A[swapdim, :]
        # A = linalg.block_diag(A[::2, ::2], A[1::2, 1::2])

        B0 = jnp.array([[0.], [1. / (tau1 * tau2)]])

        # Taylor series approximation of B matrix
        taylor = dt * jnp.array(
            [jnp.linalg.matrix_power(A0 * dt, k) / float(math.factorial(k + 1)) for k in range(50)])

        B = taylor.sum(axis=0) @ B0

        B = jnp.vstack([jnp.zeros_like(B), B])
        B = B[swapdim, :]

        # B = dt * jnp.array([[0.], [0.], [0.], [10.]])

        # observation model
        C = jnp.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        # C = jnp.array([[0., 0., 0., 0.] * delay + [1., -1., 0., 0.]])

        # noise model
        V = jnp.diag(jnp.array([1e-7, 0., sigma_w, 0.])) + motor_noise * B @ B.T
        W = jnp.diag(jnp.array([sigma, 0.]))

        # cost function
        Q = linalg.block_diag(jnp.array([[1., -1., 0., 0.], [-1., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]))

        R = jnp.eye(B.shape[1]) * c

        dyn = Dynamics(A=A, B=B, C=C, V=V)
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = DampedSpringTrackingFilter(sigma=40.)

    x = model.simulate(n=20, T=1000)

    plt.plot(x[:, 0, 0])
    plt.plot(x[:, 0, 1])
    plt.show()
