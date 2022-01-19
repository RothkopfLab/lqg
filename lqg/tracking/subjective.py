from jax import numpy as jnp
from jax.scipy import linalg

from lqg.model import System, Actor, Dynamics


class SubjectiveModel(System):
    def __init__(self, process_noise=1., c=0.5, motor_noise=0.5, subj_noise=1.,
                 sigma=6., prop_noise=3., dt=1. / 60):
        A = jnp.eye(2)
        B = jnp.array([[0.], [10. * dt]])
        C = jnp.eye(2)

        V = jnp.diag(jnp.array([process_noise, motor_noise]))
        dyn = Dynamics(A=A, B=B, C=C, V=V)

        A = jnp.eye(2)
        B = jnp.array([[0.], [10. * dt]])
        # C = jnp.array([[1., -1.]])
        C = jnp.eye(2)

        V = jnp.diag(jnp.array([subj_noise, motor_noise]))

        Q = jnp.array([[1., -1.], [-1., 1.]])
        R = jnp.eye(B.shape[1]) * c

        W = jnp.diag(jnp.array([sigma, prop_noise]))

        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class SubjectiveVelocityModel(System):
    def __init__(self, process_noise=1., c=0.5, motor_noise=0.5, subj_noise=.1, subj_vel_noise=10.,
                 sigma=6., prop_noise=3., dt=1. / 60):
        A = jnp.eye(2)
        B = jnp.array([[0.], [10. * dt]])
        # A = linalg.block_diag(*[jnp.array([[1.]]), jnp.array([[1., dt], [0., 1.]])])
        # B = jnp.array([[0.], [0.], [10. * dt]])
        # C = jnp.array([[1., -1.]])
        C = jnp.eye(2)
        # C = jnp.array([[1., 0, 0.], [0., 1., 0.]])

        V = jnp.diag(jnp.array([process_noise, motor_noise]))
        dyn = Dynamics(A=A, B=B, C=C, V=V)

        A = jnp.array([[1., 0., dt], [0., 1., 0.], [0., 0., 1.]])
        # A = jnp.array([[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        B = jnp.array([[0.], [10. * dt], [0.]])
        # B = jnp.array([[0.], [0.], [0.], [10. * dt]])
        C = jnp.array([[1., 0., 0.],
                       [0., 1., 0.]])
        # C = jnp.array([[1., 0., 0., 0.],
        #                [0., 1., 0., 0.]])
        # C = jnp.array([[1., -1., 0.]])

        V = jnp.diag(jnp.array([subj_noise, motor_noise, subj_vel_noise]))

        # V = linalg.cholesky(jnp.array([[dt ** 2 / 3 * subj_noise ** 2, 0., dt / 2 * subj_noise ** 2],
        #                                [0., motor_noise ** 2, 0.],
        #                                [dt / 2 * subj_noise ** 2, 0., subj_noise ** 2]]))

        Q = jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]])
        # Q = jnp.array([[1., -1., 0., 0.], [-1., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
        R = jnp.eye(B.shape[1]) * c

        # W = jnp.diag(jnp.array([sigma]))
        W = jnp.diag(jnp.array([sigma, prop_noise]))

        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class SubjectiveVelocityDiffModel(System):
    def __init__(self, process_noise=1., c=0.5, motor_noise=0.5, subj_noise=1., sigma=6., dt=1. / 60):
        A = jnp.eye(2)
        B = jnp.array([[0.], [10. * dt]])
        C = jnp.array([[1., -1.]])

        V = jnp.diag(jnp.array([process_noise, motor_noise]))
        dyn = Dynamics(A=A, B=B, C=C, V=V)

        A = jnp.array([[1., 0., dt], [0., 1., 0.], [0., 0., 1.]])
        B = jnp.array([[0.], [10. * dt], [0.]])
        C = jnp.array([[1., -1., 0.]])

        # V = jnp.diag(jnp.array([0., motor_noise, subj_noise]))
        V = linalg.cholesky(jnp.array([[dt ** 2 / 3 * subj_noise ** 2, 0., dt / 2 * subj_noise ** 2],
                                       [0., motor_noise ** 2, 0.],
                                       [dt / 2 * subj_noise ** 2, 0., subj_noise ** 2]]))

        Q = jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]])
        R = jnp.eye(B.shape[1]) * c

        W = jnp.diag(jnp.array([sigma]))

        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class TemporalDelayModel(System):
    def __init__(self, system, delay=3):
        dyn = system.dynamics
        d = dyn.A.shape[0]

        A = linalg.block_diag(dyn.A, jnp.diag(jnp.zeros(d * delay))) + jnp.diag(jnp.ones(d * delay), k=-d)

        B = jnp.vstack([dyn.B] + [jnp.zeros_like(dyn.B)] * delay)

        C = jnp.hstack([jnp.zeros((dyn.C.shape[0], dyn.C.shape[1] * delay)), dyn.C])

        V = linalg.block_diag(dyn.V, jnp.diag(jnp.zeros(d * delay)))

        dyn = Dynamics(A=A, B=B, C=C, V=V)

        act = system.actor
        d = act.A.shape[0]

        A = linalg.block_diag(act.A, jnp.diag(jnp.zeros(d * delay))) + jnp.diag(jnp.ones(d * delay), k=-d)

        B = jnp.vstack([act.B] + [jnp.zeros_like(act.B)] * delay)

        C = jnp.hstack([jnp.zeros((act.C.shape[0], act.C.shape[1] * delay)), act.C])

        V = linalg.block_diag(act.V, jnp.diag(jnp.zeros(d * delay)))

        W = act.W

        Q = linalg.block_diag(act.Q, *[jnp.zeros_like(act.Q)] * delay)
        R = act.R
        act = Actor(A=A, B=B, C=C, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)


class DelayedSubjectiveVelocityModel(TemporalDelayModel):
    def __init__(self, process_noise=1., c=0.5, motor_noise=0.5, subj_noise=1., subj_vel_noise=10.,
                 sigma=6., prop_noise=3., dt=1. / 60):
        system = SubjectiveVelocityModel(process_noise=process_noise, c=c, motor_noise=motor_noise,
                                         subj_noise=subj_noise, subj_vel_noise=subj_vel_noise, sigma=sigma,
                                         prop_noise=prop_noise, dt=dt)

        super().__init__(system=system, delay=12)


if __name__ == '__main__':
    from lqg.analysis.ccg import xcorr

    import matplotlib.pyplot as plt

    for model in [
        SubjectiveVelocityModel(process_noise=1., sigma=6., c=.1, motor_noise=.5, subj_noise=1.,
                                subj_vel_noise=10.),
        SubjectiveVelocityModel(process_noise=1., sigma=6., c=.1, motor_noise=.5, subj_noise=1.,
                                subj_vel_noise=0.)
    ]:  # ,
        # TemporalDelayModel(SubjectiveVelocityModel(process_noise=1., sigma=6., c=.5, motor_noise=.5, subj_noise=1.,
        #                                            subj_vel_noise=3.), delay=12)]:

        x = model.simulate(n=20, T=500, seed=0)

        # plt.plot(x[:, 0, 0])
        # plt.plot(x[:, 0, 1])
        # plt.show()

        lags, correls = xcorr(jnp.diff(x[..., 1], axis=0).T, jnp.diff(x[..., 0], axis=0).T, maxlags=120)
        plt.plot(lags[100:], correls.mean(axis=0)[100:])

    plt.axhline(0, linestyle="--", color="gray")
    plt.show()

    # import numpy as np
    #
    # rw_std = 1.
    # dt = 1. / 60.
    # A = np.array([[1., dt], [0., 1.]])
    # V = np.diag(np.array([rw_std, 0.]))
    #
    # dyn = Dynamics(A, None, None, V)
    # x = dyn.simulate(n=20, T=1000)
    #
    # Vs = np.diag(np.array([0.8, .5]))
    #
    # dyns = Dynamics(A, None, None, Vs)
    # y = dyns.simulate(n=20, T=1000)
    #
    # f, ax = plt.subplots(2)
    # ax[0].plot(x[..., 0])
    # ax[1].plot(y[..., 0])
    # plt.show()

    # mcmc = infer(x, model=model, num_samples=1000, num_warmup=500)
    # import arviz as az
    #
    # az.plot_pair(mcmc.get_samples(), kind="hexbin")
    # plt.show()

    # import inspect
    #
    # signature = inspect.signature(DiffModel.__init__)
