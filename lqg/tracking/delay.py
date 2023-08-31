from jax import numpy as jnp
from jax.scipy import linalg

from lqg.spec import LQGSpec
from lqg.lqg import System, Dynamics, Actor
from lqg.tracking.subjective import SubjectiveActor


def delay_system(spec, delay):
    T = spec.A.shape[0]
    d = spec.A.shape[1]

    A = jnp.stack([linalg.block_diag(A, jnp.diag(jnp.zeros(d * delay))) + jnp.diag(jnp.ones(d * delay), k=-d)
                   for A in
                   spec.A])
    B = jnp.stack([jnp.vstack([B] + [jnp.zeros_like(B)] * delay) for B in spec.B])
    F = jnp.stack([jnp.hstack([jnp.zeros((F.shape[0], F.shape[1] * delay)), F]) for F in spec.F])

    V = jnp.stack([linalg.block_diag(V, jnp.diag(jnp.zeros(d * delay))) for V in spec.V])

    Q = jnp.stack([linalg.block_diag(Q, *[jnp.zeros_like(Q)] * delay) for Q in spec.Q])

    state_dim = Q.shape[1]
    action_dim = spec.R.shape[1]
    obs_dim = spec.W.shape[1]

    q = jnp.zeros((T, state_dim))
    Qf = Q[-1]
    qf = q[-1]
    P = jnp.zeros((T, action_dim, state_dim))
    r = jnp.zeros((T, action_dim))
    Cx = jnp.zeros((T, state_dim, V.shape[-1], state_dim))
    Cu = jnp.zeros((T, state_dim, V.shape[-1], action_dim))
    D = jnp.zeros((T, obs_dim, spec.W.shape[-1], state_dim))

    return LQGSpec(A=A, B=B, F=F, V=V, W=spec.W, Q=Q, R=spec.R, q=q, Qf=Qf, qf=qf, P=P, r=r, Cx=Cx, Cu=Cu, D=D)


class TemporalDelayModel(System):
    def __init__(self, system, delay):
        dyn = delay_system(system.dynamics, delay=delay)
        act = delay_system(system.actor, delay=delay)

        super().__init__(actor=act, dynamics=dyn)


class DelayedSubjectiveActor(TemporalDelayModel):
    def __init__(self, process_noise=1., c=0.5, action_variability=0.5, subj_noise=1., subj_vel_noise=10.,
                 sigma_target=6., sigma_cursor=3., dt=1. / 60):
        system = SubjectiveActor(process_noise=process_noise, action_cost=c, action_variability=action_variability,
                                 subj_noise=subj_noise, subj_vel_noise=subj_vel_noise, sigma_target=sigma_target,
                                 sigma_cursor=sigma_cursor, dt=dt)

        super().__init__(system=system, delay=12)


if __name__ == '__main__':
    from jax import random
    import matplotlib.pyplot as plt

    from lqg import xcorr

    base_model = SubjectiveActor()
    delayed_model = DelayedSubjectiveActor()

    for model in [base_model, delayed_model]:
        x = model.simulate(rng_key=random.PRNGKey(0), n=20)

        lags, corrs = xcorr(jnp.diff(x[..., 1], axis=1), jnp.diff(x[..., 0], axis=1))
        plt.plot(lags, corrs.mean(axis=0))
    plt.show()
