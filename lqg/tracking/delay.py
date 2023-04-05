from jax import numpy as jnp
from jax.scipy import linalg

from lqg.model import System
from lqg.tracking.subjective import SubjectiveVelocityModel


class TemporalDelayModel(System):
    def __init__(self, system, delay=3):
        dyn = system.dynamics
        d = dyn.A.shape[0]

        A = linalg.block_diag(dyn.A, jnp.diag(jnp.zeros(d * delay))) + jnp.diag(jnp.ones(d * delay), k=-d)

        B = jnp.vstack([dyn.B] + [jnp.zeros_like(dyn.B)] * delay)

        C = jnp.hstack([jnp.zeros((dyn.C.shape[0], dyn.C.shape[1] * delay)), dyn.C])

        V = linalg.block_diag(dyn.V, jnp.diag(jnp.zeros(d * delay)))
        W = dyn.W

        dyn = Dynamics(A=A, B=B, C=C, V=V, W=W)

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
