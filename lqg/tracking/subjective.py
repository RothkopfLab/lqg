from jax import numpy as jnp

from lqg.spec import LQGSpec
from lqg.lqg import System

class SubjectiveActor(System):
    def __init__(self, process_noise=1., c=0.5, motor_noise=0.5, subj_noise=.1, subj_vel_noise=10.,
                 sigma=6., prop_noise=3., dt=1. / 60, T=1000):
        A = jnp.eye(2)
        B = jnp.array([[0.], [10. * dt]])
        F = jnp.eye(2)

        V = jnp.diag(jnp.array([process_noise, motor_noise]))
        W = jnp.diag(jnp.array([sigma, prop_noise]))

        Q = jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]])
        R = jnp.eye(B.shape[1]) * c

        A = jnp.stack((A,) * T)
        B = jnp.stack((B,) * T)
        F = jnp.stack((F,) * T)
        V = jnp.stack((V,) * T)
        W = jnp.stack((W,) * T)
        Q = jnp.stack((Q,) * T)
        R = jnp.stack((R,) * T)

        dyn = LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R)

        A = jnp.array([[1., 0., dt], [0., 1., 0.], [0., 0., 1.]])
        B = jnp.array([[0.], [10. * dt], [0.]])
        F = jnp.array([[1., 0., 0.],
                       [0., 1., 0.]])


        V = jnp.diag(jnp.array([subj_noise, motor_noise, subj_vel_noise]))

        A = jnp.stack((A,) * T)
        B = jnp.stack((B,) * T)
        F = jnp.stack((F,) * T)
        V = jnp.stack((V,) * T)

        act = LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R)

        super().__init__(actor=act, dynamics=dyn)