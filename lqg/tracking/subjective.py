from jax import numpy as jnp

from lqg.lqg import System, Actor, Dynamics


class SubjectiveActor(System):
    def __init__(self, process_noise=1., action_cost=1., action_variability=0.5, subj_noise=1., subj_vel_noise=.5,
                 sigma_target=6., sigma_cursor=6., dt=1. / 60, T=1000):
        A = jnp.eye(2)
        B = jnp.array([[0.], [10. * dt]])
        F = jnp.eye(2)

        V = jnp.diag(jnp.array([process_noise, action_variability]))
        W = jnp.diag(jnp.array([sigma_target, sigma_cursor]))

        dyn = Dynamics(A=A, B=B, F=F, V=V, W=W, T=T)

        A = jnp.array([[1., 0., dt], [0., 1., 0.], [0., 0., 1.]])
        B = jnp.array([[0.], [10. * dt], [0.]])
        F = jnp.array([[1., 0., 0.],
                       [0., 1., 0.]])

        V = jnp.diag(jnp.array([subj_noise, action_variability, subj_vel_noise]))

        Q = jnp.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 0.]])
        R = jnp.eye(B.shape[1]) * action_cost

        act = Actor(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R, T=T)

        super().__init__(actor=act, dynamics=dyn)
