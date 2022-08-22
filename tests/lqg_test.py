import jax.numpy as jnp
from jax import random

from lqg.model import System, Actor, Dynamics
from lqg.tracking import BoundedActor, SubjectiveActor


def test_simulate():
    dt = 1. / 60.

    # parameters
    action_variability = 0.5
    sigma = 6.
    sigma_prop = 3.
    action_cost = 0.5

    A = jnp.eye(2)
    B = jnp.array([[0.], [dt]])
    V = jnp.diag(jnp.array([1., action_variability]))

    C = jnp.eye(2)
    W = jnp.diag(jnp.array([sigma, sigma_prop]))

    Q = jnp.array([[1., -1.],
                   [-1., 1]])

    R = jnp.eye(1) * action_cost

    lqg = System(actor=Actor(A, B, C, V, W, Q, R),
                 dynamics=Dynamics(A, B, C, V, W))

    x = lqg.simulate(random.PRNGKey(0), x0=jnp.zeros(2), n=10, T=1000)

    assert x.shape == (1000, 10, 2)


def test_simulate_subjective():
    bounded_actor = BoundedActor(process_noise=1., sigma=6., c=.1, motor_noise=.5, prop_noise=3.)
    subjective_actor = SubjectiveActor(process_noise=1., sigma=6., c=.1, motor_noise=.5, prop_noise=3.,
                                       subj_noise=1., subj_vel_noise=0.)

    x_b = bounded_actor.simulate(rng_key=random.PRNGKey(0), n=20, T=500)
    x_s = subjective_actor.simulate(rng_key=random.PRNGKey(0), n=20, T=500)

    assert jnp.allclose(x_b, x_s)
