import jax.numpy as jnp
from jax import random

from lqg.lqg import LQG

from lqg.tracking import BoundedActor, SubjectiveActor


def test_lqg_simulate():
    """ Test that LQG simulation runs. """
    dt = 1. / 60.
    T = 1000

    # parameters
    action_variability = 0.5
    sigma = 6.
    sigma_prop = 3.
    action_cost = 0.5

    A = jnp.eye(2)
    B = jnp.array([[0.], [dt]])
    V = jnp.diag(jnp.array([1., action_variability]))

    F = jnp.eye(2)
    W = jnp.diag(jnp.array([sigma, sigma_prop]))

    Q = jnp.array([[1., -1.],
                   [-1., 1]])

    R = jnp.eye(1) * action_cost

    lqg = LQG(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R)

    x = lqg.simulate(random.PRNGKey(0), x0=jnp.zeros(2), n=10)

    # simply check that it ran through
    assert x.shape == (10, T, 2)


def test_simulate_subjective():
    """ Test that subjective model without subjective component is equal to non-subjective model. """
    bounded_actor = BoundedActor(process_noise=1., sigma_target=6., action_cost=.1, action_variability=.5,
                                 sigma_cursor=3., T=500)
    subjective_actor = SubjectiveActor(process_noise=1., sigma_target=6., action_cost=.1, action_variability=.5,
                                       sigma_cursor=3.,
                                       subj_noise=1., subj_vel_noise=0., T=500)

    x_b = bounded_actor.simulate(rng_key=random.PRNGKey(0), n=20)
    x_s = subjective_actor.simulate(rng_key=random.PRNGKey(0), n=20)

    assert jnp.allclose(x_b, x_s)


def test_belief_tracking_distribution():
    T = 500
    actor = BoundedActor(T=T)

    x = actor.simulate(rng_key=random.PRNGKey(0), n=20)

    assert actor.belief_tracking_distribution(x).shape() == (20, T - 1, actor.actor.A.shape[1])
