import jax.numpy as jnp
from jax import random

from lqg.spec import LQGSpec
from lqg.model import System
from lqg.kalman import KalmanFilter

from lqg.tracking import BoundedActor, SubjectiveActor


def test_kalman_simulate():
    T = 1000

    # parameters
    sigma = 6.

    A = jnp.eye(1)
    V = jnp.diag(jnp.array([1.]))

    F = jnp.eye(1)
    W = jnp.diag(jnp.array([sigma]))

    A = jnp.stack((A,) * T)
    B = jnp.zeros((T, 1, 1))
    F = jnp.stack((F,) * T)
    V = jnp.stack((V,) * T)
    W = jnp.stack((W,) * T)
    Q = jnp.zeros((T, 1, 1))
    R = jnp.zeros((T, 1, 1))

    kf = KalmanFilter(dynamics=LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R))

    x = kf.simulate(random.PRNGKey(0), x0=jnp.zeros(1), n=10, T=T)

    # simply check that it ran through
    assert x.shape == (1000, 10, 2)


def test_lqg_simulate():
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

    A = jnp.stack((A,) * T)
    B = jnp.stack((B,) * T)
    F = jnp.stack((F,) * T)
    V = jnp.stack((V,) * T)
    W = jnp.stack((W,) * T)
    Q = jnp.stack((Q,) * T)
    R = jnp.stack((R,) * T)

    lqg = System(actor=LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R),
                 dynamics=LQGSpec(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R))

    x = lqg.simulate(random.PRNGKey(0), x0=jnp.zeros(2), n=10, T=T)

    # simply check that it ran through
    assert x.shape == (1000, 10, 2)


def test_simulate_subjective():
    bounded_actor = BoundedActor(process_noise=1., sigma=6., c=.1, motor_noise=.5, prop_noise=3., T=500)
    subjective_actor = SubjectiveActor(process_noise=1., sigma=6., c=.1, motor_noise=.5, prop_noise=3.,
                                       subj_noise=1., subj_vel_noise=0., T=500)

    x_b = bounded_actor.simulate(rng_key=random.PRNGKey(0), n=20, T=500)
    x_s = subjective_actor.simulate(rng_key=random.PRNGKey(0), n=20, T=500)

    assert jnp.allclose(x_b, x_s)
