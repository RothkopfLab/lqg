import jax.numpy as jnp
from jax import random

from lqg.system import LQG

from lqg.tracking import BoundedActor, SubjectiveActor


def test_lqg_infer_shapes():
    """ Check that LQG conditional distribution has the correct shapes
    """
    model = SubjectiveActor(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape()[1] == (x.shape[1] - 1)


def test_lqg_likelihood():
    """ Assert that likelihood does not raise an exception (and does not include nans).
    """

    model = BoundedActor(T=500)

    x = model.simulate(random.PRNGKey(123), n=20)

    assert model.log_likelihood(x).all()

def test_numpyro_distribution():
    """ Test that conversion to numpyro model runs. """
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

    numpyro_model = lqg.to_numpyro()

    # simply check that it runs through
    assert numpyro_model is not None

    x = numpyro_model.sample(random.PRNGKey(0), sample_shape=(10,))

    assert x.shape == (10, T, 2)

    assert numpyro_model.log_prob(x) is not None