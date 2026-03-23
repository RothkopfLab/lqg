import jax.numpy as jnp
from jax import random
from numpyro import handlers

from lqg.infer.utils import infer
from lqg.system import LQG
from lqg.tracking import BoundedActor, SubjectiveActor


def test_lqg_infer_shapes():
    """Check that LQG conditional distribution has the correct shapes"""
    model = SubjectiveActor(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape()[1] == (x.shape[1] - 1)


def test_lqg_likelihood():
    """Assert that likelihood does not raise an exception (and does not include nans)."""

    model = BoundedActor(T=500)

    x = model.simulate(random.PRNGKey(123), n=20)

    assert model.log_likelihood(x).all()


def test_numpyro_distribution():
    """Test that conversion to numpyro model runs."""

    T = 500
    model = BoundedActor(T=T)

    numpyro_model = model.to_numpyro()

    # simply check that it runs through
    assert numpyro_model is not None

    x = numpyro_model.sample(random.PRNGKey(0), sample_shape=(10,))

    assert x.shape == (10, T + 1, 2)

    assert numpyro_model.log_prob(x) is not None

    key = random.PRNGKey(2)
    assert handlers.seed(numpyro_model, rng_seed=0)(rng_key=key).shape == (T + 1, 2)

    mcmc = infer(x, num_samples=10, num_warmup=10, model=BoundedActor)

    assert mcmc.get_samples() is not None
