from jax import random

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
