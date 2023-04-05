from jax import random

from lqg.tracking import BoundedActor, IdealObserver, SubjectiveActor


def test_kalman_infer_shapes():
    """ Check that KF conditional distribution has the correct shapes
    """
    kf = IdealObserver(T=500)

    x = kf.simulate(random.PRNGKey(113), n=20)

    assert kf.conditional_distribution(x).shape() == x.shape


def test_lqg_infer_shapes():
    """ Check that LQG conditional distribution has the correct shapes
    """
    model = SubjectiveActor(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape() == x.shape
