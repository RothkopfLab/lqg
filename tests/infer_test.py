from jax import random

from lqg.tracking import BoundedActor, IdealObserver


def test_kalman_infer():
    kf = IdealObserver()

    x = kf.simulate(random.PRNGKey(113), n=20)

    assert kf.conditional_distribution(x).shape() == x.shape


def test_lqg_infer():
    model = BoundedActor(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape() == x.shape
