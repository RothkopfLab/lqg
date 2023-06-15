from jax import random

from lqg.tracking import SubjectiveActor, SignalDependentNoiseTrackingTask


def test_lqg_infer_shapes():
    """ Check that LQG conditional distribution has the correct shapes
    """
    model = SubjectiveActor(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape()[1] == (x.shape[1] - 1)


def test_glqg_infer_shapes():
    """ Check that GLQG conditional distribution has the correct shapes
    """
    model = SignalDependentNoiseTrackingTask(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape()[1] == (x.shape[1] - 1)
