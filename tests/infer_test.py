# from jax import random
# from lqg.tracking import IdealObserver
#
#
# def test_kalman_infer():
#     kf = IdealObserver()
#
#     x = kf.simulate(random.PRNGKey(113), n=20, T=500)
#
#     assert kf.conditional_distribution(x).shape() == x.shape
