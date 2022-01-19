import numpy as np
import numpyro.handlers
from jax import numpy as jnp
from jax.scipy.stats import norm
from numpyro import distributions as dist


def prior():
    return {"c": dist.HalfNormal(2.),
            "sigma": dist.HalfNormal(50.),
            "motor_noise": dist.HalfNormal(1.),
            "prop_noise": dist.HalfNormal(12.5),
            "vel_noise": dist.HalfCauchy(10.),
            "subj_noise": dist.HalfNormal(1.),
            "subj_vel_noise": dist.HalfNormal(2.),
            "subj_k": dist.Gamma(.01, .01),
            "subj_lmbd": dist.Gamma(.01, .01),
            "sigma_v": dist.HalfCauchy(50.),
            "sigma_h": dist.HalfCauchy(50.),
            "sigma_z": dist.HalfCauchy(50.),
            "motor_noise_v": dist.HalfCauchy(1.),
            "motor_noise_h": dist.HalfCauchy(1.),
            "prop_noise_v": dist.HalfCauchy(1.),
            "prop_noise_h": dist.HalfCauchy(1.),
            "c_v": dist.HalfCauchy(50.),
            "c_h": dist.HalfCauchy(50.),
            "sigma_0": dist.HalfNormal(50.),
            "sigma_1": dist.HalfNormal(50.),
            "sigma_2": dist.HalfNormal(50.),
            "sigma_3": dist.HalfNormal(50.),
            "sigma_4": dist.HalfNormal(50.),
            "sigma_5": dist.HalfNormal(50.),
            "sigma_test": dist.Gamma(0.01, 0.01)
            }


def lognormal_params(mu, sigma):
    return np.log(mu ** 2 / np.sqrt(mu ** 2 + sigma ** 2)), np.log(1 + sigma ** 2 / mu ** 2)


def lognormal_from_quantiles(x1, x2, p1=0.05, p2=0.95):
    """ Compute the parameters of a log-normal distribution, such that F(x1) = p1 and F(x2) = p2

    Args:
        x1: lower value
        x2: upper value
        p1: lower probability
        p2: upper probability

    Returns:
        mu, sigma (parameters of the log-normal distribution
    """
    sigma = (jnp.log(x2) - jnp.log(x1)) / (norm.ppf(p2) - norm.ppf(p1))
    mu = (jnp.log(x2) * norm.ppf(p2) - jnp.log(x1) * norm.ppf(p1)) / (norm.ppf(p2) - norm.ppf(p1))
    return mu, sigma


def sample_params(prior_dict):
    params = {}
    for param, d in prior_dict.items():
        params[param] = numpyro.sample(param, d)

    return params


if __name__ == '__main__':
    import numpyro
    from jax import random
    import arviz as az
    import matplotlib.pyplot as plt
    import pandas as pd

    from lqg.tracking import OneDimModel as LQG

    # test priors
    param_names = ["sigma", "sigma_test"]

    x = jnp.linspace(0, 100)

    f, axes = plt.subplots(1, len(param_names), sharey=True)
    prior_dict = prior()
    for ax, name in zip(axes, param_names):
        p = prior_dict[name]

        ax.plot(x, jnp.exp(p.log_prob(x)))
        ax.set_title(name)

    plt.show()

    # # prior predictive checks
    # plist = []
    # for seed in range(25):
    #     params = numpyro.handlers.seed(sample_params, rng_seed=seed)(prior_dict)
    #     params = {k: v.item() for k, v in params.items() if k in param_names}
    #     plist.append(params)
    #
    #     lqg = LQG(**params)
    #
    #     x = lqg.simulate(n=20, T=1000, seed=seed)
    #
    #     #     # visualize trajectories
    #     #     plt.plot(x[:, 0, 0])
    #     #     plt.plot(x[:, 0, 1])
    #     #     plt.xlabel("time")
    #     #     plt.ylabel("position")
    #     #     plt.show()
    #
    #     lags, correls = xcorr(jnp.diff(x[..., 1], axis=0).T, jnp.diff(x[..., 0], axis=0).T, maxlags=60)
    #     plt.plot(lags[50:], correls.mean(axis=0)[50:])
    #
    # plt.show()
    #
    # # df = pd.DataFrame(plist)
    #
    # # import seaborn as sns
    # #
    # # sns.pairplot(df)
    # # plt.show()
    #
    # # check HDIs of priors.
    # print(az.hdi(np.array(numpyro.distributions.HalfCauchy(50.).sample(random.PRNGKey(0), (10_000,)))))
