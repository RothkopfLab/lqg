import numpy as np
import numpyro
from jax import numpy as jnp
from jax.scipy.stats import norm
from numpyro import distributions as dist

default_prior = {"action_cost": dist.HalfNormal(2.),
                 "sigma_target": dist.HalfNormal(50.),
                 "action_variability": dist.HalfNormal(1.),
                 "signal_dep_noise": dist.HalfNormal(1.),
                 "sigma_cursor": dist.HalfNormal(12.5),
                 "subj_noise": dist.HalfNormal(1.),
                 "subj_vel_noise": dist.HalfNormal(2.),
                 "sigma_target_0": dist.HalfNormal(50.),
                 "sigma_target_1": dist.HalfNormal(50.),
                 "sigma_target_2": dist.HalfNormal(50.),
                 "sigma_target_3": dist.HalfNormal(50.),
                 "sigma_target_4": dist.HalfNormal(50.),
                 "sigma_target_5": dist.HalfNormal(50.),
                 }


def prior():
    return default_prior


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
