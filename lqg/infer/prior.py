from jax import numpy as jnp
from jax.scipy.stats import norm
from numpyro import distributions as dist


def prior():
    return {"c": dist.Gamma(0.01, 0.01),
            "sigma": dist.Gamma(0.01, 0.01),
            "motor_noise": dist.Gamma(0.01, 0.01),
            "prop_noise": dist.Gamma(0.001, 0.001),
            "vel_noise": dist.Gamma(0.01, 0.01),
            "subj_noise": dist.Gamma(0.01, 0.01),
            "subj_vel_noise": dist.Gamma(0.01, 0.01),
            "subj_k": dist.Gamma(.01, .01),
            "subj_lmbd": dist.Gamma(.01, .01),
            "sigma_v": dist.Gamma(0.01, 0.01),
            "sigma_h": dist.Gamma(0.01, 0.01),
            "sigma_z": dist.Gamma(0.01, 0.01),
            "motor_noise_v": dist.Gamma(0.01, 0.01),
            "motor_noise_h": dist.Gamma(0.01, 0.01),
            "prop_noise_v": dist.Gamma(0.01, 0.01),
            "prop_noise_h": dist.Gamma(0.01, 0.01),
            "c_v": dist.Gamma(0.01, 0.01),
            "c_h": dist.Gamma(0.01, 0.01),
            "sigma_0": dist.Gamma(0.01, 0.01),
            "sigma_1": dist.Gamma(0.01, 0.01),
            "sigma_2": dist.Gamma(0.01, 0.01),
            "sigma_3": dist.Gamma(0.01, 0.01),
            "sigma_4": dist.Gamma(0.01, 0.01),
            "sigma_5": dist.Gamma(0.01, 0.01),
            }


def lognormal_params(x1, x2, p1=0.05, p2=0.95):
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
