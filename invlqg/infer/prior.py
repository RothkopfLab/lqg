from jax import numpy as jnp
from jax.scipy.stats import norm
from numpyro import distributions as dist


def prior():
    return {"c": dist.HalfCauchy(50.),
            "sigma": dist.HalfCauchy(50.),
            "motor_noise": dist.HalfCauchy(1.),
            "prop_noise": dist.HalfCauchy(25.),
            # TODO: come up with good priors
            "vel_noise": dist.Gamma(0.01, 0.01),
            "subj_noise": dist.Gamma(0.01, 0.01),
            "subj_vel_noise": dist.Gamma(0.01, 0.01),
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
            "sigma_0": dist.HalfCauchy(50.),
            "sigma_1": dist.HalfCauchy(50.),
            "sigma_2": dist.HalfCauchy(50.),
            "sigma_3": dist.HalfCauchy(50.),
            "sigma_4": dist.HalfCauchy(50.),
            "sigma_5": dist.HalfCauchy(50.),
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
