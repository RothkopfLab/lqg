import numpyro
from numpyro import distributions as dist

default_prior = {
    "action_cost": dist.LogNormal(-2.0, 1.0),
    "sigma_target": dist.HalfNormal(50.0),
    "action_variability": dist.HalfNormal(1.0),
    "signal_dep_noise": dist.HalfNormal(1.0),
    "sigma_cursor": dist.HalfNormal(12.5),
    "sigma": dist.HalfNormal(50.0),
    "subj_noise": dist.HalfNormal(1.0),
    "subj_vel_noise": dist.HalfNormal(2.0),
    "sigma_target_0": dist.HalfNormal(50.0),
    "sigma_target_1": dist.HalfNormal(50.0),
    "sigma_target_2": dist.HalfNormal(50.0),
    "sigma_target_3": dist.HalfNormal(50.0),
    "sigma_target_4": dist.HalfNormal(50.0),
    "sigma_target_5": dist.HalfNormal(50.0),
    "tau": dist.HalfNormal(0.1),
}


def prior():
    return default_prior


def sample_params(prior_dict):
    params = {}
    for param, d in prior_dict.items():
        params[param] = numpyro.sample(param, d)

    return params
