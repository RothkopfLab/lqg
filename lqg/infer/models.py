import inspect
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lqg.infer.prior import prior


def get_model_params(model_class):
    init_signature = inspect.signature(model_class.__init__)

    parameters = {}
    for name, param in init_signature.parameters.items():
        if name not in ["self", "dim", "dt", "T", "process_noise", "delay", "covar"]:
            parameters[param.name] = param.default

    return parameters


def lqg_model(x, model_type, process_noise=1., dt=1. / 60, **fixed_params):
    n, T, d = x.shape

    params = {}
    for name, default in get_model_params(model_type).items():
        if name in fixed_params:
            params[name] = fixed_params[name]
        else:
            params[name] = numpyro.param(name, jnp.array(default), constraint=dist.constraints.positive)

    lqg = model_type(process_noise=process_noise, dt=dt, T=T, **params)

    numpyro.sample("x", lqg.conditional_distribution(x).to_event(1), obs=x[:, 1:])


def common_lqg_model(x, model_type, process_noise=1., dt=1. / 60., **fixed_params):
    Nc, N, T, d = x.shape

    params = {}
    for name, default in get_model_params(model_type).items():
        if not name == "sigma_target":
            if name in fixed_params:
                params[name] = fixed_params[name]
            else:
                params[name] = numpyro.param(name, jnp.array(default), constraint=dist.constraints.positive)

    for n in range(Nc):
        xn = x[n]
        # observation noise
        sigma_n = numpyro.param(f"sigma_target_{n}", jnp.array(1.), constraint=dist.constraints.positive)

        lqg = model_type(process_noise=process_noise, dt=dt, T=T, sigma_target=sigma_n, **params)

        numpyro.sample(f"x_{n}",
                       lqg.conditional_distribution(xn).to_event(1),
                       obs=xn[:, 1:])

default_prior = prior()

# apply priors
lifted_model = numpyro.handlers.lift(lqg_model, prior=default_prior)
lifted_common_model = numpyro.handlers.lift(common_lqg_model, prior=default_prior)
