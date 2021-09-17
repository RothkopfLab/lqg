import inspect
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lqg.infer.prior import prior


def get_model_params(model_class):
    init_signature = inspect.signature(model_class.__init__)

    parameters = {}
    for name, param in init_signature.parameters.items():
        if name not in ["self", "dim", "dt", "process_noise", "delay"]:
            parameters[param.name] = param.default

    return parameters


def lqg_model(x, model_type, process_noise=1., dt=1. / 60, **fixed_params):
    params = {}
    for name, default in get_model_params(model_type).items():
        if name in fixed_params:
            params[name] = fixed_params[name]
        else:
            params[name] = numpyro.param(name, jnp.array(default), constraint=dist.constraints.positive)

    # dim = x.shape[2] // 2
    lqg = model_type(process_noise=process_noise, dt=dt, **params)

    if x is not None:
        # numpyro.sample("x0", dist.MultivariateNormal(jnp.zeros(lqg.A.shape[0]), V), obs=x[:, 0])
        numpyro.sample("x", lqg.conditional_distribution(x[:-1]),
                       obs=x[1:].transpose((1, 0, 2)))


def common_lqg_model(x, model_type, process_noise=1., dt=1. / 60., **fixed_params):
    Nc, T, N, d = x.shape

    params = {}
    for name, default in get_model_params(model_type).items():
        if not name == "sigma":
            if name in fixed_params:
                params[name] = fixed_params[name]
            else:
                params[name] = numpyro.param(name, jnp.array(default), constraint=dist.constraints.positive)

    for n in range(Nc):
        xn = x[n]
        # observation noise
        sigma_n = numpyro.param(f"sigma_{n}", jnp.array(1.), constraint=dist.constraints.positive)

        lqg = model_type(process_noise=process_noise, dt=dt, sigma=sigma_n, **params)

        numpyro.sample(f"x_{n}",
                       lqg.conditional_distribution(xn[:-1]),
                       obs=xn[1:].transpose((1, 0, 2)))


def loo_lqg_model(x, model_type, process_noise, dt=1. / 60., **fixed_params):
    Nc, T, N, d = x.shape

    params = {}
    for name, default in get_model_params(model_type).items():
        if name in fixed_params:
            params[name] = fixed_params[name]
        else:
            params[name] = numpyro.param(name, jnp.array(default), constraint=dist.constraints.positive)

    for n in range(Nc):
        xn = x[n]

        lqg = model_type(process_noise=process_noise[n], dt=dt, **params)

        numpyro.sample(f"x_{n}",
                       lqg.conditional_distribution(xn[:-1]),
                       obs=xn[1:].transpose((1, 0, 2)))


# apply priors
lifted_model = numpyro.handlers.lift(lqg_model, prior=prior())
lifted_common_model = numpyro.handlers.lift(common_lqg_model, prior=prior())
lifted_loo_model = numpyro.handlers.lift(loo_lqg_model, prior=prior())
