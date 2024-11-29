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


def lqg_model(x, model_type, process_noise=1.0, dt=1.0 / 60, **fixed_params):
    n, T, d = x.shape

    params = {}
    for name, default in get_model_params(model_type).items():
        if name in fixed_params:
            params[name] = fixed_params[name]
        else:
            params[name] = numpyro.param(
                name, jnp.array(default), constraint=dist.constraints.positive
            )

    lqg = model_type(process_noise=process_noise, dt=dt, T=T, **params)

    numpyro.sample("x", lqg.conditional_distribution(x).to_event(1), obs=x[:, 1:])


def common_lqg_model(x, model_type, process_noise=1.0, dt=1.0 / 60.0, **fixed_params):
    Nc, N, T, d = x.shape

    params = {}
    for name, default in get_model_params(model_type).items():
        if not name == "sigma_target":
            if name in fixed_params:
                params[name] = fixed_params[name]
            else:
                params[name] = numpyro.param(
                    name, jnp.array(default), constraint=dist.constraints.positive
                )

    for n in range(Nc):
        xn = x[n]
        # observation noise
        sigma_n = numpyro.param(
            f"sigma_target_{n}", jnp.array(1.0), constraint=dist.constraints.positive
        )

        lqg = model_type(
            process_noise=process_noise, dt=dt, T=T, sigma_target=sigma_n, **params
        )

        numpyro.sample(
            f"x_{n}", lqg.conditional_distribution(xn).to_event(1), obs=xn[:, 1:]
        )


default_prior = prior()


def shared_params_lqg_model(
    x,
    model_type,
    process_noise=1.0,
    dt=1.0 / 60.0,
    priors=None,
    shared_params=None,
    dim=1,
    **fixed_params,
):
    """numpyro model for LQG models with shared parameter

    Args:
        x: data
        process_noise: target random walk standard deviation
        dt: time step duration
        priors: dict of prio distributions
        shared_params: list of parameter names that are shared across conditions
        dim: dimensionality of the state
        **fixed_params: any parameter that should be fixed (i.e. not estimated from the data)

    """
    # data shape is conditions, trials, time steps, state dimensionality
    Nc, N, T, d = x.shape

    # if not specifying extra priors, use the default ones from above
    if priors is None:
        priors = default_prior

    # which parameters will be shared across conditions?
    if shared_params is None:
        shared_params = []
    shared_params = set(shared_params)

    # get the parameter names of the current model type
    model_params = set(get_model_params(model_type).keys())

    # set up empty dict of parameters
    params = {}

    # these parameters will be shared across conditions
    for name in shared_params.intersection(model_params):
        # if a parameter is given in the fixed_params dict, we assume a fixed (user-specified) value
        if name in fixed_params:
            params[name] = fixed_params[name]
        # if not, we use one prior distribution across conditions
        else:
            params[name] = numpyro.sample(name, priors[name])

    # iterate over conditions
    for n in range(Nc):
        # get the data array for the current condition
        xn = x[n]

        # for those parameters that are not shared across conditions
        for name in model_params - shared_params:
            # define a prior distribution for that parameter in the current condition
            params[name] = numpyro.sample(f"{name}_{n}", priors[name])

        # set up the model, passing both shared and condition-specific parameters via the params dict
        lqg = model_type(process_noise=process_noise, dt=dt, T=T, dim=dim, **params)

        # likelihood
        numpyro.sample(
            f"x_{n}", lqg.conditional_distribution(xn).to_event(1), obs=xn[:, 1:]
        )


# apply priors
lifted_model = numpyro.handlers.lift(lqg_model, prior=default_prior)
lifted_common_model = numpyro.handlers.lift(common_lqg_model, prior=default_prior)
