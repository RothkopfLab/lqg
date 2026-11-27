import jax.numpy as jnp
from jax import random, lax
import numpyro
from numpyro import optim
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam

from lqg.infer.models import lifted_model as lqg_model, get_model_params
from lqg.infer import prior
from lqg.tracking import BoundedActor


def infer(
    x,
    num_samples,
    num_warmup,
    model=BoundedActor,
    numpyro_fn=lqg_model,
    process_noise=1.0,
    dt=1.0 / 60,
    method="nuts",
    progress_bar=True,
    num_chains=1,
    seed=0,
    **fixed,
):
    """Bayesian inference of model parameters with MCMC.

    Samples from the posterior over the parameters of ``model`` given observed
    trajectories, using NUTS. With ``method="neutra"``, a normalizing flow is
    first fit to the posterior with SVI (10,000 steps) and used to
    reparametrize the model, which can help with difficult posterior geometry.

    Args:
        x (jnp.array): observed trajectories of shape (n, T+1, d)
        num_samples (int): number of posterior samples per chain
        num_warmup (int): number of warmup steps per chain
        model (type): model class, e.g.
            [BoundedActor][lqg.tracking.basic.BoundedActor]
        numpyro_fn (callable): numpyro model function defining the priors and
            the likelihood. Defaults to the model in ``lqg.infer.models`` with
            the default priors.
        process_noise (float): target random walk standard deviation, which is
            not estimated
        dt (float): duration of a time step in seconds
        method (str): ``"nuts"`` or ``"neutra"``
        progress_bar (bool): whether to show a progress bar
        num_chains (int): number of MCMC chains
        seed (int): random seed
        **fixed: parameters that are fixed to the given values (not estimated)

    Returns:
        numpyro.infer.MCMC: the MCMC object after running, e.g. use
            ``mcmc.get_samples()`` to get the posterior samples
    """
    if method == "nuts":
        # setup kernel
        nuts_kernel = NUTS(numpyro_fn, init_strategy=init_to_median)

    elif method == "neutra":
        # learn normalizing flow
        guide = AutoBNAFNormal(numpyro_fn)
        svi = SVI(numpyro_fn, guide, optim.Adam(0.003), Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(seed), x, model, process_noise, dt, **fixed)

        last_state, losses = lax.scan(
            lambda state, i: svi.update(state, x, model, process_noise, dt, **fixed),
            svi_state,
            jnp.zeros(10_000),
        )

        # reparametrize model
        neutra = NeuTraReparam(guide, svi.get_params(last_state))
        nuts_kernel = NUTS(neutra.reparam(numpyro_fn))

    else:
        raise ValueError("Please specify a valid inference method (nuts, neutra).")

    # run NUTS
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        progress_bar=progress_bar,
        num_chains=num_chains,
    )
    mcmc.run(random.PRNGKey(seed), x, model, process_noise, dt, **fixed)

    return mcmc


def sample_from_prior(model_type, seed, prior_dict=prior.default_prior):
    param_dict = {
        **numpyro.handlers.seed(prior.sample_params, rng_seed=seed)(prior_dict)
    }
    return {
        k: v for k, v in param_dict.items() if k in get_model_params(model_type).keys()
    }
