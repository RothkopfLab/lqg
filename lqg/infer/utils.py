import jax.numpy as jnp
from jax import random, lax
from numpyro import optim
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam

from lqg.infer.models import lifted_model as lqg_model
from lqg.tracking import DiffModel


def infer(x, num_samples, num_warmup, model=DiffModel, numpyro_fn=lqg_model, process_noise=1., dt=1. / 60,
          method="nuts", progress_bar=True, num_chains=1, seed=0, **fixed):
    if method == "nuts":
        # setup kernel
        nuts_kernel = NUTS(numpyro_fn)

    elif method == "neutra":
        # learn normalizing flow
        guide = AutoBNAFNormal(numpyro_fn)
        svi = SVI(numpyro_fn, guide, optim.Adam(0.003), Trace_ELBO())
        svi_state = svi.init(random.PRNGKey(seed), x, model, process_noise, dt, **fixed)

        last_state, losses = lax.scan(lambda state, i: svi.update(state, x, model, process_noise, dt, **fixed),
                                      svi_state, jnp.zeros(10_000))

        # reparametrize model
        neutra = NeuTraReparam(guide, svi.get_params(last_state))
        nuts_kernel = NUTS(neutra.reparam(numpyro_fn))

    else:
        raise ValueError("Please specify a valid inference method (nuts, neutra).")

    # run NUTS
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, progress_bar=progress_bar,
                num_chains=num_chains)
    mcmc.run(random.PRNGKey(seed), x, model, process_noise, dt, **fixed)

    return mcmc
