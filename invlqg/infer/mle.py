import jax.numpy as jnp
from jax import random, lax
import numpyro
from numpyro.infer import SVI, Trace_ELBO

from invlqg.tracking import OneDimModel
from invlqg.infer.models import lqg_model


def guide(x, model_type, process_noise=None, dt=None, **fixed_params):
    pass


def max_likelihood(x, model=OneDimModel, numpyro_fn=lqg_model, process_noise=1., dt=1. / 60, steps=2_000,
                   step_size=0.01, **fixed):
    # optim = numpyro.optim.ClippedAdam(step_size=step_size)
    optim = numpyro.optim.Adam(step_size=step_size)

    # optim = numpyro.optim.Minimize()
    svi = SVI(numpyro_fn, guide, optim, Trace_ELBO(), x=x, model_type=model, process_noise=process_noise, dt=dt,
              **fixed)
    params, state, losses = svi.run(random.PRNGKey(3), steps)

    return params, losses
