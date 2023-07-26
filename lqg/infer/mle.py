import jax.numpy as jnp
from jax import random, lax
import numpyro
from numpyro.infer import SVI, Trace_ELBO

from lqg.tracking import BoundedActor
from lqg.infer.models import lqg_model


def guide(x, model_type, process_noise=None, dt=None, **fixed_params):
    pass


def max_likelihood(x, model=BoundedActor, numpyro_fn=lqg_model, process_noise=1., dt=1. / 60, steps=2_000,
                   step_size=0.01, **fixed):
    # optim = numpyro.optim.ClippedAdam(step_size=step_size)
    optim = numpyro.optim.Adam(step_size=step_size)
    # optim = numpyro.optim.Minimize("BFGS")

    # optim = numpyro.optim.Minimize()
    svi = SVI(numpyro_fn, guide, optim, Trace_ELBO(), x=x, model_type=model, process_noise=process_noise, dt=dt,
              **fixed)
    params, state, losses = svi.run(random.PRNGKey(3), steps)

    return params, losses


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import arviz as az

    c_true = .5
    motor_noise_true = .25
    sigma_true = 8.
    prop_noise_true = 2.
    lqg = BoundedActor(process_noise=1.,
                       sigma_target=sigma_true, action_variability=motor_noise_true, action_cost=c_true,
                       sigma_cursor=prop_noise_true, T=500)

    results = dict(action_cost=[], sigma_target=[], action_variability=[], sigma_cursor=[])
    for seed in range(1):
        x = lqg.simulate(rng_key=random.PRNGKey(seed), n=20)

        params, losses = max_likelihood(x, steps=1_000, action_cost=.5, sigma_cursor=2.)

        for key, item in params.items():
            results[key].append(item)

        for key in results.keys():
            results[key] = jnp.stack(results[key])

        az.plot_pair(results, figsize=(8, 8))
        plt.show()
