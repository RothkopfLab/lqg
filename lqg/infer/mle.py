import jax.numpy as jnp
import jax
from jax import random
import optimistix as optx
from numpyro.infer.util import log_density
from numpyro.distributions.transforms import SoftplusTransform

from lqg.infer.utils import sample_from_prior


from lqg.infer.models import get_model_params, lqg_model
from lqg.tracking import BoundedActor

params_constraints = {
    "action_cost": SoftplusTransform(),
    "sigma_target": SoftplusTransform(),
    "action_variability": SoftplusTransform(),
    "signal_dep_noise": SoftplusTransform(),
    "sigma_cursor": SoftplusTransform(),
    "sigma": SoftplusTransform(),
    "subj_noise": SoftplusTransform(),
    "subj_vel_noise": SoftplusTransform(),
    "m": SoftplusTransform(),
    "tau": SoftplusTransform(),
}


def _log_likelihood(log_params, args):
    numpyro_fn, x, model, fixed = args
    params = {
        name: params_constraints[name](value) for name, value in log_params.items()
    }
    params.update(fixed)
    return -log_density(
        numpyro_fn,
        (x,),
        {
            "model_type": model,
            **fixed,
        },
        params,
    )[0]


def max_likelihood(
    key,
    x,
    model=BoundedActor,
    numpyro_fn=lqg_model,
    max_steps=2_000,
    **fixed,
):
    def _fit(key):
        initial_params = {
            name: value
            for name, value in sample_from_prior(model, seed=key).items()
            if name not in fixed
        }
        solver = optx.LBFGS(rtol=1e-4, atol=1e-4)
        solution = optx.minimise(
            _log_likelihood,
            solver,
            initial_params,
            args=(numpyro_fn, x, model, fixed),
            max_steps=max_steps,
            throw=False,
        )
        params = {
            name: params_constraints[name](value)
            for name, value in solution.value.items()
        }
        return params, solution.state.f_info.f

    params, nll = jax.vmap(_fit)(random.split(key, 5))
    params = {k: v[jnp.argmin(nll)] for k, v in params.items()}

    return params  # , losses


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from lqg.tracking import PointMassBoundedActor

    fixed_params = {
        "process_noise": 1.0,
        "dt": 1.0 / 60.0,
        # "sigma_cursor": 12.5,
        "tau": 0.066,
    }

    true_values = {
        key: []
        for key in get_model_params(PointMassBoundedActor)
        if key not in fixed_params
    }
    results = {
        key: []
        for key in get_model_params(PointMassBoundedActor)
        if key not in fixed_params
    }
    print("Running maximum likelihood parameter recovery")
    for seed in tqdm(range(100)):
        # true_params sampled from prior
        true_params = {
            name: value
            for name, value in sample_from_prior(
                PointMassBoundedActor, seed=seed
            ).items()
            if name not in fixed_params
        }
        print(f"True parameters: {true_params}")

        lqg = PointMassBoundedActor(
            **fixed_params,
            **true_params,
            T=500,
        )

        x = lqg.simulate(rng_key=random.PRNGKey(seed), n=20)

        params = max_likelihood(
            key=random.PRNGKey(seed),
            x=x[..., :2],
            model=PointMassBoundedActor,
            max_steps=1_000,
            **fixed_params,
        )
        print(f"Estimated parameters: {params}")

        for key, item in params.items():
            true_values[key].append(float(true_params[key]))
            results[key].append(float(item))

    figure, axes = plt.subplots(2, 3, figsize=(10, 8))
    for axis, (name, true, estimates) in zip(
        axes.flat, zip(true_values.keys(), true_values.values(), results.values())
    ):
        axis.scatter(true, estimates, alpha=0.7)
        axis.plot([min(true), max(true)], [min(true), max(true)], "k--", alpha=0.5)
        axis.set_title(name)
        axis.set_xlabel("simulated value")
        axis.set_ylabel("estimated value")
        axis.legend()

    figure.tight_layout()
    plt.show()
