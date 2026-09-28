import jax.numpy as jnp
import jax
from jax import random
import optimistix as optx
from numpyro.infer.util import log_density
import distrax

from lqg.infer.models import get_model_params, lqg_model
from lqg.tracking import BoundedActor


class Box(distrax.Chain):
    def __init__(self, low=0.0, high=1.0):
        bijectors = [
            distrax.ScalarAffine(shift=low, scale=(high - low)),
            distrax.Sigmoid(),
        ]
        super().__init__(bijectors)


class LogBox(distrax.Chain):
    def __init__(self, low=0.0, high=1.0):
        bijectors = [
            distrax.Lambda(forward=lambda x: 10**x, inverse=jnp.log10),
            distrax.ScalarAffine(
                shift=jnp.log10(low), scale=(jnp.log10(high) - jnp.log10(low))
            ),
            distrax.Sigmoid(),
        ]
        super().__init__(bijectors)


default_constraints = {
    "action_cost": LogBox(low=1e-7, high=1e-2),
    "velocity_cost": LogBox(low=1e-4, high=1.0),
    "force_cost": LogBox(low=1e-6, high=1e-1),
    "sigma_target": Box(0.01, 100.0),
    "action_variability": Box(low=0.01, high=2.0),
    "sigma_cursor": Box(low=0.01, high=100.0),
    "sigma": Box(low=0.01, high=100.0),
    "subj_noise": Box(low=0.01, high=10.0),
    "subj_vel_noise": Box(low=0.01, high=10.0),
}


def sample_within_bounds(key, param_name, constraints=None, shape=(), scale=1.0):
    if constraints is None:
        constraints = default_constraints
    constraint = constraints[param_name]
    unconstrained_sample = random.normal(key, shape) * scale
    constrained_sample = constraint.forward(unconstrained_sample)
    return constrained_sample


def _log_likelihood(log_params, args):
    numpyro_fn, x, model, fixed = args
    params = {
        name: default_constraints[name].forward(value)
        for name, value in log_params.items()
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
    constraints=None,
    **fixed,
):
    if constraints is None:
        constraints = default_constraints

    def _fit(key):
        # sample initial parameters (in unconstrained space)
        initial_params = {
            name: random.normal(
                subkey,
                (),
            )
            for name, subkey in zip(
                get_model_params(model), random.split(key, len(get_model_params(model)))
            )
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
            name: default_constraints[name].forward(value)
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
            name: sample_within_bounds(key, name, default_constraints)
            for name, key in zip(
                get_model_params(PointMassBoundedActor),
                random.split(
                    random.PRNGKey(seed),
                    num=len(get_model_params(PointMassBoundedActor)),
                ),
            )
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
