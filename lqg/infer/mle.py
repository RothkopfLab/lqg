import jax.numpy as jnp
from jax import random
import optimistix as optx
from numpyro.infer.util import log_density

from lqg.tracking import BoundedActor
from lqg.infer.models import get_model_params, lqg_model


def _log_likelihood(log_params, args):
    numpyro_fn, x, model, process_noise, dt, fixed = args
    params = {name: jnp.exp(value) for name, value in log_params.items()}
    params.update(fixed)
    return -log_density(
        numpyro_fn,
        (x,),
        {
            "model_type": model,
            "process_noise": process_noise,
            "dt": dt,
            **fixed,
        },
        params,
    )[0]


def max_likelihood(
    x,
    model=BoundedActor,
    numpyro_fn=lqg_model,
    process_noise=1.0,
    dt=1.0 / 60,
    max_steps=2_000,
    **fixed,
):
    initial_params = {
        name: jnp.log(jnp.asarray(default))
        for name, default in get_model_params(model).items()
        if name not in fixed
    }
    solver = optx.LBFGS(rtol=1e-3, atol=1e-3)
    solution = optx.minimise(
        _log_likelihood,
        solver,
        initial_params,
        args=(numpyro_fn, x, model, process_noise, dt, fixed),
        max_steps=max_steps,
        throw=False,
    )
    params = {name: jnp.exp(value) for name, value in solution.value.items()}
    params.update(fixed)
    # losses = jnp.asarray([solution.stats["f" ]])

    return params  # , losses


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    true_params = dict(
        action_cost=0.5,
        action_variability=0.25,
        sigma_target=8.0,
        sigma_cursor=2.0,
    )
    lqg = BoundedActor(
        process_noise=1.0,
        **true_params,
        T=500,
    )

    results = {key: [] for key in true_params}
    print("Running maximum likelihood parameter recovery")
    for seed in tqdm(range(100)):
        x = lqg.simulate(rng_key=random.PRNGKey(seed), n=20)

        params = max_likelihood(x, max_steps=1_000)

        for key, item in params.items():
            results[key].append(float(item))

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (name, estimates) in zip(axes.flat, results.items()):
        axis.scatter(range(len(estimates)), estimates, alpha=0.7)
        axis.axhline(
            true_params[name], color="black", linestyle="--", label="true value"
        )
        axis.set_title(name)
        axis.set_xlabel("simulation seed")
        axis.set_ylabel("estimated value")
        axis.legend()

    figure.tight_layout()
    plt.show()
