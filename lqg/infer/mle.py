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


def _log_likelihood(log_params, args, constraints):
    numpyro_fn, x, model, fixed = args
    params = {
        name: constraints[name].forward(value) for name, value in log_params.items()
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
    num_restarts=5,
    **fixed,
):
    """Maximum likelihood estimation of the parameters of an LQG model.

    The optimization is performed in an unconstrained space and mapped to the
    parameter space via the bijectors in ``constraints``. It is repeated from
    ``num_restarts`` random initializations, and the solution with the lowest
    negative log likelihood is returned.

    Args:
        key: jax.random.PRNGKey for sampling initial parameters
        x: observed data of shape (n_trials, T, d)
        model: model class (e.g. BoundedActor)
        numpyro_fn: numpyro model function defining the likelihood
        max_steps: maximum number of optimization steps per restart
        constraints: dict mapping parameter names to bijectors (e.g. Box, LogBox).
            Parameters not in this dict use the bounds in ``default_constraints``.
        num_restarts: number of random restarts of the optimization
        **fixed: parameters that are fixed to the given values (not estimated)

    Returns:
        dict: estimated parameters
    """
    constraints = {**default_constraints, **(constraints or {})}

    # bijectors are pytrees, so we close over them instead of passing them as args
    def loss(log_params, args):
        return _log_likelihood(log_params, args, constraints)

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
            loss,
            solver,
            initial_params,
            args=(numpyro_fn, x, model, fixed),
            max_steps=max_steps,
            throw=False,
        )
        params = {
            name: constraints[name].forward(value)
            for name, value in solution.value.items()
        }
        return params, solution.state.f_info.f

    params, nll = jax.vmap(_fit)(random.split(key, num_restarts))
    params = {k: v[jnp.argmin(nll)] for k, v in params.items()}

    return params  # , losses


def parameter_recovery(
    key,
    model,
    num_datasets=100,
    T=500,
    num_trials=20,
    obs_dim=2,
    constraints=None,
    max_steps=1_000,
    num_restarts=10,
    progress=True,
    **fixed,
):
    """Parameter recovery analysis for a model class.

    For each of ``num_datasets`` repetitions, true parameters are sampled within the
    bounds of ``constraints`` (see ``sample_within_bounds``), data are simulated from
    the model, and the parameters are estimated with ``max_likelihood``.

    Args:
        key: jax.random.PRNGKey
        model: model class (e.g. BoundedActor)
        num_datasets: number of simulated data sets
        T: number of time steps per trial
        num_trials: number of trials per data set
        obs_dim: number of state dimensions used for fitting (as in experimental data,
            where only target and cursor position are observed)
        constraints: dict mapping parameter names to bijectors (e.g. Box, LogBox).
            Parameters not in this dict use the bounds in ``default_constraints``.
        max_steps: maximum number of optimization steps per restart
        num_restarts: number of random restarts of the optimization
        progress: show a progress bar
        **fixed: parameters that are fixed to the given values (not estimated)

    Returns:
        dict with entries
            "true" and "estimate": dicts mapping parameter names to arrays of shape (num_datasets,),
            "ll_true" and "ll_estimate": log likelihood of the data at the true and estimated
                parameters. An estimate that is far from the true value but has a higher
                likelihood points to a non-identifiable parameter, a lower likelihood
                to a failed optimization.
    """
    constraints = {**default_constraints, **(constraints or {})}
    free = [name for name in get_model_params(model) if name not in fixed]
    missing = [name for name in free if name not in constraints]
    if missing:
        raise ValueError(
            f"No constraints for {missing}: pass them in `constraints` or fix them."
        )

    results = {
        "true": {name: [] for name in free},
        "estimate": {name: [] for name in free},
    }
    results["ll_true"], results["ll_estimate"] = [], []

    keys = random.split(key, num_datasets)
    if progress:
        from tqdm import tqdm

        keys = tqdm(keys)

    for dataset_key in keys:
        key_params, key_sim, key_fit = random.split(dataset_key, 3)

        true_params = {
            name: sample_within_bounds(subkey, name, constraints)
            for name, subkey in zip(free, random.split(key_params, len(free)))
        }
        x = model(**true_params, **fixed, T=T).simulate(rng_key=key_sim, n=num_trials)
        x = x[..., :obs_dim]

        estimates = max_likelihood(
            key=key_fit,
            x=x,
            model=model,
            constraints=constraints,
            max_steps=max_steps,
            num_restarts=num_restarts,
            **fixed,
        )

        for name in free:
            results["true"][name].append(true_params[name])
            results["estimate"][name].append(estimates[name])
        results["ll_true"].append(
            model(**true_params, **fixed, T=T).log_likelihood(x).sum()
        )
        results["ll_estimate"].append(
            model(**estimates, **fixed, T=T).log_likelihood(x).sum()
        )

    return {
        "true": {name: jnp.array(v) for name, v in results["true"].items()},
        "estimate": {name: jnp.array(v) for name, v in results["estimate"].items()},
        "ll_true": jnp.array(results["ll_true"]),
        "ll_estimate": jnp.array(results["ll_estimate"]),
    }


def plot_parameter_recovery(results, constraints=None):
    """Scatter plots of true vs. estimated parameters from ``parameter_recovery``.

    Parameters with a LogBox constraint are shown on a log scale.
    """
    import matplotlib.pyplot as plt

    constraints = {**default_constraints, **(constraints or {})}
    names = list(results["true"])

    figure, axes = plt.subplots(
        1, len(names), figsize=(3 * len(names), 3), squeeze=False
    )
    for axis, name in zip(axes.flat, names):
        true, estimate = results["true"][name], results["estimate"][name]
        lims = [min(true.min(), estimate.min()), max(true.max(), estimate.max())]
        axis.scatter(true, estimate, alpha=0.7)
        axis.plot(lims, lims, "k--", alpha=0.5)
        axis.set_title(name)
        axis.set_xlabel("simulated value")
        if isinstance(constraints.get(name), LogBox):
            axis.set_xscale("log")
            axis.set_yscale("log")
    axes.flat[0].set_ylabel("estimated value")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    from lqg import tracking

    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(
        description="Parameter recovery for a model from lqg.tracking, with the bounds in default_constraints.",
        epilog="Example: python -m lqg.infer.mle PointMassSubjectiveActor --fix tau=0.066 force_cost=0 --log-bounds action_cost=1e-10,1",
    )
    parser.add_argument(
        "model", help="model class in lqg.tracking, e.g. PointMassBoundedActor"
    )
    parser.add_argument("--num-datasets", type=int, default=100)
    parser.add_argument("--T", type=int, default=500, help="time steps per trial")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--num-restarts", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fix",
        nargs="*",
        default=[],
        metavar="NAME=VALUE",
        help="parameters to fix instead of estimating (dt is fixed to 1/60 unless given)",
    )
    parser.add_argument(
        "--bounds",
        nargs="*",
        default=[],
        metavar="NAME=LOW,HIGH",
        help="override bounds in default_constraints (Box, linear scale)",
    )
    parser.add_argument(
        "--log-bounds",
        nargs="*",
        default=[],
        metavar="NAME=LOW,HIGH",
        help="override bounds in default_constraints (LogBox, log scale)",
    )
    parser.add_argument("--save", help="save the results to this .npz file")
    args = parser.parse_args()

    model = getattr(tracking, args.model)
    fixed_params = {"dt": 1.0 / 60.0}
    for item in args.fix:
        name, value = item.split("=")
        fixed_params[name] = float(value)

    constraints = {}
    for items, bijector in [(args.bounds, Box), (args.log_bounds, LogBox)]:
        for item in items:
            name, bounds = item.split("=")
            low, high = map(float, bounds.split(","))
            constraints[name] = bijector(low=low, high=high)

    results = parameter_recovery(
        random.PRNGKey(args.seed),
        model,
        num_datasets=args.num_datasets,
        T=args.T,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        num_restarts=args.num_restarts,
        constraints=constraints,
        **fixed_params,
    )

    if args.save:
        import numpy as np

        np.savez(
            args.save,
            ll_true=results["ll_true"],
            ll_estimate=results["ll_estimate"],
            **{f"true_{k}": v for k, v in results["true"].items()},
            **{f"estimate_{k}": v for k, v in results["estimate"].items()},
        )

    plot_parameter_recovery(results, constraints)
    plt.show()
