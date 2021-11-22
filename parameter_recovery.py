import argparse
import itertools
import jax.numpy as jnp
import numpyro
import pandas as pd
from jax import random
from numpyro import handlers, distributions as dist
from tqdm import tqdm
import arviz as az
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(description="Coverage runs")
    parser.add_argument("--nsim", type=int, default=10,
                        help="Number of simulations")
    parser.add_argument("--ntrial", type=int, nargs="+", default=[10, 20, 50, 100],
                        help="Number of trials (can be a list).")
    parser.add_argument("--c", type=float, nargs="+", default=[1.],
                        help="List of control cost parameters")
    parser.add_argument("--sigma", type=float, nargs="+", default=[20.],
                        help="List of observation noise parameters")
    parser.add_argument("--motor_noise", type=float, nargs="+", default=[.5],
                        help="List of motor noise parameters")
    parser.add_argument("--prop_noise", type=float, nargs="+", default=[8.],
                        help="List of estimation noise standard deviation parameters")
    parser.add_argument("--use_prior", default=False, action="store_true",
                        help="Use prior instead of parameter lists")
    parser.add_argument("--time", type=int, default=500,
                        help="Time steps per trial")
    parser.add_argument("--nsamp", type=int, default=5000,
                        help="Number of samples drawn by NUTS")
    parser.add_argument("--nburnin", type=int, default=2000,
                        help="Number of burn-in samples.")
    parser.add_argument("--model", type=str, default="OneDimModel",
                        help="Model type (lqg.tracking)")
    parser.add_argument("--fixed", type=str, nargs="+", default=[],
                        help="Parameters fixed to their true value for inference")
    parser.add_argument("--dim", type=int, default=1,
                        help="Dimensionality of the model")
    parser.add_argument("--method", type=str, default="nuts",
                        help="Inference method")
    parser.add_argument("--outfile", type=str, default="sim_results_oldprior.csv",
                        help="File to save results (in results/)")
    return parser.parse_args()


priors = {"c": dist.Uniform(0.01, 5.),
          "sigma": dist.Uniform(1., 50.),
          "motor_noise": dist.Uniform(0.25, 1.),
          "prop_noise": dist.Uniform(1., 10.)
          }


def sample_params():
    params = {}
    for param, d in priors.items():
        params[param] = numpyro.sample(param, d)

    return params


def fit_model(i, params, n_trial):
    c_true = jnp.array(params["c"])
    sigma_true = jnp.array(params["sigma"])
    motor_noise_true = jnp.array(params["motor_noise"])
    prop_noise_true = jnp.array(params["prop_noise"])

    seed = params["seed"]

    lqg = model(process_noise=1., c=c_true.item(), sigma=sigma_true.item(),
                motor_noise=motor_noise_true.item(), prop_noise=prop_noise_true.item(),
                )

    true_params = dict(c=c_true, sigma=sigma_true, motor_noise=motor_noise_true,
                       prop_noise=prop_noise_true)

    x = lqg.simulate(n=n_trial, T=T, seed=seed)

    fixed = {key: val for key, val in true_params.items() if key in args.fixed}

    # Bayesian inference (NUTS, SVI or NeuTra)
    mcmc = infer(x, method=args.method, num_samples=args.nsamp, num_warmup=args.nburnin,
                 model=model, progress_bar=False, seed=seed, **fixed)

    inference_data = az.convert_to_inference_data(mcmc)
    summary = az.summary(inference_data)
    summary["i"] = i
    summary["n_trial"] = n_trial
    summary = summary.reset_index().rename(columns={"index": "param"})
    summary["true"] = summary["param"].map(params)
    return summary


if __name__ == "__main__":

    args = parse_args()

    from invlqg import tracking
    from invlqg.infer import infer

    n_sim = args.nsim
    T = args.time

    results = []

    model = getattr(tracking, args.model)

    if not args.use_prior:
        keys = ["c", "sigma", "motor_noise", "prop_noise", "seed"]
        param_list = itertools.product(args.c, args.sigma, args.motor_noise, args.prop_noise, range(args.nsim))
        param_dicts = [dict(zip(keys, comb)) for comb in param_list]
    else:
        param_dicts = [{**handlers.seed(sample_params, rng_seed=seed)(), "seed": seed} for seed in range(args.nsim)]

    for n_trial in args.ntrial:
        print(f"Running {n_sim} simulations of {n_trial} trials")

        current_results = Parallel(n_jobs=25)(
            delayed(fit_model)(i, params, n_trial) for i, params in enumerate(tqdm(param_dicts)))

        # sites = mcmc._states[mcmc._sample_field]
        # if isinstance(sites, dict):
        #     state_sample_field = attrgetter(mcmc._sample_field)(mcmc._last_state)
        #     # XXX: there might be the case that state.z is not a dictionary but
        #     # its postprocessed value `sites` is a dictionary.
        #     # TODO: in general, when both `sites` and `state.z` are dictionaries,
        #     # they can have different key names, not necessary due to deterministic
        #     # behavior. We might revise this logic if needed in the future.
        #     if isinstance(state_sample_field, dict):
        #         sites = {k: v for k, v in mcmc._states[mcmc._sample_field].items()
        #                  if k in state_sample_field}
        #
        # summary = numpyro.diagnostics.summary(sites)
        #
        # samp = mcmc.get_samples()
        #
        # for var in samp.keys():
        #     if var in true_params.keys():
        #         results.append({"n_trial": n_trial, "true": true_params[var].item(),
        #                         "5.0%": summary[var]["5.0%"], "95.0%": summary[var]["95.0%"],
        #                         "mean": summary[var]["mean"], "std": summary[var]["std"],
        #                         "median": summary[var]["median"],
        #                         "n_eff": summary[var]["n_eff"], "r_hat": summary[var]["r_hat"],
        #                         "param": var, "i": i})
        #

        results += current_results

    pd.concat(results).reset_index(drop=True).to_csv(f"results/{args.outfile}")
