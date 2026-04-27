import argparse
import os

import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
from jax import random
import arviz as az

numpyro.set_host_device_count(4)

from lqg.io import load_tracking_data
from lqg.infer.models import shared_params_lqg_model
from lqg.infer import get_model_params
from lqg import tracking


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Continuous Psychophysics")
    parser.add_argument("--delay", type=int, default=12,
                        help="Time delay, by which target and mouse position are shifted")
    parser.add_argument("--clip", type=int, default=180,
                        help="Clip the initial n time steps of the data")
    parser.add_argument("--nsamp", type=int, default=5_000,
                        help="Number of samples drawn by NUTS")
    parser.add_argument("--nburnin", type=int, default=1_500,
                        help="Number of burn-in samples.")
    parser.add_argument("--nchain", type=int, default=4)
    parser.add_argument("--model", type=str, default="BoundedActor",
                        help="Model type")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (for NUTS)")
    parser.add_argument("--shared_params", type=str, nargs="*",
                        default=["action_variability", "action_cost", "sigma_cursor",
                                 "subj_noise", "subj_vel_noise"],
                        help="Parameters of the model to be shared across conditions")
    parser.add_argument("--output-basename", dest="output_basename", type=str, 
                       default="", metavar="NAME", 
                       help="Basename (no .nc) for data/processed/{NAME}.nc (default: {model}-{seed})")
    parser.add_argument("--data-mat", dest="data_mat", type=str, 
                        default="data.mat", metavar="FILE", 
                        help="Mat file name under data/ or an explicit path (absolute or relative)")
    args = parser.parse_args(args=args, namespace=namespace)

    model_params = get_model_params(getattr(tracking, args.model)).keys()

    args.shared_params = [p for p in args.shared_params if p in list(model_params)]
    return args


if __name__ == '__main__':
    args = parse_args()

    data_mat = args.data_mat
    if os.path.isabs(data_mat) or os.path.dirname(data_mat):
        data_path, mat_filename = os.path.split(data_mat)
        data_path = data_path or "."
    else:
        data_path, mat_filename = "data", data_mat

    data, bws = load_tracking_data(
        delay=args.delay,
        clip=args.clip,
        subtract_mean=False,
        data_path=data_path,
        mat_filename=mat_filename,
    )

    print(data.shape)

    nuts_kernel = NUTS(shared_params_lqg_model, init_strategy=init_to_median)

    mcmc = MCMC(nuts_kernel, num_warmup=args.nburnin, num_samples=args.nsamp, num_chains=args.nchain)
    mcmc.run(random.PRNGKey(args.seed), data, getattr(tracking, args.model), shared_params=args.shared_params)

    inference_data = az.from_numpyro(mcmc)
    out_path = os.path.join("data", "processed", f"{args.output_basename}.nc")
    inference_data.to_netcdf(out_path)
