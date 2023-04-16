import argparse
from numpyro.infer import MCMC, NUTS
from jax import random
import arviz as az

from lqg.io import load_tracking_data
from lqg.infer.models import lifted_common_model as common_lqg_model
from lqg import tracking


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous Psychophysics")
    parser.add_argument("--delay", type=int, default=12,
                        help="Time delay, by which target and mouse position are shifted")
    parser.add_argument("--clip", type=int, default=120,
                        help="Clip the initial n time steps of the data")
    parser.add_argument("--nsamp", type=int, default=10_000,
                        help="Number of samples drawn by NUTS")
    parser.add_argument("--nburnin", type=int, default=2_500,
                        help="Number of burn-in samples.")
    parser.add_argument("--nchain", type=int, default=1)
    parser.add_argument("--model", type=str, default="BoundedActor",
                        help="Model type")
    parser.add_argument("--seed", type=int, default=2,
                        help="Seed for NUTS")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data, bws = load_tracking_data(delay=args.delay, clip=args.clip, subtract_mean=True)

    print(data.shape)

    nuts_kernel = NUTS(common_lqg_model)

    mcmc = MCMC(nuts_kernel, num_warmup=args.nburnin, num_samples=args.nsamp, num_chains=args.nchain)
    mcmc.run(random.PRNGKey(args.seed), data, getattr(tracking, args.model))

    inference_data = az.convert_to_inference_data(mcmc)
    inference_data.to_netcdf(f"results/{args.model}-{args.seed}.nc")
