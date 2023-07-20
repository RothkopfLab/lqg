import argparse
import matplotlib.pyplot as plt
import arviz as az
from jax import random
import numpyro

numpyro.set_host_device_count(4)

from lqg import tracking
from lqg.infer import infer
from lqg.infer.utils import sample_from_prior


def parse_args():
    parser = argparse.ArgumentParser(description="Coverage runs")
    parser.add_argument("--ntrial", type=int, default=20,
                        help="Number of trials .")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the simulation")
    parser.add_argument("--time", type=int, default=720,
                        help="Time steps per trial")
    parser.add_argument("--nsamp", type=int, default=5_000,
                        help="Number of samples drawn by NUTS")
    parser.add_argument("--nwarmup", type=int, default=2_500,
                        help="Number of burn-in samples.")
    parser.add_argument("--nchain", type=int, default=4,
                        help="Number of chains.")
    parser.add_argument("--model", type=str, default="BoundedActor",
                        help="Model type (lqg.tracking)")
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    Model = getattr(tracking, args.model)

    params = sample_from_prior(Model, args.seed)

    # setup model and simulate data
    model = Model(T=args.time, **params)

    x = model.simulate(random.PRNGKey(args.seed), n=args.ntrial)

    if args.plot:
        # visualize trajectories
        plt.plot(x[0, :, 0])
        plt.plot(x[0, :, 1])
        plt.xlabel("time")
        plt.ylabel("position")
        plt.show()

    mcmc = infer(x, model=Model,
                 num_samples=args.nsamp, num_warmup=args.nwarmup, num_chains=args.nchain)

    idata = az.convert_to_inference_data(mcmc)

    if args.plot:
        az.plot_pair(idata)
        plt.show()

    summary = az.summary(idata)
    for key in params:
        summary.loc[key, "true"] = params[key]
    summary.to_csv(f"results/parameter-recovery/{args.model}-{args.seed}.nc")
