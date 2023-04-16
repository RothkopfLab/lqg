import matplotlib.pyplot as plt
import arviz as az
from jax import random

from lqg.tracking import SignalDependentNoiseTrackingTask
from lqg.infer import infer

if __name__ == '__main__':
    # setup model and simulate data
    model = SignalDependentNoiseTrackingTask(sigma=15., prop_noise=5., motor_noise=0.5, c=0.5, T=500)
    x = model.simulate(random.PRNGKey(123), n=20)

    # visualize trajectories
    plt.plot(x[0, :, 0])
    plt.plot(x[0, :, 1])
    plt.xlabel("time")
    plt.ylabel("position")
    plt.show()

    mcmc = infer(x, model=SignalDependentNoiseTrackingTask, num_samples=5_000, num_warmup=2_000)

    data = az.convert_to_inference_data(mcmc)
    az.plot_pair(data, kind="hexbin")
    plt.show()
