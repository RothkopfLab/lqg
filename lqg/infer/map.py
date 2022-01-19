import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist

from lqg.infer.models import common_lqg_model

prior = {"c": dist.Gamma(0.01, 0.01), "motor_noise": dist.Gamma(0.01, 0.01),
         "sigma_0": dist.Gamma(0.01, 0.01), "sigma_1": dist.Gamma(0.01, 0.01), "sigma_2": dist.Gamma(0.01, 0.01),
         "sigma_3": dist.Gamma(0.01, 0.01), "sigma_4": dist.Gamma(0.01, 0.01), "sigma_5": dist.Gamma(0.01, 0.01)
         }
hierarchical_lqg_model = numpyro.handlers.lift(common_lqg_model, prior)


def lqg_guide(x):
    # control costs
    delta_c = numpyro.param("c_loc", jnp.array(1.), constraint=dist.constraints.positive)
    c = numpyro.sample("c", dist.Delta(delta_c))

    # observation noise
    delta_sigma = numpyro.param("sigma_loc", jnp.array(1.), constraint=dist.constraints.positive)
    sigma = numpyro.sample("sigma", dist.Delta(delta_sigma))

    # motor variability
    delta_motor_noise = numpyro.param("motor_noise_loc", jnp.array(1.), constraint=dist.constraints.positive)
    motor_noise = numpyro.sample("motor_noise", dist.Delta(delta_motor_noise))


def hierarchical_guide(x):
    Nc, N, T, d = x.shape

    # control costs
    delta_c = numpyro.param("c_loc", jnp.array(1.), constraint=dist.constraints.positive)
    c = numpyro.sample("c", dist.Delta(delta_c))

    # motor variability
    delta_motor_noise = numpyro.param("motor_noise_loc", jnp.array(1.), constraint=dist.constraints.positive)
    motor_noise = numpyro.sample("motor_noise", dist.Delta(delta_motor_noise))

    for n in range(Nc):
        # observation noise
        delta_sigma_n = numpyro.param(f"sigma_{n}_loc", jnp.array(1.), constraint=dist.constraints.positive)
        sigma_n = numpyro.sample(f"sigma_{n}", dist.Delta(delta_sigma_n))
