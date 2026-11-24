import jax.numpy as jnp
from jax import random
from numpyro import handlers

from lqg.infer.utils import infer
from lqg.infer.mle import LogBox, max_likelihood, parameter_recovery
from lqg.tracking import BoundedActor, PointMassSubjectiveActor, SubjectiveActor


def test_lqg_infer_shapes():
    """Check that LQG conditional distribution has the correct shapes"""
    model = SubjectiveActor(T=500)

    x = model.simulate(random.PRNGKey(113), n=20)

    assert model.conditional_distribution(x).shape()[1] == (x.shape[1] - 1)


def test_lqg_likelihood():
    """Assert that likelihood does not raise an exception (and does not include nans)."""

    model = BoundedActor(T=500)

    x = model.simulate(random.PRNGKey(123), n=20)

    assert model.log_likelihood(x).all()


def test_lqg_likelihood_with_initial_state():
    """Check that likelihood works with different given initial state."""

    model = BoundedActor(T=500)

    x = model.simulate(random.PRNGKey(123), n=20)

    assert model.log_likelihood(x, x0=jnp.zeros((x.shape[0], 1))).all()


def test_numpyro_distribution():
    """Test that conversion to numpyro model runs."""

    T = 500
    model = BoundedActor(T=T)

    numpyro_model = model.to_numpyro()

    # simply check that it runs through
    assert numpyro_model is not None

    x = numpyro_model.sample(random.PRNGKey(0), sample_shape=(10,))

    assert x.shape == (10, T + 1, 2)

    assert numpyro_model.log_prob(x) is not None

    key = random.PRNGKey(2)
    assert handlers.seed(numpyro_model, rng_seed=0)(rng_key=key).shape == (T + 1, 2)

    mcmc = infer(x, num_samples=10, num_warmup=10, model=BoundedActor)

    assert mcmc.get_samples() is not None


def test_max_likelihood_recovers_parameters():
    """Check that MLE recovers parameters from synthetic observations."""
    true_params = dict(
        action_cost=0.01,
        action_variability=0.25,
        sigma_target=8.0,
        sigma_cursor=2.0,
    )
    model = BoundedActor(T=100, **true_params)
    x = model.simulate(random.PRNGKey(7), n=100)

    params = max_likelihood(
        key=random.PRNGKey(7),
        x=x,
        model=BoundedActor,
        max_steps=300,
        sigma_target=true_params["sigma_target"],
        sigma_cursor=true_params["sigma_cursor"],
    )

    assert jnp.isclose(params["action_cost"], true_params["action_cost"], atol=0.002)
    assert jnp.isclose(
        params["action_variability"], true_params["action_variability"], atol=0.02
    )


def test_parameter_recovery_runs():
    """Check that parameter recovery runs for a model with fixed parameters and custom bounds."""
    results = parameter_recovery(
        random.PRNGKey(0),
        PointMassSubjectiveActor,
        num_datasets=2,
        T=50,
        num_trials=2,
        max_steps=5,
        num_restarts=1,
        progress=False,
        constraints={"action_cost": LogBox(low=1e-10, high=1.0)},
        dt=1 / 60.0,
        tau=0.066,
        force_cost=0.0,
    )

    assert set(results["true"]) == set(results["estimate"])
    assert "tau" not in results["true"] and "force_cost" not in results["true"]
    for values in [*results["true"].values(), *results["estimate"].values()]:
        assert values.shape == (2,)
    assert results["ll_true"].shape == results["ll_estimate"].shape == (2,)
