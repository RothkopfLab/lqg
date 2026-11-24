import jax
import pytest
import jax.numpy as jnp
from jax import random

from lqg.system import LQG

from lqg.tracking import (
    BoundedActor,
    SubjectiveActor,
    PointMassBoundedActor,
    PointMassSubjectiveActor,
    OptimalActor,
    RelativeObservationBoundedActor,
)


def test_lqg_simulate():
    """Test that LQG simulation runs."""
    dt = 1.0 / 60.0
    T = 1000

    # parameters
    action_variability = 0.5
    sigma = 6.0
    sigma_prop = 3.0
    action_cost = 0.5

    A = jnp.eye(2)
    B = jnp.array([[0.0], [dt]])
    V = jnp.diag(jnp.array([1.0, action_variability]))

    F = jnp.eye(2)
    W = jnp.diag(jnp.array([sigma, sigma_prop]))

    Q = jnp.array([[1.0, -1.0], [-1.0, 1]])

    R = jnp.eye(1) * action_cost

    lqg = LQG(A=A, B=B, F=F, V=V, W=W, Q=Q, R=R)

    x = lqg.simulate(random.PRNGKey(0), x0=jnp.zeros(2), n=10)

    # simply check that it ran through
    assert x.shape == (10, T + 1, 2)


@pytest.mark.parametrize(
    ("model_class"),
    [
        BoundedActor,
        SubjectiveActor,
        PointMassBoundedActor,
        PointMassSubjectiveActor,
        OptimalActor,
        RelativeObservationBoundedActor,
    ],
)
def test_model(model_class):
    """Test that simulation works for all models."""

    T = 500
    lqg = model_class(T=T)

    x = lqg.simulate(random.PRNGKey(0), x0=jnp.zeros(lqg.xdim), n=10)

    # simply check that it ran through
    assert x.shape == (10, T + 1, lqg.xdim)
    assert not jnp.isnan(x).any()


def test_simulate_subjective():
    """Test that subjective model without subjective component is equal to non-subjective model."""
    bounded_actor = BoundedActor(
        process_noise=1.0,
        sigma_target=6.0,
        action_cost=0.1,
        action_variability=0.5,
        sigma_cursor=3.0,
        T=500,
    )
    subjective_actor = SubjectiveActor(
        process_noise=1.0,
        sigma_target=6.0,
        action_cost=0.1,
        action_variability=0.5,
        sigma_cursor=3.0,
        subj_noise=1.0,
        subj_vel_noise=0.0,
        T=500,
    )

    x_b = bounded_actor.simulate(rng_key=random.PRNGKey(0), n=20)
    x_s = subjective_actor.simulate(rng_key=random.PRNGKey(0), n=20)

    assert jnp.allclose(x_b, x_s)


@pytest.fixture
def x64():
    """Run a test in double precision and restore the previous setting afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_simulate_point_mass_subjective(x64):
    """Test that point-mass subjective model without subjective component is equal to point-mass bounded actor.

    The two models compute their gains with matrices of different sizes. In single precision,
    rounding errors accumulate over the simulated time steps and differ between platforms,
    so we compare in double precision.
    """
    params = dict(
        process_noise=1.0,
        sigma_target=6.0,
        action_cost=0.01,
        action_variability=0.5,
        sigma_cursor=3.0,
        T=500,
    )
    bounded_actor = PointMassBoundedActor(**params)
    subjective_actor = PointMassSubjectiveActor(
        **params, subj_noise=1.0, subj_vel_noise=0.0
    )

    x_b = bounded_actor.simulate(rng_key=random.PRNGKey(0), n=20)
    x_s = subjective_actor.simulate(rng_key=random.PRNGKey(0), n=20)

    assert x_b.dtype == jnp.float64
    assert jnp.allclose(x_b, x_s)


def test_belief_tracking_distribution():
    T = 500
    actor = BoundedActor(T=T)

    x = actor.simulate(rng_key=random.PRNGKey(0), n=20)

    assert actor.belief_tracking_distribution(x).shape() == (
        20,
        T,
        actor.actor.A.shape[1],
    )
