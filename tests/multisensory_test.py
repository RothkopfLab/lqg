import jax.numpy as jnp
import pytest
from jax import random
from lqg.tracking.multisensory import (
    MultisensoryBoundedActor,
    MultisensoryPointMassBoundedActor,
    multisensory_delay_system,
)
from lqg import System
from lqg import xcorr


def test_delayed_multisensory_system():
    """Test that the delayed multisensory system can be created and simulated."""
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B = jnp.array([[0.0], [1.0]])
    V = jnp.eye(2) * 0.1
    Fs = [jnp.array([[1.0, 0.0]]), jnp.array([[0.0, 1.0]])]
    Ws = [jnp.eye(1) * 0.1, jnp.eye(1) * 0.1]
    Q = jnp.eye(2)
    R = jnp.eye(1)
    delays = [0, 2]

    system_spec = multisensory_delay_system(A, B, V, Fs, Ws, Q, R, delays=delays, T=500)

    assert system_spec is not None

    system = System(system_spec, system_spec)

    x = system.simulate(random.PRNGKey(42), n=10)

    # check that the shape of the state vector including delays is correct
    assert x.shape == (10, 501, (max(delays) + 1) * A.shape[-1])



@pytest.mark.parametrize(
    "actor_class", [MultisensoryBoundedActor, MultisensoryPointMassBoundedActor]
)
def test_multisensory_model(actor_class):

    # get CCG peak for different delays
    peaks = []
    for delays in [[0, 12], [12, 12]]:
        model = actor_class(delays=delays, sigmas=[10.0, 10.0])
        x = model.simulate(rng_key=random.PRNGKey(0), n=20)

        vels = jnp.diff(x, axis=-2)
        lags, correls = xcorr(vels[..., 1], vels[..., 0])
        peaks.append(correls.mean(axis=0).argmax())

    # check that the system in which both modalities are delayed 
    # has a later peak than the one in which only one is delayed
    assert peaks[0] < peaks[1]
