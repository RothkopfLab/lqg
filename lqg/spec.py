from typing import NamedTuple
import jax.numpy as jnp


class LQGSpec(NamedTuple):
    """ (generalized) LQG specification """

    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.array
    qf: jnp.array
    P: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray
    V: jnp.ndarray
    F: jnp.ndarray
    W: jnp.ndarray
