from jax import random, numpy as jnp

from lqg.utils import quadratic_form, bilinear_form


def test_quadratic_form():
    key1, key2 = random.split(random.PRNGKey(0))
    C = random.normal(key1, shape=(5, 10, 3))
    S = random.normal(key2, shape=(5, 5))

    # compute the quadratic form manually with transposes etc.
    H = (C.transpose((1, 2, 0)) @ S[None] @ C.transpose((1, 0, 2))).sum(axis=0)

    assert jnp.allclose(quadratic_form(C, S).sum(axis=0), H)


def test_bilinear_form():
    key1, key2, key3 = random.split(random.PRNGKey(1), 3)
    C = random.normal(key1, shape=(5, 10, 3))
    c = random.normal(key2, shape=(5, 10))
    S = random.normal(key3, shape=(5, 5))

    # compute the bilinear form manually with transposes etc.
    g = (C.transpose((1, 2, 0)) @ S[None] @ c.T[..., None]).sum(axis=(0, 2))

    assert jnp.allclose(bilinear_form(C, S, c).sum(axis=0), g)