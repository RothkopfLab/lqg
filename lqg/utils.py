from jax import vmap

# quadratic_form takes arrays A (n, d, m) and B (n, n) and computes A[:, i, :].T @ B @ A[:, i, :]
# for all i along A's second axis in a vectorized way and returns an array (d, m, m)
quadratic_form = vmap(lambda A, B: A.T @ B @ A, in_axes=(1, None))

# quadratic_form_t takes arrays A (m, d, n) and B (n, n) and computes A[:, i, :] @ B @ A[:, i, :].T
# for all i along A's second axis in a vectorized way and returns an array (d, m, m)
quadratic_form_t = vmap(lambda A, B: A @ B @ A.T, in_axes=(1, None))

# bilinear_form takes arrays A (n, d, m), B (n, n) and C (n, d, o)
# and computes A[:, i, :].T @ B @ C[:, i, :]
# for all i along A and C's second axis in a vectorized way and returns an array (d, m, o)
bilinear_form = vmap(lambda A, B, C: A.T @ B @ C, in_axes=(1, None, 1))

# bilinear_form_t takes arrays A (m, d, n), B (n, n) and C (o, d, n)
# and computes A[:, i, :] @ B @ C[:, i, :].T
# for all i along A and C's second axis in a vectorized way and returns an array (d, m, o)
bilinear_form_t = vmap(lambda A, B, C: A @ B @ C.T, in_axes=(1, None, 1))
