from typing import NamedTuple
import jax.numpy as jnp


class LQGSpec(NamedTuple):
    """Time-varying specification of a (generalized) LQG problem.

    Dynamics and observations:

    ```
    x[t+1] = A[t] x[t] + B[t] u[t] + V[t] eps[t],    eps[t] ~ N(0, I)
    y[t+1] = F[t] x[t+1] + W[t] eta[t],       eta[t] ~ N(0, I)
    ```

    Cost, up to a constant:

    ```
    sum_t ( 1/2 x[t]^T Q[t] x[t] + q[t]^T x[t]
          + u[t]^T P[t] x[t]
          + 1/2 u[t]^T R[t] u[t] + r[t]^T u[t] )
    + 1/2 x[T]^T Qf x[T] + qf^T x[T]
    ```

    All time-varying fields have a leading time axis of length ``T``. Use
    [Actor][lqg.system.Actor] or [Dynamics][lqg.system.Dynamics] to build a
    specification from time-invariant matrices.

    Attributes:
        Q (jnp.ndarray): quadratic state cost, shape (T, xdim, xdim)
        q (jnp.ndarray): linear state cost, shape (T, xdim)
        Qf (jnp.ndarray): quadratic final state cost, shape (xdim, xdim)
        qf (jnp.ndarray): linear final state cost, shape (xdim,)
        P (jnp.ndarray): action-state cross cost, shape (T, udim, xdim)
        R (jnp.ndarray): quadratic action cost, shape (T, udim, udim)
        r (jnp.ndarray): linear action cost, shape (T, udim)
        A (jnp.ndarray): state transition matrices, shape (T, xdim, xdim)
        B (jnp.ndarray): control matrices, shape (T, xdim, udim)
        V (jnp.ndarray): process noise Cholesky factors, shape (T, xdim, xdim)
        F (jnp.ndarray): observation matrices, shape (T, ydim, xdim)
        W (jnp.ndarray): observation noise Cholesky factors, shape
            (T, ydim, ydim)
    """

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
