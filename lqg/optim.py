"""
A collection of helper functions for optimization with JAX.
"""

import numpy as onp
import scipy.optimize
from jax import grad, jit
from jax.flatten_util import ravel_pytree

from jax.config import config

config.update("jax_enable_x64", True)


def minimize(
    fun,
    x0,
    method=None,
    args=(),
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
):
    """A simple wrapper for `scipy.optimize.minimize` using JAX.

    The objective is differentiated automatically with JAX, and parameters can
    be arbitrary PyTrees. The remaining keyword arguments are passed through to
    [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html);
    see its documentation for details.

    Args:
        fun (callable): The objective function to be minimized, written in JAX
            code so that it is automatically differentiable. Its signature is
            `fun(x, *args) -> float`, where `x` is a PyTree and `args` is a
            tuple of the fixed parameters needed to completely specify the
            function.
        x0: Initial guess represented as a JAX PyTree.
        method (str or callable, optional): Type of solver, e.g. `"BFGS"`,
            `"L-BFGS-B"` or `"SLSQP"`. If not given, chosen to be one of
            `BFGS`, `L-BFGS-B`, `SLSQP`, depending on whether the problem has
            constraints or bounds.
        args (tuple, optional): Extra arguments passed to the objective function
            and its derivative. Must consist of valid JAX types; e.g. the leaves
            of the PyTree must be floats.
        bounds (sequence or Bounds, optional): Bounds on variables for
            L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods. Bounds must
            be flattened manually in the same order as the inputs `x0`.
        constraints (Constraint, dict or list thereof, optional): Constraints
            definition (only for COBYLA, SLSQP and trust-constr). Constraints
            must be flattened manually in the same order as the inputs `x0`.
        tol (float, optional): Tolerance for termination.
        callback (callable, optional): Called after each iteration as
            `callback(xk)`, where `xk` is the current parameter vector
            represented as a PyTree.
        options (dict, optional): A dictionary of solver options, e.g.
            `maxiter` or `disp`.

    Returns:
        scipy.optimize.OptimizeResult: The optimization result. Its `x`
            attribute holds the solution, represented as a JAX PyTree.
    """
    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)

    # Wrap the objective function to consume flat _original_
    # numpy arrays and produce scalar outputs.
    def fun_wrapper(x_flat, *args):
        x = unravel(x_flat)
        return float(fun(x, *args))

    # Wrap the gradient in a similar manner
    jac = jit(grad(fun))

    def jac_wrapper(x_flat, *args):
        x = unravel(x_flat)
        g_flat, _ = ravel_pytree(jac(x, *args))
        return onp.array(g_flat)

    # Wrap the callback to consume a pytree
    def callback_wrapper(x_flat, *args):
        if callback is not None:
            x = unravel(x_flat)
            return callback(x, *args)

    # Minimize with scipy
    results = scipy.optimize.minimize(
        fun_wrapper,
        x0_flat,
        args=args,
        method=method,
        jac=jac_wrapper,
        callback=callback_wrapper,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options=options,
    )

    # pack the output back into a PyTree
    results["x"] = unravel(results["x"])
    return results
