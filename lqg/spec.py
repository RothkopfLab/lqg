from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp


@register_pytree_node_class
class LQGSpec:
    """ LQG specification """

    def __init__(self,
                 Q: jnp.ndarray,
                 R: jnp.ndarray,
                 A: jnp.ndarray,
                 B: jnp.ndarray,
                 V: jnp.ndarray,
                 F: jnp.ndarray,
                 W: jnp.ndarray,
                 q: jnp.ndarray = None,
                 Qf: jnp.array = None,
                 qf: jnp.array = None,
                 P: jnp.ndarray = None,
                 r: jnp.ndarray = None,
                 Cx: jnp.ndarray = None,
                 Cu: jnp.ndarray = None,
                 D: jnp.ndarray = None
                 ):

        T = Q.shape[0]
        state_dim = Q.shape[1]
        action_dim = R.shape[1]
        obs_dim = W.shape[1]

        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.V = V
        self.F = F
        self.W = W

        self.q = jnp.zeros((T, state_dim)) if q is None else q
        self.Qf = Q[-1] if Qf is None else Qf
        self.qf = self.q[-1] if qf is None else qf
        self.P = jnp.zeros((T, action_dim, state_dim)) if P is None else P
        self.r = jnp.zeros((T, action_dim)) if r is None else r
        self.Cx = jnp.zeros((T, state_dim, V.shape[-1], state_dim)) if Cx is None else Cx
        self.Cu = jnp.zeros((T, state_dim, V.shape[-1], action_dim)) if Cu is None else Cu
        self.D = jnp.zeros((T, obs_dim, W.shape[-1], state_dim)) if D is None else D

    def tree_flatten(self):
        children = (self.Q, self.R, self.A, self.B, self.V, self.W,
                    self.q, self.Qf, self.qf, self.P, self.r, self.Cx, self.Cu, self.D)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
