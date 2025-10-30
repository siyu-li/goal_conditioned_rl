import flax.linen as nn
import math
import jax


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


def default_init(scale: float = math.sqrt(2.0)):
    """Orthogonal initializer with a sane default scale.

    Note: Avoid using JAX ops (e.g., jax.numpy) in default arguments at import
    time to prevent triggering device compilation during module import.
    """
    return nn.initializers.orthogonal(scale)
