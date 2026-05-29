import diffrax as dfx
import jax
import jax.numpy as jnp

from adaptive_SNN.models.environments.base import AbstractEnvironment
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class InputTrackingEnvironment(AbstractEnvironment):
    """Environment model that tracks an input signal."""

    dim: int = 1  # Dimension of the environment process
    rate: float = 1.0  # Rate at which the environment responds to input

    def __init__(self, rate: float = 1.0, dim: int = 1):
        self.dim = dim
        self.rate = rate

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return None

    def drift(self, t, x, args, env_input=None):
        input = jnp.where(env_input is not None, env_input, jnp.zeros_like(x))
        return self.rate * (input - x)  # Simple dynamics towards input

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args, env_input=None):
        return x
