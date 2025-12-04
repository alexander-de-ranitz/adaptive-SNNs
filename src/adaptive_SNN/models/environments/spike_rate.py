import diffrax as dfx
import jax
import jax.numpy as jnp

from adaptive_SNN.models.base import EnvironmentABC
from adaptive_SNN.utils import ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class SpikeRateEnvironment(EnvironmentABC):
    """Environment model that tracks the spike rate of a neuron using an exponential filter.

    The input is expected to be the spike count at each time step. If tracking the combined rate of multiple neurons,
    the input should be the sum of their spike counts. When tracking the spike rate of multiple neurons separately,
    the input should be a vector indicating whether a spike occured for each neuron.
    """

    dim: int = 1  # Dimension of the environment process

    def __init__(self, dim: int = 1):
        self.dim = dim

    @property
    def initial(self):
        return jnp.zeros(
            self.dim,
        )

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float)

    def drift(self, t, x, args):
        return -x  # Exponential decay of spike rate

    def diffusion(self, t, x, args):
        return ElementWiseMul(jnp.zeros_like(x))

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        """Non-differential update to increment the state based on the input."""
        if args is None or "get_env_input" not in args:
            raise ValueError(
                "SpikeRateEnvironment requires 'get_env_input' in args for update."
            )
        return x + args["get_env_input"](t, x, args)  # Increment by spike count
