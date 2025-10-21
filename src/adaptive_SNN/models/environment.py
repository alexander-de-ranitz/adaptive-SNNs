import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.base import EnvironmentABC
from adaptive_SNN.utils import ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class InputTrackingEnvironment(EnvironmentABC):
    dim: int = 1  # Dimension of the environment process
    target_state: Array  # Desired target state for the environment
    rate: float = 1.0  # Rate at which the environment responds to input

    def __init__(self, rate: float = 1.0, target_state: Array = None, dim: int = 1):
        self.dim = dim
        self.rate = rate
        if target_state is None:
            target_state = jnp.zeros((dim,))
        self.target_state = target_state
        assert self.target_state.shape == (dim,), (
            "target_state must match the dimension of the environment"
        )

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float)

    def drift(self, t, x, args):
        if args is None or "env_input" not in args:
            raise ValueError(
                "EnvironmentModel requires 'env_input' in args for drift computation."
            )
        return self.rate * (args["env_input"] - x)  # Simple dynamics towards input

    def diffusion(self, t, x, args):
        return jnp.zeros((self.dim, self.dim))

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        return x


class SpikeRateEnvironment(EnvironmentABC):
    """Environment model that tracks the spike rate of a neuron."""

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
        if args is None or "env_input" not in args:
            raise ValueError(
                "SpikeRateEnvironment requires 'env_input' in args for update."
            )
        return x + args["env_input"]
