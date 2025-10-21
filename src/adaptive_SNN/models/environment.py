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
    """Environment model that tracks spike rates using a circular buffer."""

    dim: int = 1  # Dimension of the environment process
    buffer_size: int = 1000  # Size of the spike rate buffer
    dt: float  # Time step for spike rate calculation

    def __init__(self, dt, dim: int = 1, buffer_size: int = 1000):
        self.dt = dt
        self.dim = dim
        self.buffer_size = jnp.int32(buffer_size)

    @property
    def initial(self):
        return (
            jnp.zeros(
                self.dim,
            ),
            jnp.zeros((self.dim, self.buffer_size)),
            jnp.zeros(
                1,
            ),
        )

    @property
    def noise_shape(self):
        return (
            jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float),
            jax.ShapeDtypeStruct(
                shape=(self.dim, self.buffer_size), dtype=default_float
            ),
            jax.ShapeDtypeStruct(shape=(1,), dtype=default_float),
        )

    def drift(self, t, x, args):
        return jax.tree.map(lambda a: jnp.zeros_like(a), x)

    def diffusion(self, t, x, args):
        return jax.tree.map(lambda a: ElementWiseMul(jnp.zeros_like(a)), x)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        _, buffer, buffer_index = x
        if args is None or "env_input" not in args:
            raise ValueError(
                "SpikeRateEnvironment requires 'env_input' in args for update."
            )
        spikes = args["env_input"]

        # Update buffer with new spikes
        new_buffer = buffer.at[:, buffer_index.astype(jnp.int32)].set(spikes)
        new_buffer_index = (buffer_index + 1) % self.buffer_size

        # Compute spike rate
        spike_rate = jnp.sum(new_buffer, axis=1) / (self.buffer_size * self.dt)
        return (spike_rate, new_buffer, new_buffer_index)
