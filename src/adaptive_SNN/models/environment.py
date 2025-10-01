import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


# TODO: Make evironment ABC and have different environment models inherit from it


class EnvironmentModel(eqx.Module):
    dim: int = 1  # Dimension of the environment process
    target_state: Array  # Desired target state for the environment
    rate: float = 100.0  # Rate at which the environment responds to input

    def __init__(self, target_state: Array = None, dim: int = 1):
        self.dim = dim
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
        return self.rate * (
            args["env_input"](t, x, args) - x
        )  # Simple dynamics towards input

    def diffusion(self, t, x, args):
        return jnp.zeros((self.dim, self.dim))

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
