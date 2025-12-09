import diffrax as dfx
import jax
import jax.numpy as jnp

from adaptive_SNN.models.base import EnvironmentABC

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class DoubleIntegrator(EnvironmentABC):
    """Environment model representing a double integrator system."""

    dim: int = 2  # State is 2-dimensional: [position, velocity]
    rate: float = 1.0  # Rate at which the environment responds to input

    def __init__(self, rate: float = 1.0):
        self.rate = rate

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float)

    def drift(self, t, x, args):
        if args is None or "get_env_input" not in args:
            raise ValueError(
                "DoubleIntegrator requires 'get_env_input' in args for drift computation."
            )

        dxdt = jnp.array(
            [x.at[1].get(), 0.0]
        )  # dx/dt = velocity, dv/dt = 0 (no acceleration)
        return dxdt

    def diffusion(self, t, x, args):
        # TODO: implement noise
        return jnp.zeros((self.dim, self.dim))

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        acceleration = self.rate * args["get_env_input"](t, x, args)
        return x.at[1].add(acceleration)  # Update velocity with acceleration
