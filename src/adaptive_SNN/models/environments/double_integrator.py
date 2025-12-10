import diffrax as dfx
import jax
import jax.numpy as jnp

from adaptive_SNN.models.base import EnvironmentABC
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

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
        return None

    def drift(self, t, x, args):
        if args is None or "get_env_input" not in args:
            raise ValueError(
                "DoubleIntegrator requires 'get_env_input' in args for drift computation."
            )

        acceleration = self.rate * args["get_env_input"](t, x, args)

        dxdt = jnp.array(
            [x.at[1].get(), acceleration]
        )  # dx/dt = velocity, dv/dt = input
        return dxdt

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

    def update(self, t, x, args):
        return x


class DoubleIntegratorKickControl(EnvironmentABC):
    """Double integrator environment controlled using discrete kicks."""

    dim: int = 2  # State is 2-dimensional: [position, velocity]
    rate: float = 1.0  # Rate at which the environment responds to input

    def __init__(self, rate: float = 1.0):
        self.rate = rate

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return None

    def drift(self, t, x, args):
        # Position is updated as usual, velocity is not changed here, but in update
        dxdt = jnp.array([x.at[1].get(), 0.0])
        return dxdt

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

    def update(self, t, x, args):
        """Discrete update to apply velocity kick based on input."""
        if args is None or "get_env_input" not in args:
            raise ValueError(
                "DoubleIntegratorKickControl requires 'get_env_input' in args for update."
            )
        acceleration_kick = self.rate * args["get_env_input"](t, x, args)
        x = x.at[1].set(x.at[1].get() + acceleration_kick)  # Update velocity with kick
        return x
