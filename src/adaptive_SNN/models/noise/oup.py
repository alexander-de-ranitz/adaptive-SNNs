import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class OUP(eqx.Module):
    tau: float | Array = 1.0
    noise_scale: float | Array = 1.0
    mean: float | Array = 0.0
    dim: int = 1

    @property
    def initial(self):
        return jnp.ones((self.dim,)) * self.mean

    def drift(self, t, x, args):
        return -1.0 / self.tau * (x - self.mean)

    def diffusion(self, t, x, args):
        return jnp.eye(x.shape[0]) * jnp.sqrt(self.noise_scale)

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
