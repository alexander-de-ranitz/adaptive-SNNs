import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.base import NoiseModelABC

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class OUP(NoiseModelABC):
    tau: float | Array = 1.0
    noise_std: float | Array = 1.0
    mean: float | Array = 0.0
    dim: int = 1

    @property
    def initial(self):
        return jnp.ones((self.dim,)) * self.mean

    def drift(self, t, x, args):
        return -1.0 / self.tau * (x - self.mean)

    def diffusion(self, t, x, args):
        # the value given in args takes precedence as this is used by the NoisyNetwork model
        # to set state-dependent noise stds. The default is a constant noise std used otherwise.
        noise_std = (
            self.noise_std if args is None else args.get("noise_std", self.noise_std)
        )

        # Return diagonal diffusion matrix
        if isinstance(noise_std, Array) and noise_std.ndim == 1:
            return jnp.diag(noise_std) * jnp.sqrt(2.0 / self.tau)
        return jnp.eye(self.dim) * noise_std * jnp.sqrt(2.0 / self.tau)

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
