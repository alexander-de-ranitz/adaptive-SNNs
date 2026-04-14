import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from adaptive_SNN.models.noise.base import NoiseModelABC
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class PoissonJumpProcess(NoiseModelABC):
    jump_rate: float | Array
    jump_mean: float | Array
    jump_std: float | Array
    initial_value: float | Array
    tau: float
    dim: int
    dt: float
    key: Array

    def __init__(
        self,
        jump_rate: float | Array = 0.0,
        jump_mean: float | Array = 0.0,
        jump_std: float | Array = 0.0,
        initial_value: float | Array = 0.0,
        tau: float = 0.1,
        dim: int = 1,
        dt: float = 1e-4,
        key: Array = jr.PRNGKey(0),
    ):
        self.jump_rate = jump_rate
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.initial_value = initial_value
        self.tau = tau
        self.dim = dim
        self.dt = dt
        self.key = key

    @property
    def initial(self):
        return jnp.ones((self.dim,)) * self.initial_value

    def drift(self, t, x, args):
        return -x / self.tau

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def _to_vector(self, value, dtype):
        return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), (self.dim,))

    def update(self, t, x, args):
        dt = self.dt if args is None else args.get("dt", self.dt)

        jump_rate = (
            self.jump_rate if args is None else args.get("jump_rate", self.jump_rate)
        )
        jump_mean = (
            self.jump_mean if args is None else args.get("jump_mean", self.jump_mean)
        )
        jump_std = (
            self.jump_std if args is None else args.get("jump_std", self.jump_std)
        )

        jump_rate = jnp.maximum(self._to_vector(jump_rate, x.dtype), 0.0)
        jump_mean = self._to_vector(jump_mean, x.dtype)
        jump_std = jnp.maximum(self._to_vector(jump_std, x.dtype), 0.0)

        step_idx = jnp.asarray(jnp.rint(t / dt))
        current_key = jr.fold_in(self.key, step_idx)
        jump_key, mark_key = jr.split(current_key)

        jump_count = jr.poisson(jump_key, jump_rate * dt, shape=(self.dim,))
        jump_count = jump_count.astype(x.dtype)

        gaussian_marks = jr.normal(mark_key, shape=(self.dim,), dtype=x.dtype)
        increments = (
            jump_count * jump_mean + jnp.sqrt(jump_count) * jump_std * gaussian_marks
        )
        return x + increments

    @property
    def noise_shape(self):
        return None

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
