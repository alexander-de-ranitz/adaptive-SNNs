import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class RewardModel(eqx.Module):
    reward_rate: float = 100  # Rate at which the reward is integrated
    dim: int = 1  # Dimension of the reward process

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float)

    def diffusion(self, t, x, args):
        return jnp.zeros((self.dim, self.dim))  # Zero diffusion for reward

    def drift(self, t, x, args):
        if args is None or "reward" not in args:
            raise ValueError(
                "RewardModel requires 'reward' in args for drift computation."
            )
        return self.reward_rate * (args["reward"] - x)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
