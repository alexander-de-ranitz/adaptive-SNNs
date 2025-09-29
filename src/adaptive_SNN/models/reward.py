import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp


class RewardModel(eqx.Module):
    reward_rate: float = 0.1  # Rate at which the reward is integrated
    dim: int = 1  # Dimension of the reward process

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return None  # No noise in reward process

    def diffusion(self, t, x, args):
        return None

    def drift(self, t, x, args):
        if args is None or "reward" not in args:
            raise ValueError(
                "RewardModel requires 'reward' in args for drift computation."
            )
        return self.reward_rate * (args["reward"](t, x, args) - x)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
