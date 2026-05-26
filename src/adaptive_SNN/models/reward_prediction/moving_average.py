import diffrax as dfx
import jax.numpy as jnp

from adaptive_SNN.models.reward_prediction.base import (
    AbstractRewardModel,
    RewardPrediction,
)
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class MovingAverageRewardModel(AbstractRewardModel):
    reward_rate: float = 1  # Rate at which the reward is integrated
    dim: int = 1  # Dimension of the reward process

    def __init__(self, reward_rate: float = 1, dim: int = 1):
        self.reward_rate = reward_rate
        self.dim = dim

    @property
    def initial(self):
        return RewardPrediction(reward=jnp.zeros((self.dim,)))

    @property
    def noise_shape(self):
        return RewardPrediction(reward=None)

    def diffusion(self, t, x, args):
        return RewardPrediction(
            reward=DefaultIfNone(
                default=jnp.zeros_like(x.reward),
                else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
            )
        )

    def drift(self, t, x, args):
        if args is None or "reward" not in args:
            raise ValueError(
                "RewardModel requires 'reward' in args for drift computation."
            )
        return RewardPrediction(reward=self.reward_rate * (args["reward"] - x.reward))

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        """No non-differential updates needed for this reward model."""
        return x
