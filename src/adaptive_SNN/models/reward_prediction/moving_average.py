import diffrax as dfx
import jax.numpy as jnp

from adaptive_SNN.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class MovingAverageRewardPredictor(AbstractRewardPredictor):
    rate: float = 1  # Rate at which the reward is integrated
    dim: int = 1  # Dimension of the reward process

    def __init__(self, rate: float = 1, dim: int = 1):
        self.rate = rate
        self.dim = dim

    @property
    def initial(self):
        return RewardPrediction(value=jnp.zeros((self.dim,)))

    @property
    def noise_shape(self):
        return RewardPrediction(value=None)

    def diffusion(self, t, x: RewardPrediction, args):
        return RewardPrediction(
            value=DefaultIfNone(
                default=jnp.zeros_like(x.value),
                else_do=ElementWiseMul(jnp.zeros_like(x.value)),
            )
        )

    def drift(self, t, x: RewardPrediction, args, reward, network_state):
        return RewardPrediction(value=self.rate * (reward - x.value))

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: RewardPrediction, args):
        """No non-differential updates needed for this reward model."""
        return x
