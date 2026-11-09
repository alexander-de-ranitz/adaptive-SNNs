import diffrax as dfx
import jax.numpy as jnp

from adaptive_snn.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_snn.utils.operators import DefaultIfNone, ElementWiseMul


class NullRewardPredictor(AbstractRewardPredictor):
    """No reward prediction: the predicted value is constant at zero.

    Used when the RPE does not depend on a reward prediction (e.g. RPE = reward).
    """

    dim: int = 1  # Dimension of the (zero) reward prediction

    @property
    def initial(self):
        return RewardPrediction(value=jnp.zeros((self.dim,)))

    @property
    def noise_shape(self):
        return RewardPrediction(value=None)

    def drift(self, t, x: RewardPrediction, args, reward, RPE):
        return RewardPrediction(value=jnp.zeros_like(x.value))

    def diffusion(self, t, x: RewardPrediction, args):
        return RewardPrediction(
            value=DefaultIfNone(
                default=jnp.zeros_like(x.value),
                else_do=ElementWiseMul(jnp.zeros_like(x.value)),
            )
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
