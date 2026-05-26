import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class RewardPredictionRLS(RewardPrediction):
    reward: Array  # Scalar reward prediction
    weights: Array  # (n, ) Weight matrix for the RLS predictor
    P: Array  # (n, n) Inverse covariance matrix for the RLS predictor


class RLSRewardPrediction(AbstractRewardPredictor):
    """Reward prediction model that uses Recursive Least Squares (RLS) to predict the reward"""

    lambda_: float = 0.9999  # Forgetting factor for RLS
    input_dim: int = 1  # Dimension of the input features for reward prediction
    P_init: float = 100.0  # Initial value for the inverse covariance matrix P

    @property
    def initial(self):
        return RewardPredictionRLS(
            reward=jnp.zeros((1,)),
            weights=jnp.zeros((self.input_dim,)),
            P=jnp.eye(self.input_dim) * self.P_init,
        )

    @property
    def noise_shape(self):
        return RewardPredictionRLS(reward=None, weights=None, P=None)

    def drift(self, t, x: RewardPredictionRLS, args: dict) -> RewardPredictionRLS:
        """No drift in the reward prediction process."""
        return RewardPredictionRLS(
            reward=jnp.zeros_like(x.reward),
            weights=jnp.zeros_like(x.weights),
            P=jnp.zeros_like(x.P),
        )

    def diffusion(self, t, x: RewardPredictionRLS, args: dict):
        """No diffusion in the reward prediction process."""
        tree = jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr),
                else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=default_float)),
            ),
            x,
        )
        return tree

    def update(self, t, x: RewardPredictionRLS, args: dict) -> RewardPredictionRLS:
        """Update the RLS weights based on the RPE."""
        if args is None or "RPE" not in args or "features" not in args:
            raise ValueError(
                "RLSRewardPrediction requires 'RPE' and 'features' in args for update."
            )
        RPE = args["RPE"]
        features = args["features"]
        weights = x.weights

        # Update weights using RLS update rule
        gain = x.P @ features / (self.lambda_ + features.T @ x.P @ features)
        new_weights = weights + gain * RPE
        new_P = (x.P - gain[:, None] @ features[None, :] @ x.P) / self.lambda_

        # Symmetrize new_P to ensure it remains positive definite, which can help with numerical stability
        new_P = (new_P + new_P.T) / 2

        return RewardPredictionRLS(reward=x.reward, weights=new_weights, P=new_P)

    def compute_next_reward_prediction(
        self, t, x: RewardPredictionRLS, args: dict
    ) -> RewardPredictionRLS:
        """Compute the reward prediction based on the current weights and input features."""
        features = args["features"]
        weights = x.weights
        return RewardPredictionRLS(reward=features @ weights, weights=weights, P=x.P)
