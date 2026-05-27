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
    value: Array  # Scalar reward prediction
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
            value=jnp.zeros((1,)),
            weights=jnp.zeros((self.input_dim,)),
            P=jnp.eye(self.input_dim) * self.P_init,
        )

    @property
    def noise_shape(self):
        return RewardPredictionRLS(value=None, weights=None, P=None)

    def pre_step_update(self, t, x: RewardPredictionRLS, args, reward, network_state):
        features = args["feature_fn"](t, network_state, args)
        weights = x.weights
        predicted_reward = weights @ features

        # Update weights using RLS update rule
        error = reward - predicted_reward
        gain = x.P @ features / (self.lambda_ + features.T @ x.P @ features)
        new_weights = weights + gain * error
        new_P = (x.P - gain[:, None] @ features[None, :] @ x.P) / self.lambda_

        # Symmetrize new_P to ensure it remains positive definite, which can help with numerical stability
        new_P = (new_P + new_P.T) / 2

        return RewardPredictionRLS(value=predicted_reward, weights=new_weights, P=new_P)

    def drift(
        self, t, x: RewardPredictionRLS, args: dict, reward: Array, network_state: Array
    ) -> RewardPredictionRLS:
        """No drift in the reward prediction process."""
        return RewardPredictionRLS(
            value=jnp.zeros_like(x.value),
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
        return x
