from abc import ABC, abstractmethod

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class RewardPrediction(eqx.Module):
    reward: Array  # Scalar reward prediction


class AbstractRewardModel(ABC, eqx.Module):
    @property
    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def drift(self, t, x, args):
        pass

    @abstractmethod
    def diffusion(self, t, x, args):
        pass

    @property
    @abstractmethod
    def noise_shape(self):
        pass

    @abstractmethod
    def terms(self, key):
        pass

    @abstractmethod
    def update(self, t, x, args):
        """Apply non-differential updates"""
        pass


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
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x.reward),
            else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
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


class StudentRewardModel(AbstractRewardModel):
    N_neurons: int
    N_students: int

    def __init__(self, N_neurons: int, N_students: int):
        self.N_neurons = N_neurons
        self.N_students = N_students

    @property
    def initial(self):
        return RewardPrediction(reward=jnp.zeros((1,)))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x.reward),
            else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
        )

    def drift(self, t, x, args):
        return RewardPrediction(reward=jnp.zeros_like(x.reward))

    def update(self, t, x, args):
        mean_noiseless_student_output = jnp.mean(
            args["env_state"].at[1 + self.N_students :].get()
        )  # Get the mean state of the reference neurons as the expected reward signal
        teacher_signal = (
            args["env_state"].at[0].get()
        )  # Get the first neuron's state as the teacher signal
        reward = -jnp.square(teacher_signal - mean_noiseless_student_output)
        return RewardPrediction(
            reward=jnp.asarray([reward])
        )  # Return the computed reward

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class MWERewardModel(AbstractRewardModel):
    @property
    def initial(self):
        return RewardPrediction(reward=jnp.zeros((1,)))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x.reward),
            else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
        )

    def drift(self, t, x, args):
        return RewardPrediction(reward=jnp.zeros_like(x.reward))

    def update(self, t, x, args):
        mean_noiseless_student_output = jnp.mean(
            args["env_state"].at[1 + self.N_students :].get()
        )  # Get the mean state of the reference neurons as the expected reward signal
        teacher_signal = (
            args["env_state"].at[0].get()
        )  # Get the first neuron's state as the teacher signal
        reward = -jnp.square(teacher_signal - mean_noiseless_student_output)
        return RewardPrediction(
            reward=jnp.asarray([reward])
        )  # Return the computed reward

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class RewardPredictionRLS(RewardPrediction):
    reward: Array  # Scalar reward prediction
    weights: Array  # (n, ) Weight matrix for the RLS predictor
    P: Array  # (n, n) Inverse covariance matrix for the RLS predictor


class RLSRewardPrediction(AbstractRewardModel):
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
        return None

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
        return MixedPyTreeOperator(tree)

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
