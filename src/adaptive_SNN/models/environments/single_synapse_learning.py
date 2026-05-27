import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.environments import (
    AbstractEnvironment,
    AbstractEnvironmentState,
)
from adaptive_SNN.models.noise import AbstractNoiseModel, PoissonJumpProcess
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class SingleSynapseLearningState(AbstractEnvironmentState):
    reward: Array
    reward_noise: Array


class SingleSynapseLearningEnv(AbstractEnvironment):
    reward_dim: int = 1  # Dimension of the environment process
    tau_reward: float = 1  # Time constant for reward decay
    reward_noise_process: AbstractNoiseModel = PoissonJumpProcess(
        jump_rate=0.0, jump_mean=0.0, jump_std=0.0
    )

    @property
    def initial(self):
        return SingleSynapseLearningState(
            reward=jnp.zeros(self.reward_dim),
            reward_noise=self.reward_noise_process.initial,
        )

    @property
    def noise_shape(self):
        return SingleSynapseLearningState(
            reward=None, reward_noise=self.reward_noise_process.noise_shape
        )

    def drift(self, t, x: SingleSynapseLearningState, args, env_input=None):
        return SingleSynapseLearningState(
            reward=-x.reward / self.tau_reward,  # Exponential decay of spike rate
            reward_noise=self.reward_noise_process.drift(t, x.reward_noise, args),
        )

    def diffusion(self, t, x: SingleSynapseLearningState, args):
        return SingleSynapseLearningState(
            reward=DefaultIfNone(
                default=jnp.zeros_like(x.reward),
                else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
            ),
            reward_noise=self.reward_noise_process.diffusion(t, x.reward_noise, args),
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: SingleSynapseLearningState, args, env_input=None):
        """Non-differential update to increment the state based on the input."""
        return SingleSynapseLearningState(
            reward=x.reward + env_input,
            reward_noise=self.reward_noise_process.update(t, x.reward_noise, args),
        )
