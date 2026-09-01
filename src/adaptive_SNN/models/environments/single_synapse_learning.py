import diffrax as dfx
import equinox as eqx
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


class BiphasicSingleSynapseLearningState(AbstractEnvironmentState):
    reward: Array
    reward_noise: Array
    biphasic_state: Array


class BiphasicSingleSynapseLearningEnv(AbstractEnvironment):
    reward_dim: int = 1  # Dimension of the environment process
    tau_biphasic: Array = eqx.field(default_factory=lambda: jnp.array([0.2, 1.0]))
    reward_noise_process: AbstractNoiseModel = PoissonJumpProcess(
        jump_rate=0.0, jump_mean=0.0, jump_std=0.0
    )

    @property
    def initial(self):
        return BiphasicSingleSynapseLearningState(
            reward=jnp.zeros(self.reward_dim),
            reward_noise=self.reward_noise_process.initial,
            biphasic_state=jnp.zeros_like(self.tau_biphasic),
        )

    @property
    def noise_shape(self):
        return BiphasicSingleSynapseLearningState(
            reward=None,
            reward_noise=self.reward_noise_process.noise_shape,
            biphasic_state=None,
        )

    def pre_step_update(self, t, x: BiphasicSingleSynapseLearningState, args):
        """Non-differential update to increment the state based on the input."""
        return BiphasicSingleSynapseLearningState(
            reward=jnp.atleast_1d(x.biphasic_state[0] - x.biphasic_state[1]),
            reward_noise=self.reward_noise_process.pre_step_update(
                t, x.reward_noise, args
            ),
            biphasic_state=x.biphasic_state,
        )

    def drift(self, t, x: BiphasicSingleSynapseLearningState, args, env_input=None):
        return BiphasicSingleSynapseLearningState(
            reward=jnp.zeros_like(x.reward),
            reward_noise=self.reward_noise_process.drift(t, x.reward_noise, args),
            biphasic_state=-x.biphasic_state / self.tau_biphasic,
        )

    def diffusion(self, t, x: BiphasicSingleSynapseLearningState, args):
        return BiphasicSingleSynapseLearningState(
            reward=DefaultIfNone(
                default=jnp.zeros_like(x.reward),
                else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
            ),
            reward_noise=self.reward_noise_process.diffusion(t, x.reward_noise, args),
            biphasic_state=DefaultIfNone(
                default=jnp.zeros_like(x.biphasic_state),
                else_do=ElementWiseMul(jnp.zeros_like(x.biphasic_state)),
            ),
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: BiphasicSingleSynapseLearningState, args, env_input=None):
        """Non-differential update to increment the state based on the input."""
        return BiphasicSingleSynapseLearningState(
            reward=x.reward,
            reward_noise=self.reward_noise_process.update(t, x.reward_noise, args),
            biphasic_state=x.biphasic_state + env_input * 1 / self.tau_biphasic,
        )


class BaselineSingleSynapseLearningState(AbstractEnvironmentState):
    reward: Array
    mean_reward: Array
    reward_noise: Array


class BaselineSingleSynapseLearningEnv(AbstractEnvironment):
    reward_dim: int = 1  # Dimension of the environment process
    tau_reward: float = 1  # Time constant for reward decay
    tau_mean_reward: float = 10  # Time constant for mean reward decay
    reward_noise_process: AbstractNoiseModel = PoissonJumpProcess(
        jump_rate=0.0, jump_mean=0.0, jump_std=0.0
    )

    @property
    def initial(self):
        return BaselineSingleSynapseLearningState(
            reward=jnp.zeros(self.reward_dim),
            mean_reward=jnp.zeros(self.reward_dim),
            reward_noise=self.reward_noise_process.initial,
        )

    @property
    def noise_shape(self):
        return BaselineSingleSynapseLearningState(
            reward=None,
            reward_noise=self.reward_noise_process.noise_shape,
            mean_reward=None,
        )

    def pre_step_update(self, t, x: BaselineSingleSynapseLearningState, args):
        """Non-differential update to increment the state based on the input."""
        return BaselineSingleSynapseLearningState(
            reward=x.reward,
            reward_noise=self.reward_noise_process.pre_step_update(
                t, x.reward_noise, args
            ),
            mean_reward=x.mean_reward,
        )

    def drift(self, t, x: BaselineSingleSynapseLearningState, args, env_input=None):
        return BaselineSingleSynapseLearningState(
            reward=-x.reward / self.tau_reward,
            reward_noise=self.reward_noise_process.drift(t, x.reward_noise, args),
            mean_reward=(x.reward - x.mean_reward) / self.tau_mean_reward,
        )

    def diffusion(self, t, x: BaselineSingleSynapseLearningState, args):
        return BaselineSingleSynapseLearningState(
            reward=DefaultIfNone(
                default=jnp.zeros_like(x.reward),
                else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
            ),
            reward_noise=self.reward_noise_process.diffusion(t, x.reward_noise, args),
            mean_reward=DefaultIfNone(
                default=jnp.zeros_like(x.mean_reward),
                else_do=ElementWiseMul(jnp.zeros_like(x.mean_reward)),
            ),
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: BaselineSingleSynapseLearningState, args, env_input=None):
        """Non-differential update to increment the state based on the input."""
        return BaselineSingleSynapseLearningState(
            reward=x.reward + env_input,
            reward_noise=self.reward_noise_process.update(t, x.reward_noise, args),
            mean_reward=x.mean_reward,
        )
