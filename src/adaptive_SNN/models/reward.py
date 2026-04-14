from abc import ABC, abstractmethod

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


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
        return jnp.zeros((self.dim,))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

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
        return jnp.zeros((1,))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def drift(self, t, x, args):
        return jnp.zeros_like(x)

    def update(self, t, x, args):
        mean_noiseless_student_output = jnp.mean(
            args["env_state"].at[1 + self.N_students :].get()
        )  # Get the mean state of the reference neurons as the expected reward signal
        teacher_signal = (
            args["env_state"].at[0].get()
        )  # Get the first neuron's state as the teacher signal
        reward = -jnp.square(teacher_signal - mean_noiseless_student_output)
        return jnp.asarray([reward])  # Return the computed reward

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
        return jnp.zeros((1,))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def drift(self, t, x, args):
        return jnp.zeros_like(x)

    def update(self, t, x, args):
        mean_noiseless_student_output = jnp.mean(
            args["env_state"].at[1 + self.N_students :].get()
        )  # Get the mean state of the reference neurons as the expected reward signal
        teacher_signal = (
            args["env_state"].at[0].get()
        )  # Get the first neuron's state as the teacher signal
        reward = -jnp.square(teacher_signal - mean_noiseless_student_output)
        return jnp.asarray([reward])  # Return the computed reward

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
