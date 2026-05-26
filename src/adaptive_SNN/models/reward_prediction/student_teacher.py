import diffrax as dfx
import jax.numpy as jnp

from adaptive_SNN.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class StudentRewardModel(AbstractRewardPredictor):
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
        return RewardPrediction(reward=None)

    def diffusion(self, t, x, args):
        return RewardPrediction(
            reward=DefaultIfNone(
                default=jnp.zeros_like(x.reward),
                else_do=ElementWiseMul(jnp.zeros_like(x.reward)),
            )
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


class MWERewardModel(AbstractRewardPredictor):
    @property
    def initial(self):
        return RewardPrediction(reward=jnp.zeros((1,)))

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
