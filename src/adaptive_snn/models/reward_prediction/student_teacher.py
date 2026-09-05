import diffrax as dfx
import jax.numpy as jnp

from adaptive_snn.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_snn.utils.operators import DefaultIfNone, ElementWiseMul


class StudentRewardModel(AbstractRewardPredictor):
    N_neurons: int
    N_students: int

    @property
    def initial(self):
        return RewardPrediction(value=jnp.zeros((1,)))

    @property
    def noise_shape(self):
        return RewardPrediction(value=None)

    def pre_step_update(
        self, t, x, args, reward, network_state, input_spikes, env_state
    ):
        mean_noiseless_student_output = jnp.mean(
            env_state.at[1 + self.N_students :].get()
        )  # Get the mean state of the reference neurons as the expected reward signal
        teacher_signal = env_state.at[
            0
        ].get()  # Get the first neuron's state as the teacher signal
        reward = -jnp.square(teacher_signal - mean_noiseless_student_output)
        return RewardPrediction(
            value=jnp.asarray([reward])
        )  # Return the computed reward

    def diffusion(self, t, x, args):
        return RewardPrediction(
            value=DefaultIfNone(
                default=jnp.zeros_like(x.value),
                else_do=ElementWiseMul(jnp.zeros_like(x.value)),
            )
        )

    def drift(self, t, x, args, reward, RPE):
        return RewardPrediction(value=jnp.zeros_like(x.value))

    def update(self, t, x, args):
        return x

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class MWERewardModel(AbstractRewardPredictor):
    N_neurons: int
    N_students: int

    @property
    def initial(self):
        return RewardPrediction(value=jnp.zeros((1,)))

    @property
    def noise_shape(self):
        return RewardPrediction(value=None)

    def pre_step_update(
        self, t, x, args, reward, network_state, input_spikes, env_state
    ):
        mean_noiseless_student_output = jnp.mean(
            env_state.at[1 + self.N_students :].get()
        )  # Get the mean state of the reference neurons as the expected reward signal
        teacher_signal = env_state.at[
            0
        ].get()  # Get the first neuron's state as the teacher signal
        reward = -jnp.square(teacher_signal - mean_noiseless_student_output)
        return RewardPrediction(
            value=jnp.asarray([reward])
        )  # Return the computed reward

    def diffusion(self, t, x, args):
        return RewardPrediction(
            value=DefaultIfNone(
                default=jnp.zeros_like(x.value),
                else_do=ElementWiseMul(jnp.zeros_like(x.value)),
            )
        )

    def drift(self, t, x, args, reward, RPE):
        return RewardPrediction(value=jnp.zeros_like(x.value))

    def update(self, t, x, args):
        return x

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
