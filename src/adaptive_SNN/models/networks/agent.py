import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.base import AbstractLIFNetwork, LIFState
from adaptive_SNN.models.reward_prediction import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class AgentState(eqx.Module):
    network_state: LIFState
    reward_predictor_state: RewardPrediction
    RPE: Array  # Not part of the state that evolves according to the SDE, but we include it here for convenience in accessing/storing it


class Agent(eqx.Module):
    network: AbstractLIFNetwork
    reward_prediction_model: AbstractRewardPredictor

    def __init__(
        self,
        neuron_model: AbstractLIFNetwork,
        reward_prediction_model: AbstractRewardPredictor,
    ):
        self.network = neuron_model
        self.reward_prediction_model = reward_prediction_model

    @property
    def initial(self):
        return AgentState(
            self.network.initial,
            self.reward_prediction_model.initial,
            jnp.zeros(1),  # RPE initial state
        )

    def pre_step_update(self, t, x: AgentState, args, reward: Array = jnp.zeros(1)):
        """Perform any necessary updates to the state before computing the drift/diffusion.

        This is where we compute the reward prediction error (RPE) based on the current state of the reward predictor and the reward signal from the environment, and store it in the AgentState for use in the drift computation.
        """

        # Apply pre-step updates (currently no-op)
        network_state = self.network.pre_step_update(t, x.network_state, args)

        # Compute reward-prediction features and update reward predictor state
        reward_predictor_state = self.reward_prediction_model.pre_step_update(
            t, x.reward_predictor_state, args, reward, network_state
        )
        predicted_reward = reward_predictor_state.value
        RPE = reward - predicted_reward

        return AgentState(
            network_state=network_state,
            reward_predictor_state=reward_predictor_state,
            RPE=RPE,
        )

    def drift(self, t, x: AgentState, args, reward: Array):
        """Compute deterministic time derivatives for LearningModel state.

        Args:
            t: time
            x: (network_state, predicted_reward, RPE)
            args: dict
            reward: reward signal from environment at time t

        Returns:
            (d_network_state, d_predicted_reward)
        """

        (network_state, predicted_reward, RPE) = (
            x.network_state,
            x.reward_predictor_state,
            x.RPE,
        )
        neuron_drift = self.network.drift(t, network_state, args, RPE)
        reward_predictor_drift = self.reward_prediction_model.drift(
            t, predicted_reward, args, reward, network_state
        )

        return AgentState(neuron_drift, reward_predictor_drift, jnp.zeros_like(RPE))

    def diffusion(self, t, x: AgentState, args):
        neuron_diffusion = self.network.diffusion(t, x.network_state, args)
        reward_predictor_diffusion = self.reward_prediction_model.diffusion(
            t, x.reward_predictor_state, args
        )
        RPE_diffusion = DefaultIfNone(
            default=jnp.zeros_like(x.RPE), else_do=ElementWiseMul(jnp.zeros_like(x.RPE))
        )
        return MixedPyTreeOperator(
            AgentState(
                neuron_diffusion,
                reward_predictor_diffusion,
                RPE_diffusion,
            )
        )

    @property
    def noise_shape(self):
        return AgentState(
            self.network.noise_shape, self.reward_prediction_model.noise_shape, None
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: AgentState, args, input_spikes: Array | None = None):
        # Update components
        new_network_state = self.network.update(
            t, x.network_state, args, input_spikes=input_spikes
        )
        new_reward_predictor_state = self.reward_prediction_model.update(
            t, x.reward_predictor_state, args
        )  # reward and network_state are not needed for the current reward predictor update, but we include them here for future extensibility

        return AgentState(new_network_state, new_reward_predictor_state, x.RPE)

    def reset(self, t, x: AgentState, args):
        """Reset the agent by resetting the network (keeping the same weights).

        Note that the reward predictor state is not reset, as we want to maintain the learned reward predictions.
        """
        new_network_state = self.network.reset(t, x.network_state, args)
        return AgentState(
            new_network_state, x.reward_predictor_state, jnp.zeros_like(x.RPE)
        )
