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

    def pre_step_update(
        self,
        t,
        x: AgentState,
        args,
        input_spikes: Array,
        env_state: Array,
        reward: Array = jnp.zeros(1),
        disable_RPE: Array = jnp.array(False),
    ):
        """Perform any necessary updates to the state before computing the drift/diffusion."""
        new_reward_predictor_state = self.reward_prediction_model.pre_step_update(
            t,
            x.reward_predictor_state,
            args,
            reward=reward,
            network_state=x.network_state,
            input_spikes=input_spikes,
            env_state=env_state,
        )
        network_state = self.network.pre_step_update(
            t, x.network_state, args, input_spikes=input_spikes
        )

        # RPE is set to zero during warmup if disable_RPE is True
        # otherwise, remains unchanges as computed in previous step's update
        new_RPE = jnp.where(disable_RPE, jnp.zeros_like(x.RPE), x.RPE)

        return AgentState(
            network_state=network_state,
            reward_predictor_state=new_reward_predictor_state,
            RPE=new_RPE,
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
        neuron_drift = self.network.drift(t, network_state, args, RPE=RPE)
        reward_predictor_drift = self.reward_prediction_model.drift(
            t, predicted_reward, args, reward=reward, RPE=RPE
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

    def update(self, t, x: AgentState, args, reward: Array):
        # Update components
        new_network_state = self.network.update(t, x.network_state, args)
        new_reward_predictor_state = self.reward_prediction_model.update(
            t, x.reward_predictor_state, args
        )

        new_state = eqx.tree_at(lambda s: s.network_state, x, new_network_state)
        new_state = eqx.tree_at(
            lambda s: s.reward_predictor_state, new_state, new_reward_predictor_state
        )

        # Compute the RPE as the TD-error
        RPE = args["RPE_fn"](t, new_state, args, reward)

        new_state = eqx.tree_at(lambda s: s.RPE, new_state, RPE)
        return new_state

    def reset(self, t, x: AgentState, args):
        """Reset the agent by resetting the network (keeping the same weights).

        Note that the reward predictor state is not reset, as we want to maintain the learned reward predictions.
        """
        new_network_state = self.network.reset(t, x.network_state, args)
        new_reward_predictor_state = self.reward_prediction_model.reset(
            t, x.reward_predictor_state, args
        )
        return AgentState(
            new_network_state, new_reward_predictor_state, jnp.zeros_like(x.RPE)
        )
