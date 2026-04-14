import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.noisy_network import NoisyNetwork, NoisyNetworkState
from adaptive_SNN.models.noise import OUP
from adaptive_SNN.models.noise.base import NoiseModelABC
from adaptive_SNN.models.reward import AbstractRewardModel
from adaptive_SNN.models.RPE import AbstractRPEModel
from adaptive_SNN.utils.operators import MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class AgentState(eqx.Module):
    noisy_network: NoisyNetworkState
    predicted_reward: Array
    reward_noise: Array
    RPE: Array


class Agent(eqx.Module):
    noisy_network: NoisyNetwork
    reward_model: AbstractRewardModel
    reward_noise: NoiseModelABC
    RPE_model: AbstractRPEModel

    def __init__(
        self,
        neuron_model: NoisyNetwork,
        reward_model: AbstractRewardModel,
        reward_noise: OUP,
        RPE_model: AbstractRPEModel,
    ):
        self.noisy_network = neuron_model
        self.reward_model = reward_model
        self.reward_noise = reward_noise
        self.RPE_model = RPE_model

    @property
    def initial(self):
        return AgentState(
            self.noisy_network.initial,
            self.reward_model.initial,
            self.reward_noise.initial,
            self.RPE_model.initial,
        )

    def drift(self, t, x: AgentState, args):
        """Compute deterministic time derivatives for LearningModel state.

        The state consists of (network_state, predicted_reward). The args dict
        must contain the reward received from the environment.

        Args:
            t: time
            x: (network_state, predicted_reward)
            args: dict containing keys:
                - reward -> scalar
        Returns:
            (d_network_state, d_predicted_reward)
        """

        (network_state, predicted_reward, reward_noise, RPE) = (
            x.noisy_network,
            x.predicted_reward,
            x.reward_noise,
            x.RPE,
        )

        args.update(
            {"RPE": RPE + reward_noise}
        )  # Add RPE to args so that it can be used for weight updates in the network drift

        neuron_drift = self.noisy_network.drift(t, network_state, args)
        reward_drift = self.reward_model.drift(t, predicted_reward, args)
        reward_noise_drift = self.reward_noise.drift(t, reward_noise, args)
        RPE_drift = self.RPE_model.drift(t, RPE, args)
        return AgentState(neuron_drift, reward_drift, reward_noise_drift, RPE_drift)

    def diffusion(self, t, x: AgentState, args):
        (neuron_state, predicted_reward, reward_noise, RPE) = (
            x.noisy_network,
            x.predicted_reward,
            x.reward_noise,
            x.RPE,
        )
        neuron_diffusion = self.noisy_network.diffusion(t, neuron_state, args)
        reward_diffusion = self.reward_model.diffusion(t, predicted_reward, args)
        reward_noise_diffusion = self.reward_noise.diffusion(t, reward_noise, args)
        RPE_diffusion = self.RPE_model.diffusion(t, RPE, args)
        return MixedPyTreeOperator(
            AgentState(
                neuron_diffusion,
                reward_diffusion,
                reward_noise_diffusion,
                RPE_diffusion,
            )
        )

    @property
    def noise_shape(self):
        return AgentState(
            self.noisy_network.noise_shape,
            self.reward_model.noise_shape,
            self.reward_noise.noise_shape,
            self.RPE_model.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: AgentState, args):
        (network_state, predicted_reward, reward_noise, RPE) = (
            x.noisy_network,
            x.predicted_reward,
            x.reward_noise,
            x.RPE,
        )
        new_network_state = self.noisy_network.update(t, network_state, args)
        new_predicted_reward = self.reward_model.update(t, predicted_reward, args)
        new_reward_noise = self.reward_noise.update(t, reward_noise, args)
        new_RPE = self.RPE_model.update(t, RPE, args)
        return AgentState(
            new_network_state, new_predicted_reward, new_reward_noise, new_RPE
        )
