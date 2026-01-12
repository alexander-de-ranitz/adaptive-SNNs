import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.noisy_network import NoisyNetwork, NoisyNetworkState
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.utils.operators import MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class AgentState(eqx.Module):
    noisy_network: NoisyNetworkState
    reward: Array


class Agent(eqx.Module):
    noisy_network: NoisyNetwork
    reward_model: RewardModel

    def __init__(
        self,
        neuron_model: NoisyNetwork,
        reward_model: RewardModel,
    ):
        self.noisy_network = neuron_model
        self.reward_model = reward_model

    @property
    def initial(self):
        return AgentState(
            self.noisy_network.initial,
            self.reward_model.initial,
        )

    def drift(self, t, x: AgentState, args):
        """Compute deterministic time derivatives for LearningModel state.

        The state consists of (network_state, reward_state). The args dict
        must contain the reward received from the environment.

        Args:
            t: time
            x: (network_state, reward_state)
            args: dict containing keys:
                - reward -> scalar
        Returns:
            (d_network_state, d_reward_state)
        """

        (network_state, reward_state) = x.noisy_network, x.reward

        reward = args.get("reward", 0.0)
        RPE = jnp.asarray(reward - reward_state)

        # Add to args for use in models
        args.update(
            {
                "RPE": RPE,
                "reward": reward,
            }
        )

        neuron_drift = self.noisy_network.drift(t, network_state, args)
        reward_drift = self.reward_model.drift(t, reward_state, args)
        return AgentState(neuron_drift, reward_drift)

    def diffusion(self, t, x: AgentState, args):
        (neuron_state, reward_state) = x.noisy_network, x.reward
        neuron_diffusion = self.noisy_network.diffusion(t, neuron_state, args)
        reward_diffusion = self.reward_model.diffusion(t, reward_state, args)
        return MixedPyTreeOperator(AgentState(neuron_diffusion, reward_diffusion))

    @property
    def noise_shape(self):
        return AgentState(
            self.noisy_network.noise_shape,
            self.reward_model.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: AgentState, args):
        (network_state, reward_state) = x.noisy_network, x.reward
        new_network_state = self.noisy_network.update(t, network_state, args)
        return AgentState(new_network_state, reward_state)
