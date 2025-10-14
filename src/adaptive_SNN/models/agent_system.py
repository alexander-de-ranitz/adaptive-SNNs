import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_SNN.models.environment import EnvironmentModel
from adaptive_SNN.models.networks.noisy_network import NoisyNetwork
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.utils.operators import MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class AgentSystem(eqx.Module):
    noisy_network: NoisyNetwork
    reward_model: RewardModel
    environment: EnvironmentModel

    def __init__(
        self,
        neuron_model: NoisyNetwork,
        reward_model: RewardModel,
        environment: EnvironmentModel,
    ):
        self.noisy_network = neuron_model
        self.reward_model = reward_model
        self.environment = environment

    @property
    def initial(self):
        return (
            self.noisy_network.initial,
            self.reward_model.initial,
            self.environment.initial,
        )

    def drift(self, t, x, args):
        """Compute deterministic time derivatives for LearningModel state.

        The state consists of (network_state, reward_state, environment_state). The args dict
        must contain functions to compute the network output and reward, which are used to compute
        the reward prediction error (RPE) for learning.

        Args:
            t: time
            x: (network_state, reward_state, environment_state)
            args: dict containing keys:
                - network_output_fn(t, network_state, args) -> scalar
                - reward_fn(t, environment_state, args) -> scalar
        Returns:
            (d_network_state, d_reward_state, d_environment_state)
        """

        (network_state, reward_state, env_state) = x

        # Compute network output, reward, and RPE
        network_output = args["network_output_fn"](t, network_state, args)
        reward = args["reward_fn"](t, env_state, args)
        RPE = jnp.asarray(reward - reward_state)

        # Add to args for use in models
        args = {
            **args,
            "env_input": network_output,
            "RPE": RPE,
            "reward": reward,
        }

        neuron_drift = self.noisy_network.drift(t, network_state, args)
        reward_drift = self.reward_model.drift(t, reward_state, args)
        env_drift = self.environment.drift(t, env_state, args)
        return (neuron_drift, reward_drift, env_drift)

    def diffusion(self, t, x, args):
        (neuron_state, reward_state, env_state) = x
        neuron_diffusion = self.noisy_network.diffusion(t, neuron_state, args)
        reward_diffusion = self.reward_model.diffusion(t, reward_state, args)
        env_diffusion = self.environment.diffusion(t, env_state, args)
        return MixedPyTreeOperator((neuron_diffusion, reward_diffusion, env_diffusion))

    @property
    def noise_shape(self):
        return (
            self.noisy_network.noise_shape,
            self.reward_model.noise_shape,
            self.environment.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        (network_state, reward_state, env_state) = x
        new_network_state = self.noisy_network.update(t, network_state, args)
        return (new_network_state, reward_state, env_state)
