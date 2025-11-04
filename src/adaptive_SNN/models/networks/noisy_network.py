import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.base import NeuronModelABC, NoiseModelABC
from adaptive_SNN.models.networks.lif import LIFState
from adaptive_SNN.utils.operators import MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class NoisyNetworkState(eqx.Module):
    """State container for NoisyNetwork."""

    network_state: LIFState
    noise_state: Array


class NoisyNetwork(NeuronModelABC):
    base_network: NeuronModelABC
    noise_model: NoiseModelABC

    def __init__(self, neuron_model: NeuronModelABC, noise_model: NoiseModelABC):
        self.base_network = neuron_model
        self.noise_model = noise_model

    @property
    def initial(self):
        return NoisyNetworkState(self.base_network.initial, self.noise_model.initial)

    def drift(self, t, state: NoisyNetworkState, args):
        network_state, noise_state = (state.network_state, state.noise_state)

        # To allow the base network to access the noise state, we add it to the args
        network_args = {
            **args,
            "excitatory_noise": noise_state,
        }

        network_drift = self.base_network.drift(t, network_state, network_args)
        noise_drift = self.noise_model.drift(t, noise_state, args)
        return NoisyNetworkState(network_drift, noise_drift)

    def diffusion(self, t, state: NoisyNetworkState, args):
        network_state, noise_state = (state.network_state, state.noise_state)
        network_diffusion = self.base_network.diffusion(t, network_state, args)
        noise_diffusion = self.noise_model.diffusion(t, noise_state, args)
        return MixedPyTreeOperator(
            NoisyNetworkState(network_diffusion, noise_diffusion)
        )

    @property
    def noise_shape(self):
        return NoisyNetworkState(
            self.base_network.noise_shape,
            self.noise_model.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, state: NoisyNetworkState, args):
        new_network_state = self.base_network.update(t, state.network_state, args)
        return NoisyNetworkState(new_network_state, state.noise_state)
