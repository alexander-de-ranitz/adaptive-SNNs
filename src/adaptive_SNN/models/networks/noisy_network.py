import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.base import NeuronModelABC
from adaptive_SNN.models.networks.lif import LIFState
from adaptive_SNN.models.noise.oup import OUP
from adaptive_SNN.utils.operators import MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class NoisyNetworkState(eqx.Module):
    """State container for NoisyNetwork."""

    network_state: LIFState
    noise_E_state: Array  # (N_neurons,)
    noise_I_state: Array  # (N_neurons,)


class NoisyNetwork(NeuronModelABC):
    base_network: NeuronModelABC
    noise_E: OUP  # TODO: might be better to use a single OUP with 2 dimensions
    noise_I: OUP
    # TODO: For generability, it would be better to allow any noise model. We would have an ABC NoiseModel that OUP inherits from.

    def __init__(
        self, neuron_model: NeuronModelABC, noise_I_model: OUP, noise_E_model: OUP
    ):
        self.base_network = neuron_model
        self.noise_E = noise_E_model
        self.noise_I = noise_I_model

        assert self.noise_E.dim == self.base_network.N_neurons, (
            "Dimension of excitatory noise must match number of neurons"
        )
        assert self.noise_I.dim == self.base_network.N_neurons, (
            "Dimension of inhibitory noise must match number of neurons"
        )

    @property
    def initial(self):
        return NoisyNetworkState(
            self.base_network.initial, self.noise_E.initial, self.noise_I.initial
        )

    def drift(self, t, state: NoisyNetworkState, args):
        network_state, noise_E_state, noise_I_state = (
            state.network_state,
            state.noise_E_state,
            state.noise_I_state,
        )

        # TODO: As above, this could be made more general by allowing any noise model and just passing that
        network_args = {
            **args,
            "excitatory_noise": noise_E_state,
            "inhibitory_noise": noise_I_state,
        }

        network_drift = self.base_network.drift(t, network_state, network_args)
        noise_E_drift = self.noise_E.drift(t, noise_E_state, args)
        noise_I_drift = self.noise_I.drift(t, noise_I_state, args)
        return NoisyNetworkState(network_drift, noise_E_drift, noise_I_drift)

    def diffusion(self, t, state: NoisyNetworkState, args):
        network_state, noise_E_state, noise_I_state = (
            state.network_state,
            state.noise_E_state,
            state.noise_I_state,
        )
        network_diffusion = self.base_network.diffusion(t, network_state, args)
        noise_E_diffusion = self.noise_E.diffusion(t, noise_E_state, args)
        noise_I_diffusion = self.noise_I.diffusion(t, noise_I_state, args)
        return MixedPyTreeOperator(
            NoisyNetworkState(network_diffusion, noise_E_diffusion, noise_I_diffusion)
        )

    @property
    def noise_shape(self):
        return NoisyNetworkState(
            self.base_network.noise_shape,
            self.noise_E.noise_shape,
            self.noise_I.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: NoisyNetworkState, args):
        network_state, noise_E_state, noise_I_state = (
            x.network_state,
            x.noise_E_state,
            x.noise_I_state,
        )
        new_network_state = self.base_network.update(t, network_state, args)
        return NoisyNetworkState(new_network_state, noise_E_state, noise_I_state)
