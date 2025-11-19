import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.base import NeuronModelABC, NoiseModelABC
from adaptive_SNN.models.networks.lif import LIFNetwork, LIFState
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

        # The noise diffusion is state-dependent; compute it here and pass it to the noise model via args
        noise_args = {
            **args,
            "noise_std": self.compute_desired_noise_std(t, state, args),
        }
        noise_diffusion = self.noise_model.diffusion(t, noise_state, noise_args)
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

    def compute_desired_noise_std(self, t, state: NoisyNetworkState, args):
        """For each neuron, compute the desired scale of the noise to be added.

        The desired level of noise scales with the total weighted input to the neuron. Weighted with a hyperparameter-defined factor.
        The returned noise scale represents the standard deviation of the noise to be added to each neuron.

        Returns:
            Array: Noise scale for each neuron.
        """
        # TODO: Should we compute this instead?
        assumed_firing_rate = 5000.0  # We assume input neurons fire at 10 Hz on average

        # Compute the variance of the conductance fluctuations per neuron due to synaptic input
        # based on Campbell's theorem (see Papoulis, 2002)
        W = jnp.where(jnp.isfinite(state.network_state.W), state.network_state.W, 0.0)

        # Only consider excitatory weights for noise computation
        W = W * self.base_network.excitatory_mask[None, :]

        synaptic_variance = (
            0.5
            * jnp.square(W).sum(axis=1)
            * assumed_firing_rate
            * LIFNetwork.synaptic_increment**2
            * LIFNetwork.tau_E
        )

        return jnp.sqrt(synaptic_variance) * args.get("noise_scale_hyperparam", 0.0)
