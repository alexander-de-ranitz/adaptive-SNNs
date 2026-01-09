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
    min_noise_std: float = 0e-9  # TODO: Tune this, and/or make it configurable

    base_network: NeuronModelABC
    noise_model: NoiseModelABC

    def __init__(self, neuron_model: NeuronModelABC, noise_model: NoiseModelABC):
        self.base_network = neuron_model
        self.noise_model = noise_model

    @property
    def initial(self):
        return NoisyNetworkState(self.base_network.initial, self.noise_model.initial)

    def drift(self, t, state: NoisyNetworkState, args: dict):
        network_state, noise_state = (state.network_state, state.noise_state)

        # To allow the base network to access the noise state and std, we add it to the args
        args.update(
            {
                "excitatory_noise": noise_state,
                "noise_std": self.compute_desired_noise_std(t, state, args),
            }
        )

        network_drift = self.base_network.drift(t, network_state, args)
        noise_drift = self.noise_model.drift(t, noise_state, args)
        return NoisyNetworkState(network_drift, noise_drift)

    def diffusion(self, t, state: NoisyNetworkState, args: dict):
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

    def compute_desired_noise_std(self, t, state: NoisyNetworkState, args):
        """For each neuron, compute the desired scale of the noise to be added.

        The desired level of noise scales with the total weighted input to the neuron. Weighted with a hyperparameter-defined factor.
        The returned noise scale represents the standard deviation of the noise to be added to each neuron.

        Returns:
            Array: Noise scale for each neuron.
        """
        synaptic_variance = state.network_state.auxiliary_info.var_E_conductance

        # Compute desired noise std using the computed variance and a hyperparameter, then clip to min value
        noise_scale_hyperparam = args.get("noise_scale_hyperparam", 0.0)
        if noise_scale_hyperparam == 0.0:
            return jnp.zeros_like(synaptic_variance)
        desired_noise_std = jnp.sqrt(synaptic_variance) * noise_scale_hyperparam
        desired_noise_std = jnp.clip(desired_noise_std, min=self.min_noise_std)
        return desired_noise_std
