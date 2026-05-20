"""Per-synapse noisy network wrapper.

Routes a 2D per-synapse OU noise process into args["per_synapse_excitatory_noise"]
so that the multiplicative-on-weight injection in
`AbstractLIFNetwork.compute_voltage_update` sees it. This is structurally the
same as `NoisyNetwork` except the noise state has shape
(N_neurons, N_neurons + N_inputs) rather than (N_neurons,).
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.base import LIFState, NeuronModelABC
from adaptive_SNN.models.noise.base import NoiseModelABC
from adaptive_SNN.utils.operators import MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class PerSynapseNoisyNetworkState(eqx.Module):
    """State container for PerSynapseNoisyNetwork.

    The `noise_state` is 2D, one OU process per synapse.
    """

    network_state: LIFState
    noise_state: Array  # shape (N_neurons, N_neurons + N_inputs)


class PerSynapseNoisyNetwork(NeuronModelABC):
    """Compose a base LIF network with per-synapse OU noise.

    The base network's compute_voltage_update reads
    args["per_synapse_excitatory_noise"] and applies it multiplicatively on the
    per-synapse weight before multiplying by the synaptic activity G:
        effective_W_ij = (W_ij + zeta_ij)
        weighted_conductance_ij = effective_W_ij * G_ij
    This is the multiplicative-on-weight convention (unified §2.1, Eq. 2.1a).

    The `per_synapse_noise_std` is the per-synapse OU diffusion strength used
    to normalise weight updates inside the consolidation rule (analogue of
    NoisyNetwork.compute_desired_noise_std for per-neuron noise). For
    overnight r1 we use a *frozen* scalar set externally (after the variance-
    match calibration); online adaptation is a follow-up.
    """

    base_network: NeuronModelABC
    noise_model: NoiseModelABC
    min_noise_std: float = 5e-9

    def __init__(
        self,
        neuron_model: NeuronModelABC,
        noise_model: NoiseModelABC,
        min_noise_std: float | None = None,
    ):
        self.base_network = neuron_model
        self.noise_model = noise_model
        if min_noise_std is not None:
            self.min_noise_std = min_noise_std

    @property
    def initial(self):
        return PerSynapseNoisyNetworkState(
            self.base_network.initial, self.noise_model.initial
        )

    def drift(self, t, state: PerSynapseNoisyNetworkState, args: dict):
        network_state, noise_state = state.network_state, state.noise_state

        # Compute the desired per-synapse noise scale (for r1, a scalar passed
        # in via args; defaults to the noise_model's constructor noise_std).
        per_synapse_noise_std = self.compute_desired_noise_std(t, state, args)

        # Inject the noise so the base network's compute_voltage_update and the
        # consolidation rule can both reach it.
        args.update(
            {
                "per_synapse_excitatory_noise": noise_state,
                "per_synapse_noise_std": per_synapse_noise_std,
            }
        )

        network_drift = self.base_network.drift(t, network_state, args)
        noise_drift = self.noise_model.drift(t, noise_state, args)
        return PerSynapseNoisyNetworkState(network_drift, noise_drift)

    def diffusion(self, t, state: PerSynapseNoisyNetworkState, args: dict):
        network_state, noise_state = state.network_state, state.noise_state
        network_diffusion = self.base_network.diffusion(t, network_state, args)
        noise_diffusion = self.noise_model.diffusion(t, noise_state, args)
        return MixedPyTreeOperator(
            PerSynapseNoisyNetworkState(network_diffusion, noise_diffusion)
        )

    @property
    def noise_shape(self):
        return PerSynapseNoisyNetworkState(
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

    def update(self, t, state: PerSynapseNoisyNetworkState, args):
        new_network_state = self.base_network.update(t, state.network_state, args)
        return PerSynapseNoisyNetworkState(new_network_state, state.noise_state)

    def compute_desired_noise_std(self, t, state: PerSynapseNoisyNetworkState, args):
        """Return the per-synapse OU diffusion strength.

        For overnight r1, callers supply a pre-computed scalar via
        args["per_synapse_noise_std_target"] (the calibrated sigma_E from
        Eq. 2.6 of the unified model). If absent, the noise model's
        constructor value is used.

        `use_noise` is honoured: when False, the effective std is 0 (no noise).
        """
        target_std = args.get(
            "per_synapse_noise_std_target", self.noise_model.noise_std
        )
        use_noise = args.get("use_noise", jnp.array(True))
        # Scalar or 1D (per neuron) is acceptable; keep as-is.
        effective_std = jnp.where(use_noise, target_std, 0.0)
        return effective_std
