"""Per-synapse noisy network wrappers.

Two wrappers in this module:

- `PerSynapseNoisyNetwork`: 2D per-synapse OU state routed into
  args["per_synapse_excitatory_noise"]. The (alpha=1) endpoint of unified
  Eq. 2.4.

- `BroadcastingNoisyNetwork`: per-neuron 1D OU state broadcast across
  excitatory synapses, equivalent to the (alpha=0) endpoint of unified
  Eq. 2.4. Used to pair the per-neuron variants (A-PN, B-PN) with their
  per-synapse twins (A-PS, B-PS) under a shared OUA / eligibility update
  interface that reads args["per_synapse_excitatory_noise"].
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.base import LIFState, NeuronModelABC
from adaptive_SNN.models.networks.noisy_network import NoisyNetworkState
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

        Returned shape matches the 2D noise state (N_neurons, N_neurons + N_inputs)
        for clean broadcast against per_synapse_excitatory_noise downstream.
        """
        target_std = args.get(
            "per_synapse_noise_std_target", self.noise_model.noise_std
        )
        use_noise = args.get("use_noise", jnp.array(True))
        effective_scalar = jnp.where(use_noise, target_std, 0.0)
        # Broadcast to 2D state shape.
        N = self.base_network.N_neurons
        N_in = self.base_network.N_inputs
        effective_scalar = jnp.broadcast_to(
            jnp.atleast_1d(effective_scalar), (N,)
        )
        return jnp.broadcast_to(effective_scalar[:, None], (N, N + N_in)).astype(default_float)


class BroadcastingNoisyNetwork(NeuronModelABC):
    """Per-neuron OU noise broadcast across excitatory synapses.

    Routes the per-neuron OU noise state into
    args["per_synapse_excitatory_noise"] as
        zeta_ij = xi_i / |E_i| * mask_j
    so that the sum over excitatory synapses recovers xi_i. This is the
    alpha = 0 endpoint of unified Eq. 2.4 expressed in the per-synapse
    routing slot, letting per-synapse-aware consolidation rules
    (OUAMeanReversionLIFNetwork, PerSynapseGatedEligibilityLIFNetwork) be
    used at the per-neuron geometry with shared code.

    The per-synapse noise std is set to (sigma_xi / |E_i|) to match: the
    relative noise xi_i / sigma_xi is preserved per excitatory synapse.

    `|E_i|` (number of existing excitatory synapses per neuron) is computed
    from the initial weight matrix at construction time and is treated as
    a constant during the simulation.
    """

    base_network: NeuronModelABC
    noise_model: NoiseModelABC
    excitatory_count_per_neuron: Array  # shape (N_neurons,)
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
        # Count existing excitatory synapses per postsynaptic neuron from the
        # initial weight matrix.
        W0 = neuron_model.initial.W
        exc_mask = neuron_model.excitatory_mask
        existing = ~jnp.isnan(W0) & exc_mask[None, :]
        count = jnp.sum(existing, axis=1).astype(default_float)
        # Avoid division by zero: replace 0 with 1 (those neurons have no E
        # input so the noise has nothing to broadcast onto anyway).
        self.excitatory_count_per_neuron = jnp.where(count > 0, count, 1.0)

    @property
    def initial(self):
        return NoisyNetworkState(self.base_network.initial, self.noise_model.initial)

    def _broadcast(self, noise_state, noise_std):
        """Build (N, N+N_in) per-synapse noise from per-neuron OU state.

        The per-neuron noise xi_i has units of conductance (nS) and is added
        to the aggregate E conductance in NoisyNetwork's path. For broadcast
        through the multiplicative-on-weight per-synapse path it must be
        converted to W (dimensionless) units: divide by synaptic_increment.
        The broadcast also divides by |E_i| so the aggregate contribution
        sum_j (xi_i/(|E_i|*w0)) * G_ij = (xi_i / w0) * <G>_active ~ xi_i in nS.
        """
        exc_mask = self.base_network.excitatory_mask.astype(default_float)
        w0 = self.base_network.synaptic_increment
        per_neuron_scaled = noise_state / (self.excitatory_count_per_neuron * w0)
        per_syn_noise = per_neuron_scaled[:, None] * exc_mask[None, :]
        std_1d = jnp.broadcast_to(
            noise_std / (self.excitatory_count_per_neuron * w0),
            (self.base_network.N_neurons,),
        )
        per_syn_std = jnp.broadcast_to(
            std_1d[:, None], per_syn_noise.shape
        ).astype(default_float)
        return per_syn_noise, per_syn_std

    def drift(self, t, state: NoisyNetworkState, args: dict):
        network_state, noise_state = state.network_state, state.noise_state

        # Compute the per-neuron OU diffusion strength (matches Alexander's
        # NoisyNetwork.compute_desired_noise_std for back-compat with the per-
        # neuron OUP diffusion).
        per_neuron_std = self.compute_desired_noise_std(t, state, args)

        # Broadcast into per-synapse routing slot ONLY. We do NOT also set
        # excitatory_noise: that would double-count noise into the per-neuron
        # additive injection AND the per-synapse multiplicative injection.
        per_syn_noise, per_syn_std = self._broadcast(noise_state, per_neuron_std)
        args.update(
            {
                # noise_std exposed so the per-neuron OUP diffusion sees the
                # activity-dependent std (NeuralNoiseOUP reads "noise_std").
                "noise_std": per_neuron_std,
                "per_synapse_excitatory_noise": per_syn_noise,
                "per_synapse_noise_std": per_syn_std,
            }
        )

        network_drift = self.base_network.drift(t, network_state, args)
        noise_drift = self.noise_model.drift(t, noise_state, args)
        return NoisyNetworkState(network_drift, noise_drift)

    def diffusion(self, t, state: NoisyNetworkState, args: dict):
        network_state, noise_state = state.network_state, state.noise_state
        network_diffusion = self.base_network.diffusion(t, network_state, args)
        noise_diffusion = self.noise_model.diffusion(t, noise_state, args)
        return MixedPyTreeOperator(
            NoisyNetworkState(network_diffusion, noise_diffusion)
        )

    @property
    def noise_shape(self):
        return NoisyNetworkState(
            self.base_network.noise_shape, self.noise_model.noise_shape
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
        synaptic_variance = state.network_state.var_E_conductance
        use_noise = args.get(
            "use_noise", jnp.array([False] * self.base_network.N_neurons)
        )
        noise_scale_hyperparam = args.get("noise_scale_hyperparam", 0.0)
        desired_noise_std = jnp.sqrt(synaptic_variance) * noise_scale_hyperparam
        desired_noise_std = self.min_noise_std + desired_noise_std
        desired_noise_std = jnp.where(use_noise, desired_noise_std, 0.0)
        return desired_noise_std
