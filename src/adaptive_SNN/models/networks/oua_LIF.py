"""OUA mean-reversion consolidation rule (unified §4.1, impl plan §4.4).

Under the multiplicative-on-weight noise convention (unified §2.1, Eq. 2.1a)
the OUA mean-reversion update is the tau_e -> 0 limit of the eligibility
trace rule (unified §4.5, theorist Q9):

    dW_ij/dt = eta * delta_r * zeta_ij * s_ij * gamma(V_i)

where
    zeta_ij = args["per_synapse_excitatory_noise"]        (the OU excursion)
    s_ij    = state.G[i, j] / synaptic_increment           (synaptic activity)
    gamma   = voltage_gate(V_i, delta_V)  if use_gating, else 1

A mandatory a-priori clip on the per-step update prefactor protects against
surrogate-spike excursions where the gate can transiently blow up as
delta_V -> 0 (theorist Q5; impl plan §4.4). The clip bounds
    |eta * delta_r * gate * clip_factor| <= update_clip.
"""

from __future__ import annotations

import jax
import jax.random as jr
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks._gating import voltage_gate
from adaptive_SNN.models.networks.base import AbstractLIFNetwork, LIFState
from adaptive_SNN.models.networks.default_LIF import NoFeatures
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class OUAMeanReversionLIFNetwork(AbstractLIFNetwork):
    """Per-synapse OUA mean-reversion rule with optional voltage gating.

    Attributes
    ----------
    delta_V : float
        Gating steepness. Smaller delta_V -> narrower, taller gate.
    use_gating : bool
        If False, gate is identically 1 and the clip is inactive.
    update_clip : float
        Absolute upper bound on |eta * delta_r * gate(V_i)| per timestep
        (per neuron, applied before multiplication by noise * activity).
        Theorist Q5: protects against transient gate blow-up in the
        surrogate-spike regime.
    """

    delta_V: float = 1.0e-3
    use_gating: bool = True
    update_clip: float = 1.0

    def __init__(
        self,
        dt,
        N_neurons: int,
        N_inputs: int = 0,
        delta_V: float = 1.0e-3,
        use_gating: bool = True,
        update_clip: float = 1.0,
        key: jr.PRNGKey = jr.PRNGKey(0),
        **parent_kwargs,
    ):
        super().__init__(dt=dt, N_neurons=N_neurons, N_inputs=N_inputs, key=key, **parent_kwargs)
        self.delta_V = delta_V
        self.use_gating = use_gating
        self.update_clip = update_clip

    def init_features(self) -> NoFeatures:
        return NoFeatures()

    def compute_feature_diffusion(self, t, state: LIFState, args) -> NoFeatures:
        return NoFeatures()

    def compute_feature_drift(self, t, state: LIFState, args) -> NoFeatures:
        return NoFeatures()

    def compute_feature_update(self, t, state, args) -> NoFeatures:
        return NoFeatures()

    def noise_shape_features(self) -> NoFeatures:
        return NoFeatures()

    def gating_function(self, voltage: Array, delta_V: float | Array) -> Array:
        return voltage_gate(
            voltage,
            delta_V,
            reversal_potential_E=self.reversal_potential_E,
            firing_threshold=self.firing_threshold,
            resting_potential=self.resting_potential,
        )

    def compute_weight_updates(self, t, state: LIFState, args):
        """OUA mean-reversion weight update.

        Update rule (unified §4.5 / impl plan §4.4):
            dW_ij = eta * delta_r * (zeta_ij / sigma_E) * (G_ij / w0) * gate(V_i)

        Per-synapse geometry (alpha = 1): zeta_ij comes from
        args["per_synapse_excitatory_noise"] (PerSynapseOUP via
        PerSynapseNoisyNetwork).

        Per-neuron geometry (alpha = 0): xi_i comes from
        args["excitatory_noise"] (NeuralNoiseOUP via NoisyNetwork). The
        per-neuron OUA update is then
            dW_ij = eta * delta_r * (xi_i / sigma_xi) * (G_ij / w0) * mask_j * gate(V_i)
        which is the alpha=0 endpoint of unified Eq. 2.4 evaluated under the
        multiplicative-on-weight excursion (the share xi_i / |E_i| is the
        per-synapse marginal of the per-neuron noise, but we factor |E_i|
        into sigma normalisation rather than into the noise term, leaving
        relative_noise = xi_i / sigma_xi as the natural ratio).

        gamma-clip is applied multiplicatively on the per-neuron prefactor.
        """
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = args.get("RPE", jnp.array(0.0))

        # Prefer per-synapse routing when present (alpha = 1 endpoint).
        per_synapse_noise = args.get("per_synapse_excitatory_noise", None)
        if per_synapse_noise is None:
            # Per-neuron fallback (alpha = 0): broadcast xi_i across excitatory
            # synapses; the variance match is encoded in sigma_xi normalisation.
            xi = args.get("excitatory_noise", jnp.zeros((self.N_neurons,)))
            sigma_xi = args.get("noise_std", 0.0)
            exc_mask = self.excitatory_mask.astype(state.G.dtype)
            sigma_xi_arr = jnp.atleast_1d(sigma_xi)
            sigma_xi_arr = jnp.broadcast_to(sigma_xi_arr, (self.N_neurons,))
            rel_per_neuron = jnp.where(sigma_xi_arr != 0.0, xi / sigma_xi_arr, 0.0)
            relative_noise = rel_per_neuron[:, None] * exc_mask[None, :]
        else:
            per_synapse_noise_std = args.get("per_synapse_noise_std", 0.0)
            relative_noise = jnp.where(
                per_synapse_noise_std != 0.0,
                per_synapse_noise / per_synapse_noise_std,
                0.0,
            )

        # s_ij — synaptic activity (G / w0).
        s_ij = state.G / self.synaptic_increment

        # Voltage gate (per-neuron). When disabled, gate is identically 1.
        delta_V = args.get("delta_V", self.delta_V)
        if self.use_gating:
            gate = self.gating_function(state.V, delta_V)
        else:
            gate = jnp.ones_like(state.V)

        # MANDATORY gamma-clip (theorist Q5): bound the per-neuron prefactor
        # |eta * delta_r * gate| <= update_clip. The clip is non-symmetric in
        # the sense that it only scales |gate| down; never up.
        prefactor_mag = jnp.abs(learning_rate * RPE * gate)
        clip_factor = jnp.minimum(
            jnp.array(1.0),
            self.update_clip / (prefactor_mag + 1e-30),
        )
        gate_clipped = gate * clip_factor  # shape (N,)

        # Multiplicative-on-weight OUA update.
        # dW shape: (N, N + N_in)
        dW = (
            learning_rate
            * RPE
            * relative_noise
            * s_ij
            * gate_clipped[:, None]
        )

        # No update for non-existing connections.
        dW = jnp.where(jnp.isnan(state.W), 0.0, dW)
        return dW
