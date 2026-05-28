"""Per-synapse eligibility-trace consolidation rule.

Sibling of GatedLIFNetwork that consumes per-synapse OU noise (alpha = 1)
instead of broadcasting per-neuron OU noise. Implements the (alpha=1, gate-on,
eligibility) co-headline cell of unified §4.3.
"""

from __future__ import annotations

import jax
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models.networks._gating import voltage_gate
from adaptive_SNN.models.networks.base import AbstractLIFNetwork
from adaptive_SNN.models.networks.eligibility_LIF import ElibilityState, Eligibility
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class PerSynapseGatedEligibilityLIFNetwork(AbstractLIFNetwork):
    """Eligibility-trace consolidation with per-synapse OU noise + voltage gate.

    Identical to GatedLIFNetwork except the eligibility drift reads the
    per-synapse noise field args["per_synapse_excitatory_noise"] directly,
    rather than broadcasting per-neuron noise across synapses.
    """

    tau_eligibility: float = 0.1
    delta_V: float = 1.0e-3

    def __init__(
        self,
        dt,
        N_neurons: int,
        N_inputs: int = 0,
        tau_eligibility: float = 0.1,
        delta_V: float = 1.0e-3,
        key: jr.PRNGKey = jr.PRNGKey(0),
        **parent_kwargs,
    ):
        super().__init__(dt=dt, N_neurons=N_neurons, N_inputs=N_inputs, key=key, **parent_kwargs)
        self.tau_eligibility = tau_eligibility
        self.delta_V = delta_V

    def init_features(self) -> Eligibility:
        return Eligibility(
            eligibility=jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs))
        )

    def compute_feature_diffusion(self, t, state: ElibilityState, args):
        return jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr),
                else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=arr.dtype)),
            ),
            state.features,
        )

    def compute_feature_drift(self, t, state: ElibilityState, args) -> Eligibility:
        per_synapse_noise = args.get(
            "per_synapse_excitatory_noise", jnp.zeros_like(state.G)
        )
        per_synapse_noise_std = args.get("per_synapse_noise_std", 0.0)
        relative_noise = jnp.where(
            per_synapse_noise_std != 0.0,
            per_synapse_noise / per_synapse_noise_std,
            0.0,
        )
        delta_V = args.get("delta_V", self.delta_V)
        synaptic_traces = state.G
        gate = self.gating_function(state.V, delta_V)[:, None]
        d_eligibility = (
            -state.features.eligibility / self.tau_eligibility
            + relative_noise * synaptic_traces / self.synaptic_increment * gate
        )
        return Eligibility(eligibility=d_eligibility)

    def compute_feature_update(self, t, state: ElibilityState, args) -> Eligibility:
        return state.features

    def noise_shape_features(self) -> Eligibility:
        return Eligibility(eligibility=None)

    def gating_function(self, voltage, delta_V):
        return voltage_gate(
            voltage,
            delta_V,
            reversal_potential_E=self.reversal_potential_E,
            firing_threshold=self.firing_threshold,
            resting_potential=self.resting_potential,
        )

    def compute_weight_updates(self, t, state: ElibilityState, args):
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = args.get("RPE", jnp.array(0.0))
        dW = learning_rate * RPE * state.features.eligibility
        dW = jnp.where(jnp.isnan(state.W), 0.0, dW)
        return dW
