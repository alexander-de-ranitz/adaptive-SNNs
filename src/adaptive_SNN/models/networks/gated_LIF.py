import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.base import AbstractLIFNetwork
from adaptive_SNN.models.networks.eligibility_LIF import ElibilityState, Eligibility
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class GatedLIFNetwork(AbstractLIFNetwork):
    tau_eligibility: float = 0.1  # Time constant for eligibility trace

    def init_features(self) -> Eligibility:
        return Eligibility(
            eligibility=jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs))
        )

    def compute_feature_diffusion(self, t, state: ElibilityState, args):
        tree = jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr),
                else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=arr.dtype)),
            ),
            state.features,
        )
        return tree

    def compute_feature_drift(self, t, state: ElibilityState, args) -> Eligibility:
        noise_std = args.get("noise_std", 0.0)
        noise_conductance = args.get("excitatory_noise", jnp.zeros((self.N_neurons,)))

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set relative noise strength to zero
        relative_noise_strength = jnp.where(
            noise_std != 0.0, noise_conductance / noise_std, 0.0
        )

        synaptic_traces = state.G
        d_eligibility = (
            -state.features.eligibility / self.tau_eligibility
            + relative_noise_strength[:, None]
            * synaptic_traces
            / self.synaptic_increment
            * self.gating_function(state.V)[:, None]
        )
        return Eligibility(eligibility=d_eligibility)

    def compute_feature_update(self, t, state: ElibilityState, args) -> Eligibility:
        return state.features

    def noise_shape_features(self) -> Eligibility:
        return Eligibility(eligibility=None)

    def gating_function(self, voltage: Array) -> Array:
        """Gating function based on membrane voltage."""
        # Normalize voltage such that 0 corresponds to resting potential and 1 to firing threshold
        normalised_voltage = (voltage - self.resting_potential) / (
            self.firing_threshold - self.resting_potential
        )

        a = 5.0  # Sharpness of the gating function

        return jnp.exp(a * (normalised_voltage - 1))

    def compute_weight_updates(self, t, state: ElibilityState, args):
        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = args.get("RPE", jnp.array(0.0))

        dW = learning_rate * RPE * state.features.eligibility

        dW = jnp.where(
            state.W == -jnp.inf, 0.0, dW
        )  # No weight change for non-existing connections
        return dW
