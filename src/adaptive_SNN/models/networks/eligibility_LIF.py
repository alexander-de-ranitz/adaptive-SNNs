import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks import AbstractLIFNetwork, LIFState
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class Eligibility(eqx.Module):
    eligibility: Array


class ElibilityState(LIFState):
    features: Eligibility


class EligibilityLIFNetwork(AbstractLIFNetwork):
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
        noise_std = self.compute_desired_noise_std(t, state, args)
        perturbations = state.perturbations

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set relative noise strength to zero
        relative_noise_strength = jnp.where(
            noise_std != 0.0, perturbations / noise_std, 0.0
        )

        # Map the relative noise strength to each excitatory synapse
        noise_per_synapse = jnp.outer(relative_noise_strength, self.excitatory_mask)

        synaptic_traces = state.G
        d_eligibility = (
            -state.features.eligibility / self.tau_eligibility
            + noise_per_synapse * synaptic_traces / self.synaptic_increment
        )
        return Eligibility(eligibility=d_eligibility)

    def compute_feature_update(self, t, state: ElibilityState, args) -> Eligibility:
        return state.features

    def noise_shape_features(self) -> Eligibility:
        return Eligibility(eligibility=None)

    def compute_weight_updates(
        self, t, state: ElibilityState, args, RPE: Array
    ) -> Array:
        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        dW = learning_rate * RPE * state.features.eligibility
        dW = jnp.where(
            jnp.isnan(state.W), 0.0, dW
        )  # No weight change for non-existing connections
        return dW
