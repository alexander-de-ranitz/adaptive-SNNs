import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.base import AbstractLIFNetwork
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class Eligibility(eqx.Module):
    eligibility: Array


class ElibilityState(eqx.Module):
    V: Array
    S: Array
    W: Array
    G: Array
    firing_rate: Array
    mean_E_conductance: Array
    var_E_conductance: Array
    time_since_last_spike: Array
    spike_buffer: Array
    buffer_index: Array  # Scalar array to maintain JAX compatibility
    features: Eligibility


class EligibilityLIFNetwork(AbstractLIFNetwork):
    tau_eligibility: float = 0.3  # Time constant for eligibility trace

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
        )
        return Eligibility(eligibility=d_eligibility)

    def compute_feature_update(self, t, state: ElibilityState, args) -> Eligibility:
        return state.features

    def noise_shape_features(self) -> Eligibility:
        return Eligibility(eligibility=None)

    def compute_weight_updates(self, t, state: ElibilityState, args):
        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = args.get("RPE", jnp.array(0.0))

        dW = learning_rate * RPE * state.features.eligibility

        dW = jnp.where(
            state.W == -jnp.inf, 0.0, dW
        )  # No weight change for non-existing connections
        return dW
