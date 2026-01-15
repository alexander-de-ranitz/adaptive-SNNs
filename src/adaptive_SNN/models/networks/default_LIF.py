import equinox as eqx
from jax import numpy as jnp

from adaptive_SNN.models.networks.base import AbstractLIFNetwork, LIFState


class NoFeatures(eqx.Module):
    pass


class LIFNetwork(AbstractLIFNetwork):
    def init_features(self) -> NoFeatures:
        return NoFeatures()

    def compute_feature_diffusion(self, t, state: LIFState, args) -> NoFeatures:
        return NoFeatures()

    def compute_feature_drift(self, t, state: LIFState, args) -> NoFeatures:
        return NoFeatures()

    def compute_feature_update(self, t, state, args):
        return NoFeatures()

    def noise_shape_features(self):
        return NoFeatures()

    def compute_weight_updates(self, t, state: LIFState, args):
        """Compute synaptic weight changes based on noise-driven plasticity rule.

        The weight updates are computed based on the reward prediction error (RPE), synaptic activity, and the noise present in the excitatory conductances.
        dW_ij = learning_rate * RPE * noise_i * (conductance_ij / synaptic_increment)

        The noise term is normalized by the desired noise standard deviation to decouple absolute noise levels from the magnitude of weight changes.
        The weight updates are only applied to existing connections (weights != -inf).

        Args:
            t: Current time
            state: Current LIFState
            args: Dictionary of additional arguments, must contain:
                - get_learning_rate(t, state, args) -> scalar
                - RPE(t, state, args) -> scalar
                - excitatory_noise: Array of shape (N_neurons,) representing external excitatory conductance noise
                - noise_std: Array of shape (N_neurons,) representing desired noise standard deviation
                If the RPE, excitatory_noise, or noise_std are not provided, they default to 0 (no noise -> no weight change)

        Returns:
            dW: Array of shape (N_neurons, N_neurons + N_inputs) representing synaptic weight changes
        """
        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = args.get("RPE", jnp.array(0.0))

        noise_std = args.get("noise_std", 0.0)
        noise_conductance = args.get("excitatory_noise", jnp.zeros((self.N_neurons,)))

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set relative noise strength to zero
        relative_noise_strength = jnp.where(
            noise_std != 0.0, noise_conductance / noise_std, 0.0
        )

        # Map the relative noise strength to each excitatory synapse
        noise_per_synapse = jnp.outer(relative_noise_strength, self.excitatory_mask)

        dW = (
            learning_rate
            * RPE
            * noise_per_synapse
            * (state.G / self.synaptic_increment)
        )  # Since W is in arbitrary units (not nS), scale G by synaptic increment to get a sensible scale

        dW = jnp.where(
            state.W == -jnp.inf, 0.0, dW
        )  # No weight change for non-existing connections
        return dW
