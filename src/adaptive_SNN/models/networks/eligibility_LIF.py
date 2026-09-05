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
        # When learning I weights, perturbations and noise_std both cover E weights
        # (first N) and I weights (second N); otherwise only the E weights.
        if self.learn_I_weights:
            E_perturbations = state.perturbations[: self.N_neurons]
            I_perturbations = state.perturbations[self.N_neurons :]
            E_noise_std = noise_std[: self.N_neurons]
            I_noise_std = noise_std[self.N_neurons :]
        else:
            E_perturbations = state.perturbations
            I_perturbations = jnp.zeros((self.N_neurons,))
            E_noise_std = noise_std
            I_noise_std = noise_std

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set the perturbations to zero
        E_perturbations = jnp.where(
            E_noise_std != 0.0, E_perturbations / E_noise_std, 0.0
        )
        I_perturbations = jnp.where(
            I_noise_std != 0.0, I_perturbations / I_noise_std, 0.0
        )

        d_eligibility_E = (
            E_perturbations[:, None]
            / self.synaptic_increment
            * self.excitatory_mask[None, :]
            * state.G
        )
        d_eligibility_I = (
            I_perturbations[:, None]
            / self.synaptic_increment
            * self.inhibitory_mask[None, :]
            * state.G
        )
        d_eligibility_decay = -state.features.eligibility / self.tau_eligibility
        d_eligibility = d_eligibility_E + d_eligibility_I + d_eligibility_decay
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
