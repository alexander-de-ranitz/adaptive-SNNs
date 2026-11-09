import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_snn.models.networks import AbstractLIFNetwork, LIFState
from adaptive_snn.utils.operators import DefaultIfNone, ElementWiseMul


class GatedEligibility(eqx.Module):
    eligibility: Array  # Shape: (N_neurons, N_neurons + N_inputs)
    mean_scaled_perturbations: (
        Array  # Shape: (N_neurons, ) or (N_neurons * 2, ) if learn_I_weights is True
    )


class GatedLIFState(LIFState):
    features: GatedEligibility


class GatedLIFNetwork(AbstractLIFNetwork):
    tau_eligibility: float = 0.1  # Time constant for eligibility trace
    tau_mean_scaled_noise: float = 10.0  # Time constant for mean eligibility trace
    delta_V: float = 0.001  # Steepness of the gating function

    def init_features(self) -> GatedEligibility:
        return GatedEligibility(
            eligibility=jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs)),
            mean_scaled_perturbations=jnp.zeros((self.N_neurons,))
            if not self.learn_I_weights
            else jnp.zeros((self.N_neurons * 2,)),
        )

    def compute_feature_diffusion(self, t, state: GatedLIFState, args):
        tree = jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr),
                else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=arr.dtype)),
            ),
            state.features,
        )
        return tree

    def compute_feature_drift(self, t, state: GatedLIFState, args) -> GatedEligibility:
        noise_std = self.compute_desired_noise_std(t, state, args)
        delta_V = args.get("delta_V", self.delta_V)

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set the perturbations to zero
        normalized_perturbations = jnp.where(
            noise_std != 0.0, state.perturbations / noise_std, 0.0
        )

        # When learning I weights, perturbations and noise_std both cover E weights
        # (first N) and I weights (second N); otherwise only the E weights.
        gating = self.gating_function(state.V, delta_V, S=state.S[: self.N_neurons])
        if self.learn_I_weights:
            gating = jnp.concatenate([gating, gating])

        scaled_perturbations = normalized_perturbations * gating
        d_mean_scaled_perturbations = (
            scaled_perturbations - state.features.mean_scaled_perturbations
        ) / self.tau_mean_scaled_noise

        subtract_mean_perturbation = args.get("subtract_mean_perturbation", True)
        scaled_perturbations = jnp.where(
            subtract_mean_perturbation,
            scaled_perturbations - state.features.mean_scaled_perturbations,
            scaled_perturbations,
        )

        if self.learn_I_weights:
            E_perturbations = scaled_perturbations[: self.N_neurons]
            I_perturbations = scaled_perturbations[self.N_neurons :]
        else:
            E_perturbations = scaled_perturbations
            I_perturbations = jnp.zeros((self.N_neurons,))

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

        return GatedEligibility(
            eligibility=d_eligibility,
            mean_scaled_perturbations=d_mean_scaled_perturbations,
        )

    def compute_feature_update(self, t, state: GatedLIFState, args) -> GatedEligibility:
        return state.features

    def noise_shape_features(self) -> GatedEligibility:
        return GatedEligibility(eligibility=None, mean_scaled_perturbations=None)

    def gating_function(
        self, voltage: Array, delta_V: float, S: Array | None = None
    ) -> Array:
        """Gating function based on membrane voltage."""
        # Voltage might be temporarily above the firing threshold within a single step,
        # we clip the voltage to ensure the gating function does not become too large in this case
        voltage = jnp.clip(voltage, min=None, max=self.firing_threshold)

        default_area = 1.0 * (
            self.firing_threshold - self.resting_potential
        )  # Area under the default gating function (which is constant at 1)

        # We use delta_V = jnp.inf to represent a binary gating function,
        # which is high when the neuron spikes and 0 otherwise.
        # Note that S is actually 1 step delayed, but this does not make a meaningful difference
        # for us since we use dt=0.1ms and the noise varies on a much slower timescale (6 ms)
        is_binary = jnp.isinf(delta_V)
        if S is None:
            S = jnp.zeros_like(voltage)
        binary_gate = S / self.dt * default_area

        # Compute the gating function for non-binary case
        # Set delta_V to a dummy 1.0 to avoid NaNs
        delta_V = jnp.where(is_binary, 1.0, delta_V)
        driving_force = self.reversal_potential_E - voltage

        integral = lambda V: (self.reversal_potential_E + delta_V - V) * -jnp.exp(
            (V - self.firing_threshold) / delta_V
        )
        area = integral(self.resting_potential) - integral(self.firing_threshold)
        gating = (
            driving_force
            / delta_V
            * jnp.exp((voltage - self.firing_threshold) / delta_V)
        )
        normalization_factor = area / default_area

        gating = gating / normalization_factor

        return jnp.where(is_binary, binary_gate, gating)

    def compute_weight_updates(
        self, t, state: GatedLIFState, args, RPE: Array
    ) -> Array:
        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        dW = learning_rate * RPE * state.features.eligibility

        dW = jnp.where(
            jnp.isnan(state.W), 0.0, dW
        )  # No weight change for non-existing connections
        return dW
