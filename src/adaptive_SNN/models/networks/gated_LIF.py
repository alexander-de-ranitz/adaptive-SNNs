import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.base import AbstractLIFNetwork
from adaptive_SNN.models.networks.eligibility_LIF import ElibilityState, Eligibility
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class GatedLIFNetwork(AbstractLIFNetwork):
    tau_eligibility: float = 1.0  # Time constant for eligibility trace
    delta_V: float = 0.002  # Steepness of the gating function

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

        # Map the relative noise strength to each excitatory synapse
        noise_per_synapse = jnp.outer(relative_noise_strength, self.excitatory_mask)

        synaptic_traces = state.G
        d_eligibility = (
            -state.features.eligibility / self.tau_eligibility
            + noise_per_synapse
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
        default_area = 1.0 * (
            self.firing_threshold - self.resting_potential
        )  # Area under the default gating function (which is constant at 1)
        driving_force = self.reversal_potential_E - voltage

        integral = lambda V: (self.reversal_potential_E + self.delta_V - V) * -jnp.exp(
            (V - self.firing_threshold) / self.delta_V
        )
        area = integral(self.resting_potential) - integral(self.firing_threshold)
        gating = (
            driving_force
            / self.delta_V
            * jnp.exp((voltage - self.firing_threshold) / self.delta_V)
        )
        normalization_factor = area / default_area

        return gating / normalization_factor

    def compute_weight_updates(self, t, state: ElibilityState, args):
        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = jnp.sum(jnp.asarray(args.get("RPE", 0.0)))

        dW = learning_rate * RPE * state.features.eligibility

        dW = jnp.where(
            state.W == -jnp.inf, 0.0, dW
        )  # No weight change for non-existing connections
        return dW


def plot_gating_function():
    import matplotlib.pyplot as plt

    network = GatedLIFNetwork(N_neurons=1, dt=1e-4, N_inputs=0)
    voltages = jnp.linspace(-75 * 1e-3, -50 * 1e-3, 100)  # From -80 mV to +20 mV
    gating_values = network.gating_function(voltages)
    plt.figure(figsize=(3.5, 2))
    plt.hlines(
        1.0,
        -77 * 1e-3,
        -48 * 1e-3,
        colors="k",
        linestyles="--",
        label="Constant gating function",
    )
    plt.plot(voltages, gating_values, c="k", label="Voltage-dependent gating function")
    plt.xlabel("Membrane Voltage (mV)")
    plt.xlim(voltages[0], voltages[-1])
    plt.xticks(
        jnp.arange(-75, -50 + 1, 5) * 1e-3, labels=[-75, -70, -65, -60, -55, -50]
    )
    plt.ylabel("Gating Function Value")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.savefig("../figures/gating_function.pdf")
    plt.show()


if __name__ == "__main__":
    plot_gating_function()
