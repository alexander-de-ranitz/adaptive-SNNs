import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import (
    OUP,
    LIFNetwork,
    LIFState,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.analytics import (
    compute_expected_synaptic_std,
    compute_oup_diffusion_coefficient,
    compute_required_input_weight,
)
from adaptive_SNN.visualization import plot_network_stats


def main():
    t0 = 0
    t1 = 1.0
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    N_neurons = 100
    N_inputs = 100

    # Define input parameters
    target_firing_rate = 20
    input_firing_rate = 20

    computed_input_weight = compute_required_input_weight(
        target_mean_g_syn=50e-9,
        N_inputs=jnp.ceil((N_inputs + N_neurons) * 0.8 * 0.1),
        tau=LIFNetwork.tau_E,
        input_rate=target_firing_rate,
        synaptic_increment=LIFNetwork.synaptic_increment,
    )

    # The computed input weight is way too large! This needs to be tuned
    rec_weight = computed_input_weight * 0.15
    input_weight = rec_weight

    balance = 5.0

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt0,
        fully_connected_input=False,
        input_weight=input_weight,
        rec_weight=rec_weight,
        key=key,
    )

    expected_syn_std = compute_expected_synaptic_std(
        N_inputs=0.1 * (N_inputs + N_neurons),
        input_rate=target_firing_rate,
        tau=neuron_model.tau_E,
        synaptic_increment=neuron_model.synaptic_increment,
        input_weight=neuron_model.input_weight,
    )
    D = compute_oup_diffusion_coefficient(
        target_std=0.2 * expected_syn_std,
        tau=neuron_model.tau_E,
    )

    noise_model = OUP(
        tau=neuron_model.tau_E, noise_scale=0 * D, mean=0.0, dim=N_neurons
    )

    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
    )

    solver = dfx.EulerHeun()
    init_state = model.initial

    # Define args
    args = {
        "get_desired_balance": lambda t, x, args: jnp.array([balance]),
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey((t / dt0).astype(int)),
            input_firing_rate * dt0,
            shape=(N_inputs,),
        ),
    }

    def save_fn(y: NoisyNetworkState):
        return LIFState(
            V=y.network_state.V,
            S=y.network_state.S,
            G=y.network_state.G,
            W=y.network_state.W,
            time_since_last_spike=None,
            buffer_index=None,
            spike_buffer=None,
        )

    print("Running simulation...")
    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt0,
        init_state,
        save_every_n_steps=1,
        args=args,
        save_fn=save_fn,
    )

    print("Simulation complete. Generating plots...")
    plot_network_stats(
        sol,
        model,
    )


if __name__ == "__main__":
    main()
