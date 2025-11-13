import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.analytics import (
    compute_expected_synaptic_std,
    compute_oup_diffusion_coefficient,
    compute_required_input_weight,
)
from adaptive_SNN.visualization import plot_simulate_SNN_results


def main():
    t0 = 0
    t1 = 2
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    N_neurons = 1
    N_inputs = 2

    input_weight = compute_required_input_weight(
        target_mean_g_syn=50e-9,
        N_inputs=1.0,
        tau=LIFNetwork.tau_E,
        input_rate=exc_rate,
        synaptic_increment=LIFNetwork.synaptic_increment,
    )

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        fully_connected_input=True,
        input_weight=input_weight,
        key=key,
        dt=dt0,
    )

    syn_std_E = compute_expected_synaptic_std(
        N_inputs=1.0,
        input_rate=exc_rate,
        tau=neuron_model.tau_E,
        synaptic_increment=neuron_model.synaptic_increment,
        input_weight=input_weight,
    )

    D = compute_oup_diffusion_coefficient(
        target_std=syn_std_E * 0.2, tau=neuron_model.tau_E
    )
    noise_model = OUP(tau=neuron_model.tau_E, noise_scale=D, dim=N_neurons)

    key, _ = jr.split(key)
    noise_model = OUP(tau=neuron_model.tau_E, noise_scale=D, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    args = {
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey(jnp.int32(jnp.round(t / dt0))), rates * dt0, shape=(N_inputs,)
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([5.0]),
    }

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_simulate_SNN_results(sol, model, split_noise=True)


if __name__ == "__main__":
    main()
