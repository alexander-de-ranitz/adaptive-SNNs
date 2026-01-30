import diffrax as dfx
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import OUP, LIFNetwork, NoisyNetwork
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    t0 = 0
    t1 = 10
    dt0 = 1e-4
    key = jr.PRNGKey(2)

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    N_neurons = 1
    N_inputs = 2

    # noise_levels =  [0.0, 0.1, 0.2, 0.3]
    noise_level = 0.1
    gs = [1.5, 2.0, 2.5, 3.0]
    weights = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0]
    spike_rates = []
    mean_E = []
    mean_std = []
    mean_noise = []
    noise_std = []
    # for i, noise_level in enumerate(noise_levels):
    for i, g in enumerate(gs):
        for j, w in enumerate(weights):
            # Set up models
            neuron_model = LIFNetwork(
                N_neurons=N_neurons,
                N_inputs=N_inputs,
                fully_connected_input=True,
                input_weight=w,
                input_types=jnp.array([1, 0]),
                key=key,
                dt=dt0,
            )

            noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)
            model = NoisyNetwork(
                neuron_model=neuron_model,
                noise_model=noise_model,
                min_noise_std=5e-9,
            )

            # Run simulation
            solver = dfx.EulerHeun()
            init_state = model.initial

            spike_key = jr.fold_in(jr.PRNGKey(0), i * 100 + j)
            sim_key = jr.fold_in(jr.PRNGKey(1), i * 100 + j)

            args = {
                "get_input_spikes": lambda t, x, args: jr.poisson(
                    jr.fold_in(spike_key, (t / dt0).astype(int)),
                    rates * dt0,
                    shape=(N_neurons, N_inputs),
                ),
                "get_desired_balance": lambda t, x, args: jnp.array([g]),
                "noise_scale_hyperparam": noise_level,
            }

            def save_fn(t, state, args):
                return save_part_of_state(
                    state,
                    S=True,
                    V=True,
                    G=True,
                    W=True,
                    mean_E_conductance=True,
                    noise_state=True,
                    var_E_conductance=True,
                )

            sol = simulate_noisy_SNN(
                model,
                solver,
                t0,
                t1,
                dt0,
                init_state,
                save_at=SaveAt(t0=True, t1=True, steps=True, fn=save_fn),
                args=args,
                key=sim_key,
            )

            n_spikes = jnp.sum(sol.ys.network_state.S)
            spike_rates += [n_spikes / t1]
            cutoff = int(5.0 / dt0)  # Ignore first second for statistics
            mean_E += [jnp.mean(sol.ys.network_state.mean_E_conductance[cutoff:])]
            mean_std += [
                jnp.mean(jnp.sqrt(sol.ys.network_state.var_E_conductance[cutoff:]))
            ]
            mean_noise += [jnp.mean(sol.ys.noise_state[cutoff:])]
            noise_std += [jnp.std(sol.ys.noise_state[cutoff:])]

    noise_levels = gs
    spike_rates = jnp.array(spike_rates).reshape(len(noise_levels), len(weights))
    mean_E = jnp.array(mean_E).reshape(len(noise_levels), len(weights))
    mean_std = jnp.array(mean_std).reshape(len(noise_levels), len(weights))
    mean_noise = jnp.array(mean_noise).reshape(len(noise_levels), len(weights))
    noise_std = jnp.array(noise_std).reshape(len(noise_levels), len(weights))

    plt.subplot(1, 2, 1)
    for i, noise_level in enumerate(noise_levels):
        plt.plot(weights, spike_rates[i, :], label=f"g: {noise_level}")
    plt.xlabel("Input Weight")
    plt.ylabel("Output Firing Rate (Hz)")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, noise_level in enumerate(noise_levels):
        plt.plot(weights, noise_std[i, :] / mean_std[i, :], label=f"g: {noise_level}")
    plt.xlabel("Input Weight")
    plt.ylabel("Noise Std / Excitatory Conductance Std")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
