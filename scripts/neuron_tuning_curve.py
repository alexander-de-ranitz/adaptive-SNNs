import time

import diffrax as dfx
import equinox as eqx
import jax.random as jr
import numpy as np
from jax import numpy as jnp
from joblib import memory
from matplotlib import pyplot as plt

from adaptive_SNN.models import OUP, LIFNetwork, NoisyNetwork, NoisyNetworkState
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.analytics import (
    compute_expected_synaptic_std,
    compute_oup_diffusion_coefficient,
    compute_required_input_weight,
)

memory = memory.Memory(location="./.joblib_cache", verbose=0)


def main():
    @memory.cache
    def get_data(input_rates, conductances):
        t0 = 0
        t1 = 5
        dt = 1e-4
        key = jr.PRNGKey(1)

        N_neurons = 1
        N_inputs = 2

        noise_factor = 0.0  # Fraction of expected synaptic std to set as noise std
        data = {}
        # Set up models
        neuron_model = LIFNetwork(
            N_neurons=N_neurons,
            N_inputs=N_inputs,
            fully_connected_input=True,
            input_weight=1.0,
            key=key,
            dt=dt,
        )

        N_sim = len(input_rates) * len(conductances)
        sim_counter = 0
        for input_rate in input_rates:
            for mean_g in conductances:
                start = time.time()
                exc_to_inh_ratio = 4.0
                inh_rate = input_rate / exc_to_inh_ratio
                rates = jnp.array([input_rate, inh_rate])  # firing rate in Hz

                input_weight = compute_required_input_weight(
                    target_mean_g_syn=mean_g,
                    N_inputs=1.0,
                    tau=LIFNetwork.tau_E,
                    input_rate=input_rate,
                    synaptic_increment=LIFNetwork.synaptic_increment,
                )

                syn_std_E = compute_expected_synaptic_std(
                    N_inputs=1.0,
                    input_rate=input_rate,
                    tau=neuron_model.tau_E,
                    synaptic_increment=neuron_model.synaptic_increment,
                    input_weight=input_weight,
                )

                D = compute_oup_diffusion_coefficient(
                    target_std=syn_std_E * noise_factor, tau=neuron_model.tau_E
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

                init_state = eqx.tree_at(
                    lambda s: s.network_state.W,
                    init_state,
                    init_state.network_state.W * input_weight,
                )

                args = {
                    "get_input_spikes": lambda t, x, args: jr.poisson(
                        jr.PRNGKey(jnp.int32(jnp.round(t / dt))),
                        rates * dt,
                        shape=(N_inputs,),
                    ),
                    "get_desired_balance": lambda t, x, args: jnp.array([5.0]),
                }

                def save_fn(y: NoisyNetworkState):
                    return y.network_state.S[0], y.network_state.V[0]

                sol = simulate_noisy_SNN(
                    model,
                    solver,
                    t0,
                    t1,
                    dt,
                    init_state,
                    save_every_n_steps=1,
                    args=args,
                    save_fn=save_fn,
                )

                data[(input_rate, mean_g)] = (
                    jnp.sum(sol.ys[0]) / (t1 - t0),
                    sol.ys[1],
                )  # Firing rate, voltage trace

                end = time.time()
                print(
                    f"Completed simulation {sim_counter + 1}/{N_sim} in {end - start:.2f} seconds.",
                    end="\r",
                )
                sim_counter += 1
        return data

    input_rates = np.logspace(np.log10(10), np.log10(5000), 20)
    conductances = np.logspace(np.log10(1e-9), np.log10(50e-9), 20)

    data = get_data(input_rates, conductances)

    data_array = jnp.array(
        [[data[(ir, mg)][0] for mg in conductances] for ir in input_rates]
    ).reshape(len(input_rates), len(conductances))
    plt.imshow(
        data_array,
        extent=(
            min(conductances),
            max(conductances),
            min(input_rates),
            max(input_rates),
        ),
        aspect="auto",
        origin="lower",
        cmap="Grays_r",
    )
    plt.colorbar(label="Firing Rate (spikes/s)")
    plt.xlabel("Mean Synaptic Conductance (nS)")
    plt.ylabel("Input Rate (Hz)")
    plt.xticks(
        0.0 + jnp.linspace(conductances[0], conductances[-1], len(conductances)),
        labels=[f"{g * 1e9:.1f}" for g in conductances],
    )
    plt.yticks(
        0.0 + jnp.linspace(input_rates[0], input_rates[-1], len(input_rates)),
        labels=[f"{ir:.0f}" for ir in input_rates],
    )
    plt.title("Neuron Tuning Curve")
    plt.show()


if __name__ == "__main__":
    main()
