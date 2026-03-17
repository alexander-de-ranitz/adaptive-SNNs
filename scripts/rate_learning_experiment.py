import time

import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from helpers import SimulationConfig, run_simulation
from jax import numpy as jnp

from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import (
    plot_learning_detailed,
)


def main():
    start = time.time()
    t0 = 0
    t1 = 30
    dt = 1e-4
    lr = 0.5
    noise_level = 0.1
    N_neurons = 1
    N_inputs = 500
    model_cls = GatedLIFNetwork  # Change to GatedLIFNetwork to test the gated model
    network_output_fn = lambda t, agent_state, args: jnp.squeeze(
        agent_state.noisy_network.network_state.S[0]
    )

    spike_key = jr.PRNGKey(101)
    input_spike_fn = lambda t, x, args: jr.poisson(
        jr.fold_in(spike_key, jnp.rint(t / dt)),
        10 * dt,
        shape=(N_neurons, N_inputs),
    )

    target_state = 10
    reward_fn = lambda t, x, args: jnp.squeeze(
        -jnp.square(x.environment_state - target_state)
    )
    cfg = SimulationConfig(
        model=model_cls,
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        fraction_excitatory_input=0.8,
        balance=1.75,
        input_types=None,
        t0=t0,
        t1=t1,
        dt=dt,
        initial_weight=1,
        weight_std=0.2,
        lr=lr,
        noise_level=noise_level,
        min_noise_std=1e-10,
        warmup_time=20,
        reward_rate=0.1,
        key_seed=0,
        save_at=SaveAt(
            ts=jnp.linspace(20, t1, 5000),
            fn=lambda t, x, args: save_part_of_state(
                x,
                environment_state=True,
                W=True,
                G=True,
                reward_signal=True,
                predicted_reward=True,
                eligibility=True,
                V=True,
                noise_state=True,
                S=True,
                mean_E_conductance=True,
                var_E_conductance=True,
            ),
        ),
        save_file=f"results/rate_learning_experiment/simulation_results_{'gated' if model_cls == GatedLIFNetwork else 'eligibility'}_lr{lr:.2f}_nl{noise_level:.2f}.npz",
        network_output_fn=network_output_fn,
        input_spike_fn=input_spike_fn,
        reward_fn=reward_fn,
        save_results=True,
    )

    sol, model = run_simulation(cfg)
    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")
    # plot_learning_results(sol)
    plot_learning_detailed(sol, model)


if __name__ == "__main__":
    main()
