import time

import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import (
    OUP,
    LIFNetwork,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import plot_network_stats


def main():
    t0 = 0
    t1 = 1
    dt = 1e-4
    key = jr.PRNGKey(1)

    N_neurons = 500
    N_inputs = 100

    input_weights = jnp.linspace(5.0, 20.0, 5)
    rec_weights = jnp.linspace(5.0, 20.0, 5)
    balances = [4.0, 6.0, 8.0]

    input_firing_rate = 20

    for balance in balances:
        for input_weight in input_weights:
            for rec_weight in rec_weights:
                print(
                    f"Running simulation with input weight {input_weight} and recurrent weight {rec_weight}"
                )
                # Set up models
                neuron_model = LIFNetwork(
                    N_neurons=N_neurons,
                    N_inputs=N_inputs,
                    dt=dt,
                    fully_connected_input=True,
                    fraction_excitatory_input=0.8,
                    fraction_excitatory_recurrent=0.8,
                    input_weight=input_weight,
                    rec_weight=rec_weight,
                    key=key,
                )

                noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

                model = NoisyNetwork(
                    neuron_model=neuron_model,
                    noise_model=noise_model,
                )

                solver = dfx.EulerHeun()
                init_state = model.initial

                # Define args
                args = {
                    "get_desired_balance": lambda t, x, args: jnp.array([balance]),
                    "get_input_spikes": lambda t, x, args: jr.bernoulli(
                        jr.PRNGKey(jnp.round(t / dt).astype(int)),
                        input_firing_rate * dt,
                        shape=(N_neurons, N_inputs),
                    ),
                    "noise_scale_hyperparam": 0.1,
                }

                def save_fn(t, y: NoisyNetworkState, args):
                    return save_part_of_state(y, S=True)

                start = time.time()
                sol = simulate_noisy_SNN(
                    model,
                    solver,
                    t0,
                    t1,
                    dt,
                    init_state,
                    save_at=dfx.SaveAt(
                        t0=False, ts=jnp.arange(0.5, t1, dt), fn=save_fn
                    ),
                    args=args,
                )
                end = time.time()
                print(f"Simulation completed in: {end - start:.2f} seconds")

                plot_network_stats(
                    sol,
                    model,
                    save_path=f"../figures/network_tuning_N_{N_neurons}_iw_{input_weight}_rw_{rec_weight}_balance_{balance}.png",
                )

                jnp.savez(
                    f"data/network_tuning/N_{N_neurons}_iw_{input_weight}_rw_{rec_weight}_balance_{balance}.npz",
                    ts=sol.ts,
                    S=sol.ys.network_state.S,
                )


if __name__ == "__main__":
    main()
