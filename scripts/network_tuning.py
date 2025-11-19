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
from adaptive_SNN.visualization import plot_network_stats


def main():
    t0 = 0
    t1 = 2
    dt = 1e-4
    key = jr.PRNGKey(1)

    N_neurons = 200
    N_inputs = 100

    i_w = 6.0
    r_w = 7.0

    input_weights = [i_w * 1 / (jnp.log(N_inputs))]
    rec_weights = [r_w * 1 / (jnp.log(N_neurons * LIFNetwork.connection_prob))]

    balance = 10.0
    input_firing_rate = 20

    for input_weight in input_weights:
        for rec_weight in rec_weights:
            # Set up models
            neuron_model = LIFNetwork(
                N_neurons=N_neurons,
                N_inputs=N_inputs,
                dt=dt,
                fully_connected_input=True,
                input_weight=input_weight,
                rec_weight=rec_weight,
                key=key,
            )

            D = 0.0

            noise_model = OUP(
                tau=neuron_model.tau_E, noise_std=0 * D, mean=0.0, dim=N_neurons
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
                "get_input_spikes": lambda t, x, args: jr.bernoulli(
                    jr.PRNGKey(jnp.int32(jnp.round(t / dt))),
                    input_firing_rate * dt,
                    shape=(N_neurons, N_inputs),
                ),
            }

            def save_fn(y: NoisyNetworkState):
                return LIFState(
                    V=None,
                    S=y.network_state.S,
                    G=None,
                    W=None,
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
                dt,
                init_state,
                save_every_n_steps=1,
                args=args,
                save_fn=save_fn,
            )

            print("Simulation complete. Generating plots...")
            plot_network_stats(
                sol,
                model,
                save_path=f"../figures/network_tuning_N{N_neurons}_iw{i_w}_rw{r_w}_balance{balance}.png",
            )


if __name__ == "__main__":
    main()
