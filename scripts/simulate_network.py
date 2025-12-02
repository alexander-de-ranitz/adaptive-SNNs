import diffrax as dfx
import equinox as eqx
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
from adaptive_SNN.visualization import plot_network_stats, plot_simulate_SNN_results


def main():
    t0 = 0
    t1 = 1.0
    dt = 1e-4
    key = jr.PRNGKey(2)

    N_neurons = 500
    N_inputs = 100

    # Define input parameters
    input_firing_rate = 20

    input_weight = 7.0
    rec_weight = 7.0

    balance = 2.5

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=input_weight,
        rec_weight=rec_weight,
        fraction_excitatory_input=0.8,
        fraction_excitatory_recurrent=0.8,
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
        "noise_scale_hyperparam": 0.0,
    }

    def save_fn(t, y, args):
        state: NoisyNetworkState = save_part_of_state(y, S=True, W=True, V=True, G=True)
        state = eqx.tree_at(
            lambda state: state.network_state.G, state, replace_fn=lambda G: G[0, :]
        )
        state = eqx.tree_at(
            lambda state: state.network_state.W, state, replace_fn=lambda W: W[0, :]
        )
        return state

    print("Running simulation...")
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt,
        init_state,
        save_at=dfx.SaveAt(t0=True, t1=True, steps=True, fn=save_fn),
        args=args,
    )

    print("Simulation complete. Generating plots...")
    plot_network_stats(
        sol,
        model,
    )

    plot_simulate_SNN_results(
        sol, model, split_noise=True, split_recurrent=True, neurons_to_plot=[0]
    )


if __name__ == "__main__":
    main()
