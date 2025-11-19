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
from adaptive_SNN.utils.analytics import compute_required_input_weight
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import plot_noise_STA


def main():
    noise_levels = [0.05, 0.1, 0.5, 2.0]

    sols = []
    for noise_level in noise_levels:
        t0 = 0
        t1 = 100
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
            fraction_excitatory_input=0.5,
            key=key,
            dt=dt0,
        )

        key, _ = jr.split(key)
        noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)
        model = NoisyNetwork(
            neuron_model=neuron_model,
            noise_model=noise_model,
        )

        # Run simulation
        solver = dfx.EulerHeun()
        init_state = model.initial

        args = {
            "get_input_spikes": lambda t, x, args: jr.poisson(
                jr.PRNGKey(jnp.int32(jnp.round(t / dt0))),
                rates * dt0,
                shape=(N_neurons, N_inputs),
            ),
            "get_desired_balance": lambda t, x, args: jnp.array([5.0]),
            "noise_scale_hyperparam": noise_level,
        }

        def save_fn(t, state: NoisyNetworkState, args):
            return save_part_of_state(state, S=True, noise_state=True, V=True)

        sol = simulate_noisy_SNN(
            model,
            solver,
            t0,
            t1,
            dt0,
            init_state,
            save_at=dfx.SaveAt(t0=True, t1=True, steps=True, fn=save_fn),
            args=args,
        )

        sols.append(sol)

    plot_noise_STA(sols, model, noise_levels=noise_levels)


if __name__ == "__main__":
    main()
