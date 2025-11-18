import time

import diffrax as dfx
import equinox as eqx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import (
    OUP,
    Agent,
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
    SystemState,
)
from adaptive_SNN.models.environment import SpikeRateEnvironment
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.analytics import (
    compute_expected_synaptic_std,
    compute_oup_diffusion_coefficient,
    compute_required_input_weight,
)
from adaptive_SNN.visualization import plot_learning_results


def main():
    t0 = 0
    t1 = 100
    dt = 1e-4
    key = jr.PRNGKey(0)
    key, _ = jr.split(key)
    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    input_weight = compute_required_input_weight(
        target_mean_g_syn=10e-9,
        N_inputs=1.0,
        tau=LIFNetwork.tau_E,
        input_rate=exc_rate,
        synaptic_increment=LIFNetwork.synaptic_increment,
    )

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=input_weight,
        fraction_excitatory_input=0.5,
        key=key,
    )
    key, _ = jr.split(key)

    syn_std = compute_expected_synaptic_std(
        N_inputs=1.0,
        input_rate=exc_rate,
        tau=neuron_model.tau_E,
        synaptic_increment=neuron_model.synaptic_increment,
        input_weight=input_weight,
    )

    D = compute_oup_diffusion_coefficient(
        target_std=syn_std * 2, tau=neuron_model.tau_E
    )
    noise_model = OUP(tau=neuron_model.tau_E, noise_scale=D, dim=N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
    )

    agent = Agent(
        neuron_model=network,
        reward_model=RewardModel(reward_rate=10),
    )

    model = AgentEnvSystem(
        agent=agent,
        environment=SpikeRateEnvironment(
            dim=1,
        ),
    )
    solver = dfx.EulerHeun()
    init_state = model.initial

    # Define args
    # Create a base seed that's unique for each loop iteration
    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(t < 1, 0.0, 0.025),
        "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
            agent_state.noisy_network.network_state.S[0]
        ),
        "reward_fn": lambda t, environment_state, args: -jnp.abs(
            environment_state[0] - target_state
        ),
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey(jnp.int32(jnp.round(t / dt))),
            rates * dt,
            shape=(N_neurons, N_inputs),
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([4.0]),
    }

    initial_weight_factors = jnp.array([1.5, 2.5, 3.5])

    def get_weights(state: SystemState):
        return state.agent_state.noisy_network.network_state.W

    sols = []
    for i in range(len(initial_weight_factors)):
        init = eqx.tree_at(
            get_weights,
            init_state,
            get_weights(init_state) * initial_weight_factors[i],
        )
        key = jr.PRNGKey(i * 12345)
        print(f"Running simulation {i + 1}/3 ...")
        start = time.time()

        def save_fn(y: SystemState):
            return eqx.tree_at(
                lambda s: (
                    s.agent_state.noisy_network.network_state.spike_buffer,
                    s.agent_state.noisy_network.network_state.G,
                ),
                y,
                replace_fn=lambda x: None,
            )

        sol = simulate_noisy_SNN(
            model,
            solver,
            t0,
            t1,
            dt,
            init,
            save_every_n_steps=1000,
            args=args,
            save_fn=save_fn,
            key=key,
        )
        end = time.time()
        print(f"Simulation completed in {end - start:.2f} seconds.")

        sols.append(sol)

    print("Plotting results...")
    plot_learning_results(
        sols,
        model,
        args,
        target_state,
        save_path="../figures/learning_output_rate_batched_new.png",
    )


if __name__ == "__main__":
    main()
