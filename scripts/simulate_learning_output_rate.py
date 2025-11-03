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
from adaptive_SNN.visualization.plotting import (
    plot_learning_results_multiple,
)


def main():
    t0 = 0
    t1 = 100
    dt0 = 1e-4
    key = jr.PRNGKey(0)
    key, _ = jr.split(key)
    N_neurons = 1
    N_inputs = 2

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt0,
        input_neuron_types=jnp.array([1.0, 0.0]),
        fully_connected_input=True,
        input_weight=1.0,
        key=key,
    )
    key, _ = jr.split(key)

    noise_E_model = OUP(tau=250.0, noise_scale=1e-7, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(tau=250.0, noise_scale=1e-7, mean=0.0, dim=N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
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

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rate = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    # Define args
    # Create a base seed that's unique for each loop iteration
    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(t < 10, 0.0, 0.01),
        "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
            agent_state.noisy_network.network_state.S[0]
        ),
        "reward_fn": lambda t, environment_state, args: -jnp.abs(
            environment_state[0] - target_state
        ),
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey((t / dt0).astype(int)), rate * dt0, shape=(N_inputs,)
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([4.0]),
    }

    initial_weight_factors = jnp.array([1.5, 2.5, 3.5])

    def get_weights(state: SystemState):
        return state.agent_state.noisy_network.network_state.W

    sols = []
    for i in range(3):
        init = eqx.tree_at(
            get_weights,
            init_state,
            get_weights(init_state) * initial_weight_factors[i],
        )
        key = jr.PRNGKey(i * 12345)
        print(f"Running simulation {i + 1}/3 ...")
        start = time.time()
        sol = simulate_noisy_SNN(
            model,
            solver,
            t0,
            t1,
            dt0,
            init,
            save_every_n_steps=1000,
            args=args,
            key=key,
        )
        end = time.time()
        print(f"Simulation completed in {end - start:.2f} seconds.")

        sols.append(sol)
        # print("Plotting results...")
        # plot_learning_results(sol, model, t0, t1, dt0, args, save_path=f"../figures/learning_output_rate_{i}" + ".png")

    print("Plotting all results together...")
    plot_learning_results_multiple(
        sols,
        model,
        t0,
        t1,
        dt0,
        args,
        target_state,
        save_path="../figures/learning_output_rate_batched.png",
    )


if __name__ == "__main__":
    main()
