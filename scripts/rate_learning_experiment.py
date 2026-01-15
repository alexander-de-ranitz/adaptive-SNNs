import time

import diffrax as dfx
import equinox as eqx
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import (
    OUP,
    Agent,
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
    SystemState,
)
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    t0 = 0
    t1 = 200
    dt = 1e-4

    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    iterations = 5
    noise_level = 0.15
    lr = 6.5
    initial_weight_factors = jnp.array([0.5, 1.0])

    network_key = jr.PRNGKey(0)

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=5.0,
        input_types=jnp.array([1, 0]),
        key=network_key,
    )

    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=0 * 5e-9,
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

    def get_weights(state: SystemState):
        return state.agent_state.noisy_network.network_state.W

    data = {}
    key = jr.PRNGKey(1)
    for i in range(iterations):
        for j in range(len(initial_weight_factors)):
            key, spike_key, simulation_key = jr.split(key, 3)

            args = {
                "get_learning_rate": lambda t, x, args: jnp.where(t < 5, 0.0, lr),
                "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
                    agent_state.noisy_network.network_state.S[0]
                ),
                "reward_fn": lambda t, environment_state, args: -jnp.abs(
                    environment_state[0] - target_state
                ),
                "get_input_spikes": lambda t, x, args: jr.poisson(
                    jr.fold_in(spike_key, jnp.round(t / dt).astype(int)),
                    rates * dt,
                    shape=(N_neurons, N_inputs),
                ),
                "get_desired_balance": lambda t, x, args: jnp.array([0.86]),
                "noise_scale_hyperparam": noise_level,
            }

            init = eqx.tree_at(
                get_weights,
                init_state,
                get_weights(init_state) * initial_weight_factors[j],
            )

            start = time.time()

            def save_fn(t, y: SystemState, args):
                return save_part_of_state(y, W=True, environment_state=True)

            sol = simulate_noisy_SNN(
                model,
                solver,
                t0,
                t1,
                dt,
                init,
                save_at=SaveAt(fn=save_fn, ts=jnp.linspace(t0, t1, 500)),
                args=args,
                key=simulation_key,
            )

            end = time.time()
            print(
                f"Simulation {i * len(initial_weight_factors) + j + 1}/{iterations * len(initial_weight_factors)} completed in {end - start:.2f} seconds.",
                end="\r",
            )

            E_weights = get_weights(sol.ys)[:, :, 1]
            if j not in data:
                data[j] = (E_weights, sol.ys.environment_state)
            else:
                data[j] = (
                    jnp.hstack((data[j][0], E_weights)),
                    jnp.hstack((data[j][1], sol.ys.environment_state)),
                )

    ts = sol.ts
    for j in range(len(initial_weight_factors)):
        plt.subplot(2, 1, 1)
        plt.plot(ts, data[j][0], c="k", alpha=0.1)
        plt.subplot(2, 1, 2)
        plt.plot(ts, data[j][1], c="k", alpha=0.1)

        plt.subplot(2, 1, 1)
        mean_weights = jnp.mean(data[j][0], axis=1).squeeze()
        std_weights = jnp.std(data[j][0], axis=1).squeeze()
        plt.plot(ts, mean_weights, label=f"Init factor {initial_weight_factors[j]}")
        plt.fill_between(
            ts,
            mean_weights - std_weights,
            mean_weights + std_weights,
            alpha=0.3,
        )

        plt.subplot(2, 1, 2)
        mean_env = jnp.mean(data[j][1], axis=1).squeeze()
        std_env = jnp.std(data[j][1], axis=1).squeeze()

        plt.plot(ts, mean_env, label=f"Init factor {initial_weight_factors[j]}")
        plt.fill_between(
            ts,
            mean_env - std_env,
            mean_env + std_env,
            alpha=0.3,
        )

    plt.subplot(2, 1, 1)
    plt.title("Synaptic weight evolution")
    plt.xlabel("Time step")
    plt.ylabel("Synaptic weight")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Environment state evolution")
    plt.xlabel("Time step")
    plt.ylabel("Environment state")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
