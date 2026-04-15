from matplotlib import pyplot as plt

from adaptive_SNN.utils.runner import run_simulation
from scripts.simulation_configs.single_synapse_learning import (
    create_default_config_single_synapse_task,
)


def main():
    config = create_default_config_single_synapse_task()
    config.t1 = 3

    config.initial_weight_matrix = config.initial_weight_matrix.at[1, :].set(
        config.initial_weight_matrix[0, :]
    )
    config.initial_weight_matrix = config.initial_weight_matrix.at[2, :].set(
        config.initial_weight_matrix[0, :]
    )

    sol, model = run_simulation(config, save_results=False)

    state = sol.ys
    V = state.agent_state.noisy_network.network_state.V[:, 0]
    V2 = state.agent_state.noisy_network.network_state.V[:, 1]
    V3 = state.agent_state.noisy_network.network_state.V[:, 2]
    noise = state.agent_state.noisy_network.noise_state[:, :]

    plt.plot(sol.ts, V, label="Voltage")
    plt.plot(sol.ts, V2, label="Voltage 2")
    plt.plot(sol.ts, V3, label="Voltage 3")
    plt.legend()
    plt.show()

    for i in range(noise.shape[1]):
        plt.plot(sol.ts, noise[:, i], label=f"Noise {i}", alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
