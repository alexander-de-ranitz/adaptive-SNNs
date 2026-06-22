import time

import diffrax as dfx
from matplotlib import pyplot as plt

from adaptive_SNN.models import SystemState
from adaptive_SNN.simulation_configs.pendulum_config import create_pendulum_config
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    start = time.time()
    config = create_pendulum_config()
    config.t1 = 0.01
    config.save_at = dfx.SaveAt(
        steps=True,
        fn=lambda t, state, args: save_part_of_state(
            state,
            environment_state=True,
            agent_output=True,
            reward_signal=True,
            value=True,
            RPE=True,
        ),
    )
    config.save_file = "results/pendulum_spiking.npz"
    sol, model = run_simulation(config, save_results=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds")
    state: SystemState = sol.ys
    ts = sol.ts
    reward = state.reward_signal
    predicted_reward = state.agent_state.reward_predictor_state.value
    RPE = state.agent_state.RPE

    plt.subplot(3, 1, 1)
    plt.plot(ts, reward, label="Reward Signal")
    plt.plot(ts, predicted_reward, label="Predicted Reward")
    plt.plot(ts, RPE, label="Reward Prediction Error (RPE)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(ts, state.environment_state[:, 0], label="Pendulum Angle")
    plt.plot(ts, state.environment_state[:, 1], label="Pendulum Angular Velocity")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(ts, state.agent_output, label="Agent Output")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()
    # S = state.agent_state.network_state.S

    # plot_spike_raster(ts, S)

    # plt.subplot(1, 3, 1)
    # firing_rates = jnp.mean(S, axis=0) / (ts[1] - ts[0])
    # plt.hist(firing_rates, bins=50)
    # plt.xlabel("Firing Rate (Hz)")

    # plt.subplot(1, 3, 2)
    # synchrony = compute_synchrony(S)
    # print(f"Synchrony: {synchrony:.4f}")
    # CV_ISI = compute_CV_ISI(S, ts)
    # plt.hist(CV_ISI)
    # plt.xlabel("CV of ISI")

    # plt.subplot(1, 3, 3)
    # plt.scatter(firing_rates, CV_ISI, alpha=0.5)
    # plt.xlabel("Firing Rate (Hz)")
    # plt.ylabel("CV of ISI")
    # plt.show()


if __name__ == "__main__":
    main()
