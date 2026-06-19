import time

import diffrax as dfx
from jax import numpy as jnp
from matplotlib import pyplot as plt

from adaptive_SNN.models import SystemState
from adaptive_SNN.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    start = time.time()
    config = create_pendulum_AC_config(N_neurons=2)
    config.lr = 0.0
    config.t1 = 10.0
    config.save_at = dfx.SaveAt(
        ts=jnp.linspace(config.t0, config.t1, 1000),
        fn=lambda t, state, args: save_part_of_state(
            state,
            environment_state=True,
            agent_output=True,
            reward_signal=True,
            value=True,
            RPE=True,
            input_features=True,
            weights=True,
        ),
    )
    config.save_file = "results/pendulum_spiking.npz"
    sol, model = run_simulation(config, save_results=False, return_final_state=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds")
    state: SystemState = sol.ys[0]
    final_state: SystemState = sol.ys[1]
    ts = sol.ts
    reward = state.reward_signal
    predicted_reward = state.agent_state.reward_predictor_state.value
    RPE = state.agent_state.RPE
    input_features = state.agent_state.reward_predictor_state.input_features
    W_critic = state.agent_state.reward_predictor_state.weights

    n_plots = 5
    ax1 = plt.subplot(n_plots, 1, 1)
    plt.plot(ts, reward, label="Reward Signal")
    # plt.plot(ts, predicted_reward, label="Predicted Reward")
    plt.plot(ts, RPE, label="Reward Prediction Error (RPE)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(n_plots, 1, 2, sharex=ax1)
    plt.plot(ts, state.environment_state[:, 0], label="Pendulum Angle")
    plt.plot(ts, state.environment_state[:, 1], label="Pendulum Angular Velocity")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(n_plots, 1, 3, sharex=ax1)
    plt.plot(ts, predicted_reward, label="Predicted Reward")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(n_plots, 1, 4, sharex=ax1)
    plt.imshow(
        input_features.T,
        aspect="auto",
        origin="lower",
        extent=[ts[0], ts[-1], 0, input_features.shape[1]],
        interpolation="none",
    )
    plt.xlabel("Time (s)")
    plt.subplot(n_plots, 1, 5, sharex=ax1)
    plt.imshow(
        W_critic.T,
        aspect="auto",
        origin="lower",
        extent=[ts[0], ts[-1], 0, W_critic.shape[1]],
        interpolation="none",
    )
    plt.xlabel("Time (s)")
    plt.show()

    plt.imshow(
        final_state.agent_state.reward_predictor_state.weights[:-1].reshape(16, 16).T,
        aspect="auto",
        origin="lower",
    )
    plt.show()


if __name__ == "__main__":
    main()
