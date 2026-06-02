import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = "results/pendulum_20260530_175744/results/pendulum_seed0.npz"


def load_pendulum_results(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data["ts"], data["ys"]


def main():
    ts, ys = load_pendulum_results(FILE_PATH)
    ts = ts.item()
    env_states, rewards, predicted_rewards, filtered_spike_trains = ys
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(ts, env_states[:, 0], label="Angle")
    plt.plot(ts, env_states[:, 1], label="Angular Velocity")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(ts, rewards.squeeze(), label="Reward")
    plt.plot(ts, predicted_rewards.squeeze(), label="Predicted Reward", linestyle="--")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.imshow(
        filtered_spike_trains.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[ts[0], ts[-1], 0, filtered_spike_trains.shape[1]],
    )
    plt.colorbar(label="Filtered Spike Train")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
