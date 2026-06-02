import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from adaptive_SNN.utils.metrics import compute_CV_ISI, compute_synchrony
from adaptive_SNN.visualization.api.plotting import plot_spike_raster

FILE_PATH = "results/pendulum_spikes_20260601_191206/results/pendulum_seed0.npz"


def load_pendulum_results(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data["ts"], data["ys"]


def main():
    ts, ys = load_pendulum_results(FILE_PATH)
    ts = ts.item()
    S = ys.item()

    plot_spike_raster(ts, S)

    plt.subplot(1, 3, 1)
    firing_rates = jnp.mean(S, axis=0) / (ts[1] - ts[0])
    plt.hist(firing_rates, bins=50)
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Number of Neurons")

    plt.subplot(1, 3, 2)
    synchrony = compute_synchrony(S)
    print(f"Synchrony: {synchrony:.4f}")
    CV_ISI = compute_CV_ISI(S, ts)
    plt.hist(CV_ISI)
    plt.ylabel("Number of Neurons")
    plt.xlabel("CV of ISI")

    plt.subplot(1, 3, 3)
    plt.scatter(firing_rates, CV_ISI, alpha=0.5)
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("CV of ISI")
    plt.show()


if __name__ == "__main__":
    main()
