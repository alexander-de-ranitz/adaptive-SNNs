import os

import matplotlib.pyplot as plt
import numpy as np

DIR = "results/network_20260602_174335/results/"


def load_results(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data["ts"], data["ys"]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    i = 0
    axs = axs.flatten()
    for file in os.listdir(DIR):
        if file.endswith(".npz"):
            ts, ys = load_results(os.path.join(DIR, file))
            ts = ts.item()
            W, G, charge_in, charge_out, env_states = ys
            # axs[i].plot(ts, charge_in[:, 0], label="Charge In")
            # axs[i].plot(ts, charge_out[:, 0], label="Charge Out")
            # axs[i].set_title("Charge In and Out")
            # axs[i].legend()
            # axs[i].set_xlabel("Time (s)")
            # i += 1
            # firing_rates = np.mean(env_states, axis=0)
            # print(firing_rates.shape)
            # axs[i].hist(firing_rates, bins=np.arange(0, 50))
            # axs[i].set_xlabel("Firing Rate")
            # axs[i].set_ylabel("Frequency")

            axs[i].plot(ts, charge_in + charge_out, label="Charge In + Out")
            axs[i].legend()
            # axs[i].plot(ts, W[:, 0], label="W")
            # axs[i].plot(ts, np.mean(env_states, axis=1))
            axs[i].set_title(f"Distribution of Firing Rates ({file})")
            i += 1
    plt.show()


if __name__ == "__main__":
    main()
