import time

import diffrax as dfx
import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability
import jax.random as jr
import numpy as np
from diffrax import SaveAt
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from adaptive_SNN.models import (
    LIFNetwork,
    NeuralNoiseOUP,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.metrics import compute_charge_ratio, compute_CV_ISI
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    t0 = 0
    t1 = 25
    dt0 = 1e-4
    key = jr.PRNGKey(1)

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    min_noise_std = 0.0
    noise_level = 0.0
    N_neurons = 1
    N_inputs = 2

    E_weights = jnp.arange(0.5, 2.6, 0.1)
    I_weights = jnp.arange(5, 26, 1)

    n_sim = len(E_weights) * len(I_weights)
    count = 0
    recompute_results = False
    if recompute_results:
        CV_ISIs = []
        firing_rates = []
        charge_ratios = []
        mean_voltages = []

        for E_weight in E_weights:
            for I_weight in I_weights:
                initial_weight_matrix = jnp.array([[-jnp.inf, E_weight, I_weight]])
                input_types = jnp.array([1, 0])  # 1 for excitatory, 0 for inhibitory
                # Set up models
                neuron_model = LIFNetwork(
                    N_neurons=N_neurons,
                    N_inputs=N_inputs,
                    fully_connected_input=True,
                    fraction_excitatory_input=0.5,
                    input_types=input_types,
                    initial_weight_matrix=initial_weight_matrix,
                    weight_std=0.0,
                    key=key,
                    dt=dt0,
                )

                key, _ = jr.split(key)
                noise_model = NeuralNoiseOUP(tau=neuron_model.tau_E, dim=N_neurons)
                model = NoisyNetwork(
                    neuron_model=neuron_model,
                    noise_model=noise_model,
                    min_noise_std=min_noise_std,
                )

                # Run simulation
                solver = dfx.EulerHeun()
                init_state = model.initial

                def get_spikes(t, x, args):
                    return jr.poisson(
                        jr.fold_in(key, jnp.rint(t / dt0)),
                        rates * dt0,
                        shape=(N_neurons, N_inputs),
                    )

                args = {
                    "get_input_spikes": get_spikes,
                    "get_desired_balance": lambda t, x, args: jnp.array([0.0]),
                    "noise_scale_hyperparam": noise_level,
                }

                def save_fn(t, state, args):
                    return save_part_of_state(
                        state,
                        V=True,
                        G=True,
                        W=True,
                        S=True,
                        noise_state=True,
                        # mean_E_conductance=True,
                        # var_E_conductance=True,
                    )

                start = time.time()
                sol = simulate_noisy_SNN(
                    model,
                    solver,
                    t0,
                    t1,
                    dt0,
                    init_state,
                    save_at=SaveAt(
                        ts=jnp.linspace(5, t1, int((t1 - 5) // dt0)), fn=save_fn
                    ),
                    args=args,
                )
                end = time.time()

                # plot_simulate_SNN_results(sol, model, split_noise=True)

                state: NoisyNetworkState = sol.ys

                charge_ratio = compute_charge_ratio(sol.ts, state, model)
                cv_isi = compute_CV_ISI(state.network_state.S, sol.ts)
                firing_rate = jnp.sum(state.network_state.S, axis=0) / (t1 - t0)
                mean_voltage = jnp.mean(state.network_state.V)

                CV_ISIs.append(cv_isi)
                firing_rates.append(firing_rate)
                charge_ratios.append(charge_ratio)
                mean_voltages.append(mean_voltage)

                print(
                    f"Finished simulation {count + 1}/{n_sim} in {end - start:.2f} seconds",
                    end="\r",
                    flush=True,
                )
                count += 1

        jnp.savez(
            "results/noisy_neuron_experiment.npz",
            CV_ISIs=jnp.array(CV_ISIs),
            firing_rates=jnp.array(firing_rates),
            charge_ratios=jnp.array(charge_ratios),
            mean_voltages=jnp.array(mean_voltages),
        )
    else:
        loaded_results = jnp.load(
            "results/noisy_neuron_experiment.npz", allow_pickle=True
        )
        CV_ISIs = loaded_results["CV_ISIs"]
        firing_rates = loaded_results["firing_rates"]
        charge_ratios = loaded_results["charge_ratios"]
        mean_voltages = loaded_results["mean_voltages"]

    # Make heatmaps of the results
    def to_plot_matrix(values):
        # Data is collected with E-weight outer loop and I-weight inner loop.
        # Reshape to [E, I] and transpose so plot axes are x=E and y=I.
        return jnp.array(values).reshape(len(E_weights), len(I_weights)).T

    def add_pixel_outline(ax, mask, color="white", linewidth=1.5):
        mask_np = np.asarray(mask, dtype=bool)
        n_rows, n_cols = mask_np.shape
        segments = []

        for row in range(n_rows):
            for col in range(n_cols):
                if not mask_np[row, col]:
                    continue

                x0, x1 = col - 0.5, col + 0.5
                y0, y1 = row - 0.5, row + 0.5

                if row == 0 or not mask_np[row - 1, col]:
                    segments.append([(x0, y0), (x1, y0)])
                if row == n_rows - 1 or not mask_np[row + 1, col]:
                    segments.append([(x0, y1), (x1, y1)])
                if col == 0 or not mask_np[row, col - 1]:
                    segments.append([(x0, y0), (x0, y1)])
                if col == n_cols - 1 or not mask_np[row, col + 1]:
                    segments.append([(x1, y0), (x1, y1)])

        if segments:
            ax.add_collection(
                LineCollection(segments, colors=color, linewidths=linewidth)
            )

    metric_matrices = [
        to_plot_matrix(CV_ISIs),
        to_plot_matrix(firing_rates),
        to_plot_matrix(charge_ratios),
        to_plot_matrix(mean_voltages),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    axs = axs.flatten()
    images = [
        ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest")
        for ax, matrix in zip(axs, metric_matrices)
    ]

    titles = [
        "CV of ISI",
        "Firing Rate",
        "I/E Charge Ratio",
        "Mean Voltage",
    ]

    target_ranges = [
        (0.9, 1.1),
        (0.0, 5.0),
        (0.9, 1.1),
        (-60e-3, jnp.inf),
    ]

    x_tick_labels = [f"{float(w):.1f}" for w in E_weights][::2]
    y_tick_labels = [f"{int(w)}" for w in I_weights][::2]

    for ax, title in zip(axs, titles):
        ax.set_xticks(jnp.arange(len(x_tick_labels)) * 2)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(jnp.arange(len(y_tick_labels)) * 2)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("E Weight")
        ax.set_ylabel("I Weight")
        ax.set_title(rf"\textbf{{{title}}}")

    target_masks = []
    for matrix, target in zip(metric_matrices, target_ranges):
        if target is None:
            target_masks.append(None)
            continue

        lower, upper = target
        mask = jnp.isfinite(matrix) & (matrix >= lower) & (matrix <= upper)
        target_masks.append(mask)

    valid_target_masks = [mask for mask in target_masks if mask is not None]
    common_target_mask = jnp.logical_and.reduce(jnp.stack(valid_target_masks), axis=0)

    for ax, mask in zip(axs, target_masks):
        if mask is not None and bool(jnp.any(mask)):
            add_pixel_outline(ax, mask, color="lightgray", linewidth=1.5)

        if bool(jnp.any(common_target_mask)):
            add_pixel_outline(ax, common_target_mask, color="white", linewidth=2.0)

    # Add colorbars
    for ax, img in zip(axs, images):
        fig.colorbar(img, ax=ax)
        ax.label_outer()

    plt.show()


if __name__ == "__main__":
    main()
