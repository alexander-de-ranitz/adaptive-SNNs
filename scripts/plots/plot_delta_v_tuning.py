import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.transforms import ScaledTranslation

from adaptive_SNN.models.networks import GatedLIFNetwork


@dataclass
class RunFile:
    path: Path
    dv: float
    method: str


DATA_DIR = Path("results/delta_v_tuning_20260418_003224/results")
OUTPUT_PATH = Path("figures/delta_v_tuning")


def parse_run_file(file: Path) -> RunFile:
    parts = file.stem.split("_")
    dv = float(parts[1])
    method = parts[-1]
    assert method in {"gated", "default"}, method
    return RunFile(path=file, dv=dv, method=method)


def load_run_arrays(file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(file, allow_pickle=True) as data:
        agent_state = data["sol"].item().ys.agent_state
    eligibility = np.asarray(
        agent_state.network_state.features.eligibility[:, 1, -1]
    ).squeeze()
    reward_noise = np.asarray(agent_state.reward_noise).squeeze()
    rpe = np.asarray(agent_state.RPE).squeeze()
    return eligibility, reward_noise, rpe


def get_grouped_files() -> list[tuple[float, str, list[Path]]]:
    grouped: dict[float, list[RunFile]] = {}
    for name in os.listdir(DATA_DIR):
        if not name.endswith(".npz"):
            continue
        run = parse_run_file(DATA_DIR / name)
        if run.dv not in grouped:
            grouped[run.dv] = []
        grouped[run.dv].append(run)

    result: list[tuple[float, str, list[Path]]] = []
    for dv in sorted(grouped.keys()):
        runs = grouped[dv]
        methods = {run.method for run in runs}
        assert len(methods) == 1, f"Mixed methods for dv={dv}: {methods}"
        result.append((dv, runs[0].method, [run.path for run in runs]))
    return result


def compute_summary_stats(files: list[Path]) -> dict[str, float]:
    all_dW_task = []
    all_dW_noise = []
    print(f"Computing stats for {len(files)} files...")
    N_points = None
    for file in files:
        eligibility, reward_noise, rpe = load_run_arrays(file)
        N_points = eligibility.size if N_points is None else N_points
        assert eligibility.size == N_points, (
            f"Expected {N_points} points, got {eligibility.size} in file {file}"
        )
        dW_task = (eligibility * rpe).ravel()
        dW_noise = (eligibility * reward_noise).ravel()
        all_dW_task.append(dW_task)
        all_dW_noise.append(dW_noise)
    print("N points per file:", N_points)
    all_dW_task = np.vstack(all_dW_task)
    all_dW_noise = np.vstack(all_dW_noise)

    # TODO: hacky way to simulate having fewer, longer runs instead of many short runs
    # this produces more stable estimates. Instead, we should just run longer simulations
    N = 6  # amount of artificial "runs" to generate
    segments_task = [
        all_dW_task[i * N : (i + 1) * N, :] for i in range(all_dW_task.shape[0] // N)
    ]
    segments_noise = [
        all_dW_noise[i * N : (i + 1) * N, :] for i in range(all_dW_noise.shape[0] // N)
    ]
    all_dW_task = np.hstack(segments_task)
    all_dW_noise = np.hstack(segments_noise)

    print(
        f"all_dw shape: {all_dW_task.shape}, all_dW_noise shape: {all_dW_noise.shape}"
    )
    alignment = np.sum(all_dW_task, axis=1) / np.sum(np.abs(all_dW_task), axis=1)
    snr = np.sum(all_dW_task, axis=1) / np.sum(np.abs(all_dW_noise), axis=1)
    print(snr.shape, alignment.shape)
    return {
        "alignment": np.mean(alignment),
        "SNR": np.mean(snr),
        "alignment_std": np.std(alignment),
        "SNR_std": np.std(snr),
    }


def compute_exponent(dv: float) -> float:
    return int(round(-np.log2(dv)))


def plot_figure():
    groups = get_grouped_files()
    groups.sort(
        key=lambda x: (x[1] == "default", x[0])
    )  # Sort by dv, then put gated last
    fig = plt.figure(figsize=(8.0, 4.5))

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    gated_dvs = np.array(
        [dv for dv, method, _ in groups if method == "gated"], dtype=float
    )
    dv_min = float(gated_dvs.min())
    dv_max = float(gated_dvs.max())

    dv_norm = LogNorm(vmin=dv_min, vmax=dv_max)
    dv_cmap = LinearSegmentedColormap.from_list(
        "Greens_r_trimmed",
        plt.cm.Greens_r(np.linspace(0.0, 0.75, 256)),
    )

    for i, (dv, method, files) in enumerate(groups):
        stats = compute_summary_stats(files)
        if method == "gated":
            color = dv_cmap(dv_norm(dv))
            label = (
                rf"$2^{{{-compute_exponent(dv)}}}$" if method == "gated" else "Default"
            )

            # Alignment on first ax
            ax1.bar(label, stats["alignment"], color=color)
            ax1.errorbar(
                label,
                stats["alignment"],
                yerr=stats["alignment_std"],
                color="black",
                capsize=5,
                linestyle="None",
            )  # Add error bars for alignment

            # SNR on second ax
            ax2.bar(label, stats["SNR"], color=color)
            ax2.errorbar(
                label,
                stats["SNR"],
                yerr=stats["SNR_std"],
                color="black",
                capsize=5,
                linestyle="None",
            )  # Add error bars for SNR

            # Gating function on third ax
            network = GatedLIFNetwork(N_neurons=1, dt=1e-4, N_inputs=0)
            voltages = np.linspace(
                -75 * 1e-3, -50 * 1e-3, 5000
            )  # From -80 mV to +20 mV
            gating_values = network.gating_function(voltages, delta_V=dv)
            gating_values = gating_values / gating_values.max()  # Normalize to [0, 1]
            ax3.plot(
                voltages * 1e3, gating_values, label=label, color=color, linewidth=2.5
            )
        else:
            xlim = ax1.get_xlim()
            ax1.hlines(
                stats["alignment"],
                xmin=xlim[0],
                xmax=xlim[1],
                color="black",
                linestyles="dashed",
                label="No gating",
            )
            ax1.set_xlim(xlim)

            xlim2 = ax2.get_xlim()
            ax2.hlines(
                stats["SNR"],
                xmin=xlim2[0],
                xmax=xlim2[1],
                color="black",
                linestyles="dashed",
                label="No gating",
            )
            ax2.set_xlim(xlim2)

            ax3.hlines(
                1.0,
                xmin=-75,
                xmax=-50,
                color="black",
                linestyles="dashed",
                label="No gating",
            )
            ax3.set_xlim(-75, -50)

    sm = plt.cm.ScalarMappable(
        norm=dv_norm,
        cmap=dv_cmap,
    )
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=ax3, label=r"$\Delta V$", pad=0.01)
    colorbar.locator = LogLocator(base=2)
    colorbar.formatter = LogFormatterMathtext(base=2)
    colorbar.update_ticks()
    ax2.set_ylabel(r"Signal-to-Noise Ratio $\uparrow$")
    ax1.set_xlabel(r"$\Delta V$")
    ax1.set_ylabel(r"Gradient Alignment $\uparrow$")
    ax2.set_xlabel(r"$\Delta V$")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.set_xlabel("Membrane Voltage (mV)")
    ax3.set_ylabel("Gating Function Value")

    for ax, label in zip(
        [ax1, ax2, ax3], [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}"]
    ):
        offset = ScaledTranslation(-0.12, 0.12, fig.dpi_scale_trans)
        ax.text(
            0,
            1,
            label,
            transform=ax.transAxes + offset,
            fontsize=10,
            va="top",
            ha="right",
        )
    plt.show()


if __name__ == "__main__":
    groups = get_grouped_files()
    print(f"Loaded {sum(len(files) for _, _, files in groups)} runs")
    plot_figure()
