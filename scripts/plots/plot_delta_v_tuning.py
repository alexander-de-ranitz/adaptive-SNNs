import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.transforms import ScaledTranslation

from adaptive_snn.models.networks import GatedLIFNetwork

DATA_DIR = Path("results/single_synapse_learning_20261108_163645/results")
OUTPUT_PATH = Path("figures/delta_v_tuning_new")

# Style of the ungated rule (a constant gate) in panel C
NO_GATING_STYLE = dict(color="black", linestyles="dashed")

# Bars per rule in panels A and B, in this order: centering per neuron (filled), per synapse (dotted), none (hatched)
CENTERINGS = ["neuron", "synapse", "none"]
CENTERING_LABELS = {
    "neuron": "Centered (neuron)",
    "synapse": "Centered (synapse)",
    "none": "Uncentered",
}
BAR_WIDTH = 0.27


def centering_style(centering: str, color) -> dict:
    """Bar style that distinguishes the centering variants of a rule of the given color."""
    if centering == "neuron":
        return dict(facecolor=color, edgecolor=color, hatch=None)
    if centering == "synapse":
        return dict(facecolor=to_rgba(color, 0.35), edgecolor=color, hatch="....")
    return dict(facecolor="white", edgecolor=color, hatch="////")


def plot_grouped_bars(
    ax,
    groups: list[tuple[pd.DataFrame, object]],
    metric: str,
    tick_labels: list[str],
    centerings: list[str] = CENTERINGS,
    width: float = BAR_WIDTH,
):
    """One group of bars per rule (rows of the summary, color), with one bar per centering variant in centerings. A
    variant that a rule does not have leaves its slot empty, such that the bar positions are aligned across groups."""
    offsets = (np.arange(len(centerings)) - (len(centerings) - 1) / 2) * width
    for x, (rows, color) in enumerate(groups):
        for centering, offset in zip(centerings, offsets):
            match = rows[rows["centering"] == centering]
            if match.empty:
                continue
            row = match.iloc[0]
            ax.bar(
                x + offset,
                row[f"{metric}_mean"],
                width=width,
                **centering_style(centering, color),
            )
            ax.errorbar(
                x + offset,
                row[f"{metric}_mean"],
                yerr=row[f"{metric}_ci"],
                color="black",
                linewidth=1,
                capsize=0,
            )
    ax.set_xticks(range(len(groups)), tick_labels)


def build_dataframe() -> pd.DataFrame:
    """One row per (file, trace). Each file holds the weight updates of many parallel trials per window, of shape
    (n_windows, n_trials, n_traces), split into the part driven by the trial's own counterfactual RPE (perturbed neuron
    minus unperturbed twin) and the part driven by the summed RPEs of all other trials (the external noise)."""
    rows = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.endswith(".npz"):
            continue
        data = np.load(DATA_DIR / name)
        n_windows, n_trials, n_traces = data["task_updates"].shape
        dw_task = data["task_updates"].reshape(-1, n_traces)
        dw_noise_raw = data["noise_updates"].reshape(-1, n_traces)
        # Remove the contribution of the mean RPE of the other trials from the external noise
        dw_noise = dw_noise_raw - (n_trials - 1) * data["mean_rpe"] * data[
            "eligibility_sums"
        ].reshape(-1, n_traces)

        # SNR = E[dw]^2 / Var[dw]. The external noise is independent and zero-mean, such that E[dw] = E[dw_task] and
        # Var[dw] = Var[dw_task] + Var[dw_noise]. The squared mean is corrected for its estimation bias (var / n).
        var_task = dw_task.var(0, ddof=1)
        snr = (dw_task.mean(0) ** 2 - var_task / dw_task.shape[0]) / (
            var_task + dw_noise.var(0, ddof=1)
        )
        # Drift caused by the non-zero mean RPE (raw external noise), relative to the expected task-driven update
        drift = dw_noise_raw.mean(0) / dw_task.mean(0)

        for trace, label in enumerate(data["trace_labels"]):
            label = str(label)
            if label.endswith("_syncentered"):
                centering = "synapse"
            elif label.endswith("_centered"):
                centering = "neuron"
            else:
                centering = "none"
            base = label.removesuffix("_syncentered").removesuffix("_centered")
            is_gated = base.startswith("gated_")
            rows.append(
                {
                    "filename": name,
                    "method": "gated" if is_gated else base,
                    "dv": float(base.removeprefix("gated_")) if is_gated else np.nan,
                    "centering": centering,  # "none", "neuron" or "synapse"
                    "N": n_trials,
                    "SNR": snr[trace],
                    "drift": drift[trace],
                }
            )
    return pd.DataFrame(rows)


def compute_exponent(dv: float) -> int:
    return int(round(-np.log2(dv)))


def plot_figure(df: pd.DataFrame, save_path: Path = OUTPUT_PATH, show: bool = True):
    metrics = ["SNR", "drift"]
    # Mean and 95% CI over runs (seeds), each run being an independent network of parallel trials
    summary = df.groupby(["method", "dv", "centering"], as_index=False, dropna=False)[
        metrics
    ].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    for m in metrics:
        summary[f"{m}_ci"] = 1.96 * summary[f"{m}_std"] / np.sqrt(summary[f"{m}_count"])
    for _, row in summary.iterrows():
        print(
            f"{row['method']:8s} dv={row['dv']:.3g} centering={row['centering']:7s} ({int(row['SNR_count'])} files): "
            + ", ".join(
                f"{m} {row[f'{m}_mean']:.4f} ± {row[f'{m}_ci']:.4f}" for m in metrics
            )
        )

    fig = plt.figure(figsize=(8.0, 4.5))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    gated = summary[summary["method"] == "gated"].sort_values("dv")
    dvs = np.unique(gated["dv"].to_numpy(dtype=float))
    dv_norm = LogNorm(vmin=dvs.min(), vmax=dvs.max())
    dv_cmap = LinearSegmentedColormap.from_list(
        "Greens_r_trimmed",
        plt.cm.Greens_r(np.linspace(0.0, 0.75, 256)),
    )
    dv_labels = [rf"$2^{{{-compute_exponent(dv)}}}$" for dv in dvs]

    # Panel A: SNR of every rule with per-neuron centering (the SNR hardly depends on the centering variant).
    # Panel B: drift due to the mean RPE, for every rule with each centering variant.
    groups = [(gated[gated["dv"] == dv], dv_cmap(dv_norm(dv))) for dv in dvs] + [
        (summary[summary["method"] == "spike"], "tab:red"),
        (summary[summary["method"] == "ungated"], "black"),
    ]
    tick_labels = dv_labels + ["Spike", "None"]
    plot_grouped_bars(ax1, groups, "SNR", tick_labels, centerings=["neuron"], width=0.7)
    plot_grouped_bars(ax2, groups, "drift", tick_labels)
    ax2.axhline(0.0, color="grey", linewidth=0.8)

    # Panel C: shape of the gating functions
    network = GatedLIFNetwork(N_neurons=1, dt=1e-4, N_inputs=0)
    voltages = np.linspace(-75 * 1e-3, -50 * 1e-3, int(1e5))
    for dv, label in zip(dvs, dv_labels):
        gating_values = network.gating_function(voltages, delta_V=dv)
        gating_values = gating_values / gating_values.max()  # Normalize to [0, 1]
        ax3.plot(
            voltages * 1e3,
            gating_values,
            label=label,
            color=dv_cmap(dv_norm(dv)),
            linewidth=2.5,
        )
    ax3.hlines(1.0, xmin=-75, xmax=-50, **NO_GATING_STYLE)
    ax3.set_xlim(-75, -50)
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(
        ymin, ymax + 0.5 * (ymax - ymin)
    )  # Headroom for the legend above the bars
    ax2.legend(
        handles=[
            Patch(
                **centering_style(centering, "grey"), label=CENTERING_LABELS[centering]
            )
            for centering in CENTERINGS
        ],
        loc="upper right",
    )

    sm = plt.cm.ScalarMappable(norm=dv_norm, cmap=dv_cmap)
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=ax3, label=r"$\Delta V$", pad=0.01)
    colorbar.locator = LogLocator(base=2)
    colorbar.formatter = LogFormatterMathtext(base=2)
    colorbar.update_ticks()
    ax1.set_ylabel(rf"SNR, $N={df['N'].iloc[0]}$ $\uparrow$")
    ax2.set_ylabel(r"Drift due to mean RPE (rel.)")
    ax1.set_xlabel(r"Gating ($\Delta V$)")
    ax2.set_xlabel(r"Gating ($\Delta V$)")
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

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.01)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Loaded {df['filename'].nunique()} runs")
    plot_figure(df=df, save_path=OUTPUT_PATH, show=True)
