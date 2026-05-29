import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from matplotlib.transforms import ScaledTranslation

from adaptive_SNN.models.networks import GatedLIFNetwork
from adaptive_SNN.utils.runner import _load_existing_solution


@dataclass
class RunFile:
    path: Path
    dv: float
    method: str
    perturbation_size: float | None


DATA_DIR = Path("results/delta_v_tuning_20260529_152552/results")
OUTPUT_PATH = Path("figures/delta_v_tuning")


def parse_run_file(file: Path) -> RunFile:
    dv = re.search(r"dv_(\d+\.?\d*)_", file.name).group(1)
    dv = float(dv)
    noise_match = re.search(r"noise_(\d+\.?\d*)_", file.name)
    if noise_match is None:
        noise = float("nan")
    else:
        noise = float(noise_match.group(1))
    method = "gated" if dv != 0.0 else "default"
    return RunFile(path=file, dv=dv, method=method, perturbation_size=noise)


def load_run_arrays(file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sol, _ = _load_existing_solution(file)
    return sol.ys[0][0, -1], sol.ys[1], sol.ys[2]


def compute_file_stats(file: Path) -> dict[str, float]:
    eligibility, reward_noise, rpe = load_run_arrays(file)
    dW_task = (eligibility * rpe).ravel()
    dW_noise = (eligibility * reward_noise).ravel()

    alignment = np.sum(dW_task) / np.sum(np.abs(dW_task))
    snr = np.sum(dW_task) / np.sum(np.abs(dW_noise))

    result = {
        "alignment": float(alignment),
        "SNR": float(snr),
    }
    del eligibility, reward_noise, rpe, dW_task, dW_noise
    jax.clear_caches()
    gc.collect()
    return result


def build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for name in os.listdir(DATA_DIR):
        if not name.endswith(".npz"):
            continue
        run = parse_run_file(DATA_DIR / name)
        stats = compute_file_stats(run.path)
        rows.append(
            {
                "filename": run.path.name,
                "dv": run.dv,
                "method": run.method,
                "perturbation_size": run.perturbation_size,
                "alignment": stats["alignment"],
                "SNR": stats["SNR"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "filename",
            "dv",
            "method",
            "perturbation_size",
            "alignment",
            "SNR",
        ],
    )


def compute_exponent(dv: float) -> float:
    return int(round(-np.log2(dv)))


def plot_figure(
    df: pd.DataFrame | None = None,
    save_path: Path = OUTPUT_PATH,
    show: bool = True,
):
    if df is None:
        df = build_dataframe()

    summary = df.groupby(["dv", "method"], as_index=False).agg(
        alignment=("alignment", "mean"),
        SNR=("SNR", "mean"),
        alignment_std=("alignment", "std"),
        SNR_std=("SNR", "std"),
        run_count=("filename", "count"),
    )
    summary["sort_key"] = summary["method"].eq("default").astype(int)
    summary = summary.sort_values(by=["sort_key", "dv"]).drop(columns="sort_key")
    summary["alignment_std"] = summary["alignment_std"].fillna(0.0)
    summary["SNR_std"] = summary["SNR_std"].fillna(0.0)

    fig = plt.figure(figsize=(8.0, 4.5))

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    gated_dvs = summary.loc[summary["method"] == "gated", "dv"].to_numpy(dtype=float)
    dv_min = float(gated_dvs.min())
    dv_max = float(gated_dvs.max())

    dv_norm = LogNorm(vmin=dv_min, vmax=dv_max)
    dv_cmap = LinearSegmentedColormap.from_list(
        "Greens_r_trimmed",
        plt.cm.Greens_r(np.linspace(0.0, 0.75, 256)),
    )

    for _, row in summary.iterrows():
        dv = float(row["dv"])
        method = str(row["method"])
        stats = row.to_dict()
        print(f"Processing dv={dv}, method={method}, {int(stats['run_count'])} files")
        if method == "gated":
            color = dv_cmap(dv_norm(dv))
            label = rf"$2^{{{-compute_exponent(dv)}}}$"

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
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Loaded {len(df)} runs")
    for ps, group_df in df.groupby(["perturbation_size"]):
        print(f"Perturbation size: {ps}, {len(group_df)} runs")
        plot_figure(
            df=group_df,
            save_path=OUTPUT_PATH.parent / f"delta_v_tuning_noise_{ps}.pdf",
            show=False,
        )
