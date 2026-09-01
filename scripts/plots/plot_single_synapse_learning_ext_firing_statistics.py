import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.transforms import ScaledTranslation

DATA_DIR = Path(
    "results/single_synapse_learning_extended_firing_statistics_20260805_205127/results"
)
OUTPUT_PATH = Path("figures/single_synapse_learning_ext_firing_statistics")

ROW_SPECS = [
    ("firing_rate", "Firing Rate (Hz)"),
    ("CV_ISI", "CV ISI"),
    ("mean_V", "Mean V (mV)"),
]

MIN_FIRING_RATE_FOR_CV_ISI = 1.0  # Hz; CV ISI is unreliable/undefined below this rate


def parse_run_file(file: Path) -> dict:
    noise_match = re.search(r"noise_(\d+\.?\d*)_", file.name)
    if noise_match is None:
        noise = float("nan")
    else:
        noise = float(noise_match.group(1))
    balance_match = re.search(r"b_(\d+\.?\d*)_", file.name)
    if balance_match is None:
        balance = float("nan")
    else:
        balance = float(balance_match.group(1))
    w_match = re.search(r"w_(\d+\.?\d*)_", file.name)
    if w_match is None:
        w = float("nan")
    else:
        w = float(w_match.group(1))
    return {"path": file, "perturbation_size": noise, "balance": balance, "weight": w}


def build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for name in os.listdir(DATA_DIR):
        if not name.endswith(".npz"):
            continue
        info = parse_run_file(DATA_DIR / name)
        data = jnp.load(info["path"])
        info.update(
            {
                "firing_rate": float(data["firing_rate"]),
                "CV_ISI": float(data["CV_ISI"].item()),
                "mean_V": float(data["mean_V"]),
            }
        )
        rows.append(info)
    return pd.DataFrame(
        rows,
        columns=[
            "path",
            "perturbation_size",
            "balance",
            "weight",
            "firing_rate",
            "CV_ISI",
            "mean_V",
        ],
    )


def format_tick(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def plot_firing_statistics(
    df: pd.DataFrame, save_path: Path | None = OUTPUT_PATH, show: bool = True
):
    """Make a 3x3 grid of heatmaps showing the relevant statistics.

    Each heatmap has w on the x-axis and b on the y-axis. There is a column for each
    perturbation size, and a row for each statistic (firing rate, CV ISI, mean V). Color
    scale is shared across columns within a row so statistics are comparable across
    perturbation sizes.
    """
    perturbation_sizes = sorted(df["perturbation_size"].unique())
    n_cols = len(perturbation_sizes)
    fig, axs = plt.subplots(
        3, n_cols, figsize=(2.6 * n_cols, 6.5), sharex=True, sharey=True
    )

    for row_idx, (column, row_label) in enumerate(ROW_SPECS):
        pivots = []
        for perturbation_size in perturbation_sizes:
            subset = df[df["perturbation_size"] == perturbation_size]
            if column == "CV_ISI":
                subset = subset.copy()
                subset.loc[
                    subset["firing_rate"] <= MIN_FIRING_RATE_FOR_CV_ISI, "CV_ISI"
                ] = np.nan
            pivots.append(
                subset.pivot(index="balance", columns="weight", values=column)
            )

        vmin = min(np.nanmin(p.values) for p in pivots)
        vmax = max(np.nanmax(p.values) for p in pivots)

        for col_idx, (perturbation_size, pivot) in enumerate(
            zip(perturbation_sizes, pivots)
        ):
            ax = axs[row_idx, col_idx]
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            if row_idx == 0:
                ax.set_title(
                    rf"Perturbation: \SI{{{perturbation_size:g}}}{{\nano\siemens}}"
                )
            if row_idx == len(ROW_SPECS) - 1:
                ax.set_xlabel("Weight $w$")
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nBalance $b$")

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([format_tick(v) for v in pivot.columns], rotation=90)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([format_tick(v) for v in pivot.index])

        fig.colorbar(im, ax=axs[row_idx, :], label=row_label, fraction=0.02, pad=0.02)

    for row_idx in range(len(ROW_SPECS)):
        offset = ScaledTranslation(-0.35, 0.12, fig.dpi_scale_trans)
        axs[row_idx, 0].text(
            0,
            1,
            rf"\textbf{{{chr(65 + row_idx)}}}",
            transform=axs[row_idx, 0].transAxes + offset,
            fontsize=10,
            va="top",
            ha="right",
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Loaded {len(df)} runs")
    plot_firing_statistics(df=df, save_path=OUTPUT_PATH, show=True)
