import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import ScaledTranslation

from adaptive_snn.models.networks import AbstractLIFNetwork

DATA_DIR = Path(
    "results/single_synapse_learning_extended_firing_statistics_20261009_122621/results"
)
OUTPUT_PATH = Path("figures/single_synapse_learning_ext_firing_statistics")

# (column, label, scale applied to the stored SI value)
ROW_SPECS = [
    ("firing_rate", "Firing Rate (Hz)", 1.0),
    ("CV_ISI", "CV ISI", 1.0),
    ("mean_V", "Mean V (mV)", 1e3),
    ("mean_E_conductance", "Mean $g_E$ (nS)", 1e9),
    ("mean_I_conductance", "Mean $g_I$ (nS)", 1e9),
    ("E_I_ratio", "$g_E / g_I$", 1.0),  # Derived: ratio of the mean conductances
    (
        "conductance_ratio",
        "$(g_E + g_I) / g_L$",
        1.0,
    ),  # Derived: ratio of total conductance to leak
]
DERIVED_COLUMNS = {
    "E_I_ratio",
    "conductance_ratio",
}  # Columns that are not stored in the .npz files but derived from them

MIN_FIRING_RATE_FOR_CV_ISI = 1.0  # Hz; CV ISI is unreliable/undefined below this rate

# Cells that fulfill these criteria are deemed realistic(ish) highlighted
REALSTIC_RANGE = {
    "firing_rate": (5.0, 25.0),
    "conductance_ratio": (2.0, 10.0),
}


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
        data = np.load(info["path"])
        for column, _, scale in ROW_SPECS:
            if column not in DERIVED_COLUMNS:
                info[column] = float(np.asarray(data[column]).item()) * scale
        info["E_I_ratio"] = info["mean_E_conductance"] / info["mean_I_conductance"]
        info["conductance_ratio"] = (
            info["mean_E_conductance"] + info["mean_I_conductance"]
        ) / (1e9 * AbstractLIFNetwork.leak_conductance)
        rows.append(info)
    return pd.DataFrame(
        rows,
        columns=["path", "perturbation_size", "balance", "weight"]
        + [column for column, _, _ in ROW_SPECS],
    )


def format_tick(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def draw_mask_outline(ax, mask: np.ndarray, color: str = "white", linewidth=1.5):
    """Draw the boundary of the True cells of a 2D mask on an imshow axis."""
    padded = np.pad(mask, 1, constant_values=False)
    segments = []
    # Horizontal edges: between vertically adjacent cells that differ
    rows, cols = np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])
    segments += [[(c - 0.5, r - 0.5), (c + 0.5, r - 0.5)] for r, c in zip(rows, cols)]
    # Vertical edges: between horizontally adjacent cells that differ
    rows, cols = np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])
    segments += [[(c - 0.5, r - 0.5), (c - 0.5, r + 0.5)] for r, c in zip(rows, cols)]
    ax.add_collection(
        LineCollection(
            segments, colors=color, linewidths=linewidth, capstyle="projecting"
        )
    )


def plot_firing_statistics(
    df: pd.DataFrame, save_path: Path | None = OUTPUT_PATH, show: bool = True
):
    """Make a grid of heatmaps showing the relevant statistics.

    Each heatmap has w on the x-axis and b on the y-axis. There is a column for each
    perturbation size, and a row for each statistic in ROW_SPECS. Color
    scale is shared across columns within a row so statistics are comparable across
    perturbation sizes.
    """
    perturbation_sizes = sorted(df["perturbation_size"].unique())
    n_cols = len(perturbation_sizes)
    n_rows = len(ROW_SPECS)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows), sharex=True, sharey=True
    )

    for row_idx, (column, row_label, _) in enumerate(ROW_SPECS):
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

            valid = np.ones_like(pivot.values, dtype=bool)
            for col, (min_v, max_v) in REALSTIC_RANGE.items():
                piv = df[df["perturbation_size"] == perturbation_size].pivot(
                    index="balance", columns="weight", values=col
                )
                valid &= (piv.values >= min_v) & (piv.values <= max_v)
            draw_mask_outline(ax, valid)

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
