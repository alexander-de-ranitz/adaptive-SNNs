import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DATA_DIR = Path(
    "results/single_synapse_learning_extended_alignment_and_SNR_20261009_105439/results"
)
OUTPUT_PATH = Path("figures/single_synapse_learning/")

# Each result file holds per-chunk metrics of shape (n_chunks, n_traces): the gated traces (one per
# delta_V), followed by the other rules. Newer files name each column in "trace_labels".
UNGATED_CENTERED = "No gating (centered)"
UNGATED = "No gating"
SPIKE_CENTERED = "Spike gate (centered)"
SPIKE = "Spike gate"
METHOD_NAMES = {
    "ungated_centered": UNGATED_CENTERED,
    "ungated": UNGATED,
    "spike_centered": SPIKE_CENTERED,
    "spike": SPIKE,
}


def parse_run_name(file: Path) -> dict[str, float]:
    def find(pattern: str) -> float:
        match = re.search(pattern, file.name)
        return float(match.group(1)) if match is not None else float("nan")

    return {
        "perturbation_size": find(r"noise_(\d+\.?\d*)_"),
        "weight": find(r"w_(\d+\.?\d*)_"),
        "balance": find(r"b_(\d+\.?\d*)_"),
        "seed": find(r"seed_(\d+)_"),
    }


def build_dataframe() -> pd.DataFrame:
    """One row per (file, trace, chunk)."""
    rows: list[dict[str, float | str]] = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.endswith(".npz"):
            continue
        params = parse_run_name(DATA_DIR / name)
        data = np.load(DATA_DIR / name)
        delta_Vs = data["delta_Vs"]
        alignment, snr = data["alignment"], data["snr"]
        if "trace_labels" in data:
            extra = [str(label) for label in data["trace_labels"][len(delta_Vs) :]]
        else:
            extra = ["ungated_centered", "ungated"]
        assert alignment.shape[1] == len(delta_Vs) + len(extra)

        labels = ["gated"] * len(delta_Vs) + [METHOD_NAMES[label] for label in extra]
        dvs = list(delta_Vs) + [np.nan] * len(extra)
        for trace, (method, dv) in enumerate(zip(labels, dvs)):
            for chunk in range(alignment.shape[0]):
                rows.append(
                    {
                        "filename": name,
                        **params,
                        "method": method,
                        "dv": dv,
                        "chunk": chunk,
                        "alignment": alignment[chunk, trace],
                        "SNR": snr[chunk, trace],
                    }
                )
    return pd.DataFrame(rows)


def compute_exponent(dv: float) -> int:
    return int(round(-np.log2(dv)))


def aggregate_over_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over chunks (and seeds, if multiple)."""
    return (
        df.groupby(
            ["method", "dv", "perturbation_size", "balance", "weight"], dropna=False
        )
        .agg(
            alignment=("alignment", "mean"),
            alignment_std=("alignment", "std"),
            snr=("SNR", "mean"),
            snr_std=("SNR", "std"),
        )
        .reset_index()
    )


def delta_v_figure(
    df: pd.DataFrame,
    balance: float,
    save_path: Path | None = OUTPUT_PATH,
    show: bool = True,
):
    """Alignment and SNR vs delta_V, one column per perturbation size and one line per initial E weight.

    The two ungated variants are shown as separate points to the right of the gated sweep.
    Error bars/bands show the std over chunks.
    """
    df_agg = aggregate_over_chunks(df[df["balance"] == balance])
    noise_levels = sorted(df_agg["perturbation_size"].unique())
    weights = sorted(df_agg["weight"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(weights)))

    exponents = sorted(compute_exponent(dv) for dv in df_agg["dv"].dropna().unique())
    ungated_x = {
        UNGATED_CENTERED: exponents[-1] + 1.5,
        UNGATED: exponents[-1] + 2.5,
    }

    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(noise_levels),
        figsize=(4 * len(noise_levels), 6),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    metrics = [("alignment", "Alignment"), ("snr", "SNR")]

    for col, noise in enumerate(noise_levels):
        subset_noise = df_agg[df_agg["perturbation_size"] == noise]
        for row, (metric, metric_label) in enumerate(metrics):
            ax = axs[row, col]
            for weight, color in zip(weights, colors):
                subset = subset_noise[subset_noise["weight"] == weight]

                gated = subset[subset["method"] == "gated"].copy()
                gated["exponent"] = gated["dv"].map(compute_exponent)
                gated = gated.sort_values("exponent")
                ax.plot(
                    gated["exponent"],
                    gated[metric],
                    color=color,
                    marker="o",
                    markersize=3,
                    label=f"w={weight:g}",
                )
                ax.fill_between(
                    gated["exponent"],
                    gated[metric] - gated[f"{metric}_std"],
                    gated[metric] + gated[f"{metric}_std"],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )

                for method, x in ungated_x.items():
                    ungated = subset[subset["method"] == method]
                    ax.errorbar(
                        [x],
                        ungated[metric],
                        yerr=ungated[f"{metric}_std"],
                        color=color,
                        marker="s" if method == UNGATED_CENTERED else "D",
                        markersize=4,
                        capsize=2,
                    )

            if metric == "alignment":
                ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
                ax.set_title(rf"$\sigma_\xi={noise:g}$ nS")
            ax.axvline(exponents[-1] + 0.75, color="k", linewidth=0.5, alpha=0.3)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(metric_label)

    tick_positions = exponents + list(ungated_x.values())
    tick_labels = [rf"$2^{{-{e}}}$" for e in exponents] + [
        "No gating\n(centered)",
        "No gating",
    ]
    for ax in axs[-1]:
        ax.set_xticks(tick_positions, labels=tick_labels, rotation=90)
        ax.set_xlabel(r"$\Delta V$")
    axs[0, -1].legend(
        title="Initial E weight",
        fontsize="small",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    fig.suptitle(f"Balance = {balance:g}")
    fig.tight_layout()
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"delta_v_tuning_b_{balance:g}.png", dpi=300)
    if show:
        plt.show()


def heatmap_figure(
    df: pd.DataFrame,
    save_path: Path | None = OUTPUT_PATH,
    show: bool = True,
):
    """Alignment heatmaps (balance vs initial E weight), one row per trace and one column per perturbation size.

    Rows are ordered from no gating to the steepest gating. Values are means over chunks.
    """
    df_agg = aggregate_over_chunks(df)
    df_agg["exponent"] = df_agg["dv"].map(
        lambda dv: np.nan if np.isnan(dv) else compute_exponent(dv)
    )
    noise_levels = sorted(df_agg["perturbation_size"].unique())
    balances = sorted(df_agg["balance"].unique())
    weights = sorted(df_agg["weight"].unique())

    exponents = sorted(int(e) for e in df_agg["exponent"].dropna().unique())
    methods = set(df_agg["method"])
    rows = (
        [(m, None) for m in (UNGATED, UNGATED_CENTERED) if m in methods]
        + [("gated", e) for e in exponents]
        + [(m, None) for m in (SPIKE_CENTERED, SPIKE) if m in methods]
    )

    vmax = df_agg["alignment"].max()
    vmin = df_agg["alignment"].min()

    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=len(noise_levels),
        figsize=(3.2 * len(noise_levels) + 1.5, 2.0 * len(rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, (method, exponent) in enumerate(rows):
        if method == "gated":
            subset_method = df_agg[
                (df_agg["method"] == "gated") & (df_agg["exponent"] == exponent)
            ]
            row_label = rf"$\Delta V=2^{{-{exponent}}}$"
        else:
            subset_method = df_agg[df_agg["method"] == method]
            row_label = method.replace(" (", "\n(")

        for col, noise in enumerate(noise_levels):
            ax = axs[row, col]
            subset = subset_method[subset_method["perturbation_size"] == noise]
            pivot = subset.pivot(index="balance", columns="weight", values="alignment")
            pivot = pivot.reindex(index=balances, columns=weights)
            im = ax.imshow(
                pivot.values,
                cmap="Greens",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                origin="lower",
            )
            if row == 0:
                ax.set_title(rf"$\sigma_\xi={noise:g}$ nS")
            if col == 0:
                ax.set_ylabel(row_label + "\n\nBalance")
            if row == len(rows) - 1:
                ax.set_xlabel("Initial E weight")

    for ax in axs.flat:
        ax.set_xticks(np.arange(len(weights)), labels=[f"{w:g}" for w in weights])
        ax.set_yticks(np.arange(len(balances)), labels=[f"{b:g}" for b in balances])
        ax.tick_params(labelsize="small")

    fig.colorbar(im, ax=axs, orientation="vertical", label="Alignment", shrink=0.5)
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "delta_v_tuning_alignment_heatmap.png", dpi=300)
    if show:
        plt.show()


if __name__ == "__main__":
    df = build_dataframe()
    df.to_csv(DATA_DIR / "delta_v_tuning_results.csv", index=False)
    heatmap_figure(df=df, save_path=None, show=True)
