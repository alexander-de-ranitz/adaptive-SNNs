import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from adaptive_SNN.utils.runner import _load_existing_solution


@dataclass
class RunFile:
    path: Path
    dv: float
    method: str
    perturbation_size: float | None


DATA_DIR = Path("results/delta_v_tuning_20260605_204715/results")
OUTPUT_PATH = Path("../figures/single_synapse_learning/")


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
    return (
        sol.ys[0].squeeze(),
        sol.ys[1].squeeze(),
        sol.ys[2].squeeze(),
    )  # reward, reward_noise, eligibility


def compute_file_stats(file: Path) -> dict[str, float]:
    try:
        data = np.load(file)
        return {"alignment": float(data["alignment"]), "SNR": float(data["snr"])}
    except:
        print("Precomputed stats not found, computing from raw data...")
    reward, reward_noise, eligibility = load_run_arrays(file)
    rpe = reward + reward_noise
    dW_task = (eligibility * rpe).ravel()
    dW_noise = (eligibility * reward_noise).ravel()

    alignment = np.sum(dW_task) / np.sum(np.abs(dW_task))
    snr = np.sum(dW_task) / np.sum(np.abs(dW_noise))

    result = {
        "alignment": float(alignment),
        "SNR": float(snr),
    }
    return result


def build_dataframe() -> pd.DataFrame:
    if os.path.exists(DATA_DIR / "delta_v_tuning_results.csv"):
        return pd.read_csv(DATA_DIR / "delta_v_tuning_results.csv")

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

    baseline = df[df["dv"] == 0.0].sort_values("perturbation_size")
    gated = df[df["dv"] != 0.0]

    gated_dvs = gated["dv"].unique()
    p_norm = LogNorm(
        vmin=float(gated["perturbation_size"].min()),
        vmax=float(gated["perturbation_size"].max()),
    )
    p_cmap = LinearSegmentedColormap.from_list(
        "p_cmap",
        plt.cm.summer(np.linspace(0.0, 1.0, 256)),
    )
    for perturbation_size, subset in gated.groupby("perturbation_size"):
        results = subset.groupby("dv").agg(
            alignment=("alignment", "mean"), snr=("SNR", "mean")
        )
        baseline_alignment = baseline[
            baseline["perturbation_size"] == perturbation_size
        ]["alignment"].mean()
        plt.plot(
            results.index,
            results["alignment"] / baseline_alignment,
            label=rf"$\sigma_\xi={perturbation_size:.2f} \text{{nS}}$",
            marker="o",
            markersize=5,
            c=p_cmap(p_norm(perturbation_size)),
        )
    plt.xscale("log")
    plt.xlabel("Delta V (log scale)")
    plt.ylabel("Relative Improvement in Alignment")
    plt.xticks(gated_dvs, [rf"$2^{{{-compute_exponent(dv)}}}$" for dv in gated_dvs])
    plt.xticks([], minor=True)
    xmin, xmax = plt.xlim()
    plt.hlines(1.0, xmin=xmin, xmax=xmax, color="k", linestyle="--")
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.title("Effect of Delta V and Noise on Alignment")
    plt.show()


if __name__ == "__main__":
    df = build_dataframe()
    df.to_csv(DATA_DIR / "delta_v_tuning_results.csv", index=False)
    plot_figure(df=df, save_path=OUTPUT_PATH, show=True)
