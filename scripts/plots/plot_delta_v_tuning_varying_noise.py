import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from adaptive_SNN.utils.runner import _load_existing_solution


@dataclass
class RunFile:
    path: Path
    dv: float
    method: str
    perturbation_size: float
    balance: float
    weight: float


DATA_DIR = Path("results/delta_v_tuning_20260714_164259/results")
OUTPUT_PATH = Path("../figures/single_synapse_learning/")


def parse_run_file(file: Path) -> RunFile:
    dv = re.search(r"dv_(\d+\.?\d*)_", file.name).group(1)
    dv = float(dv)
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
    method = "gated" if dv != 0.0 else "default"
    return RunFile(
        path=file,
        dv=dv,
        method=method,
        perturbation_size=noise,
        balance=balance,
        weight=w,
    )


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
                "balance": run.balance,
                "weight": run.weight,
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
            "balance",
            "weight",
            "alignment",
            "SNR",
        ],
    )


def compute_exponent(dv: float) -> float:
    return int(round(-np.log2(dv)))


def heatmap_figure(
    df: pd.DataFrame | None = None,
    save_path: Path = OUTPUT_PATH,
    show: bool = True,
):
    if df is None:
        df = build_dataframe()

    # We are varying over
    # 1. delta_v (4)
    # 2. perturbation_size (3)
    # 3. balance (5)
    # 4. initial E weight (4)
    # We make one heatmap of balance vs weight for each combination of delta_v and perturbation_size
    fig, axs = plt.subplots(
        nrows=len(df["dv"].unique()),
        ncols=len(df["perturbation_size"].unique()),
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )
    axs = axs.flatten()

    df_agg = (
        df.groupby(["dv", "perturbation_size", "balance", "weight"])
        .agg(
            alignment=("alignment", "mean"),
            snr=("SNR", "mean"),
            alignment_std=("alignment", "std"),
            snr_std=("SNR", "std"),
        )
        .reset_index()
    )
    # Sort such that dv=0.0 (no gating) is first, then decreasing dv
    df_agg = df_agg.sort_values(
        by=["dv", "perturbation_size"],
        key=lambda x: x.where(x != 0.0, np.inf),
        ascending=[False, True],
    ).reset_index()

    print(df_agg.head())

    vmax = df_agg["snr"].max()
    vmin = df_agg["snr"].min()
    for i, ((dv, perturbation_size), subset) in enumerate(
        df_agg.groupby(["dv", "perturbation_size"], sort=False)
    ):
        print(f"Plotting heatmap for dv={dv}, perturbation_size={perturbation_size}")
        ax = axs[i]
        pivot_table = subset.pivot(index="balance", columns="weight", values="snr")

        im = ax.imshow(
            pivot_table.values,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        dv_label = (
            rf"$\Delta V=2^{{{-compute_exponent(dv)}}}$" if dv != 0.0 else r"No Gating"
        )
        ax.set_title(dv_label + rf", $\sigma_\xi={perturbation_size:.2f} nS$")
        ax.set_xlabel("Initial E Weight")
        ax.set_ylabel("Balance")
        ax.set_xticks(
            np.arange(len(pivot_table.columns)),
            labels=[f"{x:.2f}" for x in pivot_table.columns],
        )
        ax.set_yticks(
            np.arange(len(pivot_table.index)),
            labels=[f"{x:.5f}" for x in pivot_table.index],
        )
        ax.label_outer()
    fig.colorbar(im, ax=axs, orientation="vertical", label="SNR")
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "delta_v_tuning_heatmap_SNR.png", dpi=300)
    if show:
        plt.show()


if __name__ == "__main__":
    df = build_dataframe()
    df.to_csv(DATA_DIR / "delta_v_tuning_results.csv", index=False)
    heatmap_figure(df=df, save_path=None, show=True)
