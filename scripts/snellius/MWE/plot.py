from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, PyTree

try:
    from scripts.snellius.MWE.define_experiments import (
        ExperimentConfig,
        generate_experiment_configs,
    )
except ModuleNotFoundError:
    from define_experiments import ExperimentConfig, generate_experiment_configs


# Edit these values directly when you want to plot a different result set.
RESULTS_DIR = Path("results/MWE_tune_delta_v_20260413_223241")
OUTPUT_PATH = Path("figures/MWE/MWE_tune_delta_v_20260413_223241.png")
SHOW_FIGURE = False
PLOT_INDIVIDUAL_RUNS = False  # Set to True to plot individual runs with low opacity
NORMALIZED_DW = False


@dataclass(frozen=True)
class RunRecord:
    noise_level: float
    learning_rate: float
    reward_noise_level: float
    run_type: str
    key_seed: int
    times: np.ndarray
    state: PyTree
    cfg_idx: int | None = None
    delta_v: float | None = None


def parse_filename(
    file_name: str,
) -> tuple[float, float, float, str, int, int | None, float | None]:
    new_pattern = re.compile(
        r"^results_cfg(?P<cfg_idx>\d+)_lr(?P<lr>[-+0-9.eE]+)_rnr(?P<rnr>[-+0-9.eE]+)"
        r"(?:_[^_]+)*_(?P<run_type>eligibility|gated)(?:\.[^_]+)?_seed(?P<key>\d+)(?:_[^_]+)*\.npz$"
    )
    new_match = new_pattern.match(file_name)
    if new_match is not None:
        lr = float(new_match.group("lr"))
        rnl = float(new_match.group("rnr"))
        corrupted_dv_match = re.search(
            r"(?:^|_)dv(?P<dv_int>[-+]?\d+)_(?:eligibility|gated)\.(?P<dv_frac>\d+)(?:_|\.|$)",
            file_name,
        )
        if corrupted_dv_match is not None:
            dv_int = corrupted_dv_match.group("dv_int")
            dv_frac = corrupted_dv_match.group("dv_frac")
            delta_v = float(f"{dv_int}.{dv_frac}")
        else:
            dv_match = re.search(r"(?:^|_)dv(?P<dv>[-+0-9.eE]+)(?:_|$)", file_name)
            delta_v = float(dv_match.group("dv")) if dv_match is not None else None
        run_type = new_match.group("run_type")
        run_type = "Default" if run_type == "eligibility" else "Gated"
        key_seed = int(new_match.group("key"))
        cfg_idx = int(new_match.group("cfg_idx"))
        return 0.0, lr, rnl, run_type, key_seed, cfg_idx, delta_v

    old_pattern = re.compile(
        r"^results_nl(?P<nl>[-+0-9.eE]+)_lr(?P<lr>[-+0-9.eE]+)_rnl(?P<rnl_pre>[^_]+)_"
        r"(?P<run_type>eligibility|gated)(?P<rnl_post>\.[^_]+)?_key(?P<key>\d+)\.npz$"
    )
    old_match = old_pattern.match(file_name)
    if old_match is None:
        raise ValueError(f"Could not parse filename: {file_name}")

    nl = float(old_match.group("nl"))
    lr = float(old_match.group("lr"))
    rnl_pre = old_match.group("rnl_pre")
    rnl_post = old_match.group("rnl_post") or ""
    reward_noise_level = float(f"{rnl_pre}{rnl_post}")
    run_type = old_match.group("run_type")
    run_type = "Default" if run_type == "eligibility" else "Gated"
    key_seed = int(old_match.group("key"))
    return nl, lr, reward_noise_level, run_type, key_seed, None, None


def extract_state(npz_path: Path) -> tuple[np.ndarray, Array]:
    sol = np.load(npz_path, allow_pickle=True)["sol"].item()
    times = np.asarray(sol.ts)
    return times, sol.ys


def resolve_results_dir(input_dir: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    if list(input_dir.glob("*.npz")):
        return input_dir

    nested_results = input_dir / "results"
    if nested_results.exists() and list(nested_results.glob("*.npz")):
        return nested_results

    raise FileNotFoundError(f"No .npz files found in {input_dir} or {nested_results}.")


def load_runs(results_dir: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for npz_path in sorted(results_dir.glob("*.npz")):
        try:
            nl, lr, rnl, run_type, key_seed, cfg_idx, delta_v = parse_filename(
                npz_path.name
            )
            times, state = extract_state(npz_path)
            runs.append(
                RunRecord(
                    noise_level=nl,
                    learning_rate=lr,
                    reward_noise_level=rnl,
                    run_type=run_type,
                    key_seed=key_seed,
                    times=times,
                    state=state,
                    cfg_idx=cfg_idx,
                    delta_v=delta_v,
                )
            )
        except Exception as exc:
            print(f"Skipping {npz_path.name}: {exc}")

    if not runs:
        raise RuntimeError(f"No valid runs could be loaded from {results_dir}")
    return runs


def extract_weight_trace(state: PyTree) -> np.ndarray:
    return state.agent_state.noisy_network.network_state.W[:, 1, -1]


def mean_w_over_keys(
    records: list[RunRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = records[0].times
    traces = np.vstack([extract_weight_trace(record.state) for record in records])
    return times, np.mean(traces, axis=0), np.std(traces, axis=0)


def compute_dw_split(record: RunRecord) -> tuple[np.ndarray, np.ndarray]:
    state = record.state
    eligibility = state.agent_state.noisy_network.network_state.features.eligibility[
        :, 1, -1
    ].squeeze()
    reward_noise = state.agent_state.reward_noise.squeeze()
    RPE = state.agent_state.RPE.squeeze()
    dW_signal = eligibility * RPE
    dW_noise = eligibility * reward_noise
    dW_signal_cum = np.cumsum(dW_signal)
    dW_noise_cum = np.cumsum(dW_noise)
    if NORMALIZED_DW:
        norm_factor_signal = np.sum(np.abs(dW_signal))
        norm_factor_noise = np.sum(np.abs(dW_noise))
        dW_signal_cum /= norm_factor_signal
        dW_noise_cum /= norm_factor_noise
    return dW_signal_cum, dW_noise_cum


def mean_dw_split(
    records: list[RunRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dW_signals = []
    dW_noises = []
    for record in records:
        dW_signal, dW_noise = compute_dw_split(record)
        dW_signals.append(dW_signal)
        dW_noises.append(dW_noise)

    times = records[0].times
    mean_signal = np.mean(dW_signals, axis=0)
    mean_noise = np.mean(dW_noises, axis=0)
    std_signal = np.std(dW_signals, axis=0)
    std_noise = np.std(dW_noises, axis=0)
    return times, mean_signal, mean_noise, std_signal, std_noise


def combined(records: list[RunRecord]) -> tuple[np.ndarray, np.ndarray]:
    dW_combined = []
    for record in records:
        state = record.state
        eligibility = (
            state.agent_state.noisy_network.network_state.features.eligibility[
                :, 1, -1
            ].squeeze()
        )
        reward_noise = state.agent_state.reward_noise.squeeze()
        RPE = state.agent_state.RPE.squeeze()
        combined_reward = RPE + reward_noise
        dW = eligibility * combined_reward
        dW_cum = np.cumsum(dW)
        if NORMALIZED_DW:
            norm_factor = np.sum(np.abs(dW))
            dW_cum /= norm_factor
        dW_combined.append(dW_cum)

    mean_combined = np.mean(dW_combined, axis=0)
    std_combined = np.std(dW_combined, axis=0)
    return mean_combined, std_combined


def _is_default_config(cfg: ExperimentConfig, default_cfg: ExperimentConfig) -> bool:
    return (
        cfg.learning_rate == default_cfg.learning_rate
        and cfg.min_noise_std == default_cfg.min_noise_std
        and cfg.reward_noise_rate == default_cfg.reward_noise_rate
        and cfg.reward_noise_std == default_cfg.reward_noise_std
        and cfg.input_rate == default_cfg.input_rate
        and cfg.RPE_decay_tau == default_cfg.RPE_decay_tau
    )


def _matches_single_variation(
    cfg: ExperimentConfig,
    default_cfg: ExperimentConfig,
    attr_name: str,
    attr_value: float,
) -> bool:
    if getattr(cfg, attr_name) != attr_value:
        return False
    for other_attr in (
        "learning_rate",
        "min_noise_std",
        "reward_noise_rate",
        "reward_noise_std",
        "input_rate",
        "RPE_decay_tau",
    ):
        if other_attr == attr_name:
            continue
        if getattr(cfg, other_attr) != getattr(default_cfg, other_attr):
            return False
    return True


def _build_parameter_search_index(
    configs: list[ExperimentConfig],
) -> tuple[list[tuple[str, str, list[float]]], dict[tuple[str, float], int]]:
    default_cfg = ExperimentConfig()
    default_idx = None
    for idx, cfg in enumerate(configs):
        if _is_default_config(cfg, default_cfg):
            default_idx = idx
            break
    if default_idx is None:
        raise RuntimeError(
            "Could not find the default configuration in generated configs."
        )

    rows = [
        ("reward_noise_rate", "rnl", [0.0, default_cfg.reward_noise_rate, 4.0]),
        ("min_noise_std", "mns", [2e-9, default_cfg.min_noise_std, 5e-9]),
        ("reward_noise_std", "rns", [0.5, default_cfg.reward_noise_std, 2.0]),
        ("input_rate", "ir", [25.0, default_cfg.input_rate, 100.0]),
        ("RPE_decay_tau", "tau", [0.01, default_cfg.RPE_decay_tau, 0.1]),
    ]

    cfg_index_for_panel: dict[tuple[str, float], int] = {}
    for attr_name, _, values in rows:
        for value in values:
            if value == getattr(default_cfg, attr_name):
                cfg_index_for_panel[(attr_name, value)] = default_idx
                continue

            matches = [
                idx
                for idx, cfg in enumerate(configs)
                if _matches_single_variation(cfg, default_cfg, attr_name, value)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one config for {attr_name}={value}, found {len(matches)}"
                )
            cfg_index_for_panel[(attr_name, value)] = matches[0]

    return rows, cfg_index_for_panel


def print_parameter_search_noise_std_ratios(runs: list[RunRecord]):
    runs = [run for run in runs if run.cfg_idx is not None]
    if not runs:
        raise RuntimeError(
            "print_parameter_search_noise_std_ratios requires cfg-indexed result files generated by the current launch.py naming scheme."
        )

    configs = generate_experiment_configs()
    rows, cfg_index_for_panel = _build_parameter_search_index(configs)

    grouped: dict[tuple[int, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.cfg_idx, run.run_type)].append(run)

    for row_idx, (attr_name, attr_label, values) in enumerate(rows):
        for value in values:
            cfg_idx = cfg_index_for_panel[(attr_name, value)]
            default_records = grouped[(cfg_idx, "Default")]
            gated_records = grouped[(cfg_idx, "Gated")]

            _, _, _, _, default_std_noise = mean_dw_split(default_records)
            _, _, _, _, gated_std_noise = mean_dw_split(gated_records)
            valid = default_std_noise != 0.0
            ratio_trace = np.divide(
                gated_std_noise,
                default_std_noise,
                out=np.full_like(gated_std_noise, np.nan, dtype=float),
                where=valid,
            )
            finite_ratio = ratio_trace[np.isfinite(ratio_trace)]
            if finite_ratio.size == 0:
                final_ratio = np.nan
                avg_ratio = np.nan
            else:
                final_ratio = finite_ratio[-1]
                avg_ratio = np.mean(finite_ratio)

            print(
                f"Std ratio gated vs default for {attr_label}={value:g}: "
                f"final = {final_ratio:.6g} | avg = {avg_ratio:.6g}"
            )

        if row_idx < len(rows) - 1:
            print("-" * 73)


def print_parameter_search_relative_SNR(runs: list[RunRecord]):
    runs = [run for run in runs if run.cfg_idx is not None]
    if not runs:
        raise RuntimeError(
            "print_parameter_search_relative_SNR requires cfg-indexed result files generated by the current launch.py naming scheme."
        )

    configs = generate_experiment_configs()
    rows, cfg_index_for_panel = _build_parameter_search_index(configs)

    grouped: dict[tuple[int, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.cfg_idx, run.run_type)].append(run)

    for row_idx, (attr_name, attr_label, values) in enumerate(rows):
        for value in values:
            cfg_idx = cfg_index_for_panel[(attr_name, value)]
            default_records = grouped[(cfg_idx, "Default")]
            gated_records = grouped[(cfg_idx, "Gated")]

            default_mean, default_std = combined(default_records)
            gated_mean, gated_std = combined(gated_records)

            valid = default_mean != 0.0
            ratio_trace = np.divide(
                gated_mean,
                default_mean,
                out=np.full_like(gated_mean, np.nan, dtype=float),
                where=valid,
            )
            finite_ratio = ratio_trace[np.isfinite(ratio_trace)]
            if finite_ratio.size == 0:
                final_ratio = np.nan
                avg_ratio = np.nan
            else:
                final_ratio = finite_ratio[-1]
                avg_ratio = np.mean(finite_ratio)

            print(
                f"Relative SNR for {attr_label}={value:g}: "
                f"final = {final_ratio:.6g} | avg = {avg_ratio:.6g}"
            )

        if row_idx < len(rows) - 1:
            print("-" * 73)


def plot_parameter_search(
    runs: list[RunRecord],
    output_path: Path | None,
    show: bool,
    plot_kind: str,
):
    runs = [run for run in runs if run.cfg_idx is not None]
    if not runs:
        raise RuntimeError(
            "plot_parameter_search requires cfg-indexed result files generated by the current launch.py naming scheme."
        )

    configs = generate_experiment_configs()
    rows, cfg_index_for_panel = _build_parameter_search_index(configs)

    grouped: dict[tuple[int, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.cfg_idx, run.run_type)].append(run)

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(16.0, 20.0),
        constrained_layout=True,
    )
    colors = {"Default": "tab:blue", "Gated": "tab:orange"}

    for row_idx, (attr_name, attr_label, values) in enumerate(rows):
        for col_idx, value in enumerate(values):
            ax = axes[row_idx, col_idx]
            cfg_idx = cfg_index_for_panel[(attr_name, value)]
            has_data = False

            for run_type in ("Default", "Gated"):
                records = grouped[(cfg_idx, run_type)]
                if not records:
                    continue
                has_data = True
                if plot_kind == "environment":
                    start_idx = records[0].times.searchsorted(100.0)
                    times = records[0].times[start_idx:]
                    traces = np.vstack(
                        [
                            record.state.environment_state[start_idx:, 0]
                            - record.state.environment_state[start_idx:, 1]
                            for record in records
                        ]
                    )
                    mean_trace = np.mean(traces, axis=0)
                    std_trace = np.std(traces, axis=0)
                    ax.plot(
                        times,
                        mean_trace,
                        color=colors[run_type],
                        linewidth=2.0,
                        label=f"{run_type} (n={len(records)})",
                    )
                    ax.fill_between(
                        times,
                        mean_trace - std_trace,
                        mean_trace + std_trace,
                        color=colors[run_type],
                        alpha=0.2,
                        label="_nolegend_",
                    )
                elif plot_kind == "dw_split":
                    times, mean_signal, mean_noise, std_signal, std_noise = (
                        mean_dw_split(records)
                    )
                    mean_combined, std_combined = combined(records)
                    # total_run_time = times[-1] - times[0]
                    # compression_ratio = (total_run_time / 1e-4) / len(times)
                    # scale = records[0].learning_rate * compression_ratio
                    scale = 1
                    ax.plot(
                        times,
                        mean_signal * scale,
                        color=colors[run_type],
                        linestyle="-",
                        linewidth=2.0,
                        label=f"{run_type} signal",
                    )
                    ax.plot(
                        times,
                        mean_noise * scale,
                        color=colors[run_type],
                        linestyle="--",
                        linewidth=2.0,
                        label=f"{run_type} noise",
                    )
                    ax.fill_between(
                        times,
                        (mean_signal - std_signal) * scale,
                        (mean_signal + std_signal) * scale,
                        color=colors[run_type],
                        alpha=0.2,
                        label="_nolegend_",
                    )
                    ax.fill_between(
                        times,
                        (mean_noise - std_noise) * scale,
                        (mean_noise + std_noise) * scale,
                        color=colors[run_type],
                        alpha=0.2,
                        label="_nolegend_",
                    )
                    # Also plot combined
                    ax.plot(
                        times,
                        mean_combined * scale,
                        color=colors[run_type],
                        linestyle=":",
                        linewidth=2.0,
                        label=f"{run_type} combined",
                    )
                    ax.fill_between(
                        times,
                        (mean_combined - std_combined) * scale,
                        (mean_combined + std_combined) * scale,
                        color=colors[run_type],
                        alpha=0.2,
                        label="_nolegend_",
                    )
                elif plot_kind == "weights":
                    times, mean_trace, std_trace = mean_w_over_keys(records)
                    ax.plot(
                        times,
                        mean_trace,
                        color=colors[run_type],
                        linewidth=2.0,
                        label=f"{run_type} (n={len(records)})",
                    )
                    ax.fill_between(
                        times,
                        mean_trace - std_trace,
                        mean_trace + std_trace,
                        color=colors[run_type],
                        alpha=0.2,
                        label="_nolegend_",
                    )
                else:
                    raise ValueError(f"Unknown plot_kind: {plot_kind}")

            default_cfg = ExperimentConfig()
            is_default_column = value == getattr(default_cfg, attr_name)
            default_suffix = " (default)" if is_default_column else ""
            ax.set_title(f"{attr_label}={value:g}{default_suffix}")
            ax.set_xlabel("time")
            ax.grid(alpha=0.3)
            if has_data:
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    alpha=0.7,
                )

    if plot_kind == "environment":
        axes[0, 0].set_ylabel("Error")
        fig.suptitle(
            "Parameter search: environment state (mean over key seeds)", fontsize=14
        )
    else:
        axes[0, 0].set_ylabel("Weight contribution")
        fig.suptitle("Parameter search: dW split (mean over key seeds)", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_runs(
    runs: list[RunRecord],
    output_path: Path | None,
    show: bool,
):
    grouped: dict[tuple[float, float, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.reward_noise_level, run.learning_rate, run.run_type)].append(run)

    pairs = sorted({(run.reward_noise_level, run.learning_rate) for run in runs})
    if not pairs:
        raise RuntimeError("No hyperparameter pairs found in loaded runs.")

    n_plots = len(pairs)
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    colors = {"Default": "tab:blue", "Gated": "tab:orange"}

    lrs = [100, 200, 400]
    lrs_extended = [50.0] + lrs
    for ax, (rnl, lr) in zip(axes, pairs):
        for run_type in ("Default", "Gated"):
            if lr not in lrs:
                continue  # Skip cases with no learning or excessively high learning rate that causes divergence
            if run_type == "Gated":
                lr_index = np.nonzero(np.isclose(lrs, lr))[0][0]
                lr = lrs_extended[lr_index]
            key = (rnl, lr, run_type)
            if key not in grouped:
                continue
            records = grouped[key]
            if PLOT_INDIVIDUAL_RUNS:
                for record in records:
                    ax.plot(
                        record.times,
                        extract_weight_trace(record.state),
                        color=colors[run_type],
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.1,
                        label="_nolegend_",
                    )

            times, mean_trace, std_trace = mean_w_over_keys(records)
            ax.plot(
                times,
                mean_trace,
                color=colors[run_type],
                linestyle="-",
                linewidth=2.0,
                label=f"{run_type} (n={len(grouped[key])})",
            )
            ax.fill_between(
                times,
                mean_trace - std_trace,
                mean_trace + std_trace,
                color=colors[run_type],
                alpha=0.2,
                label="_nolegend_",
            )

        ax.set_title(f"rnl={rnl:g}, lr={lr:g}")
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    axes[0].set_ylabel("Weight")
    fig.suptitle("Gated vs eligibility (mean over key seeds)", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dw_split(runs: list[RunRecord], output_path: Path | None, show: bool):
    grouped: dict[tuple[float, float, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.reward_noise_level, run.learning_rate, run.run_type)].append(run)

    pairs = sorted({(run.reward_noise_level, run.learning_rate) for run in runs})
    if not pairs:
        raise RuntimeError("No hyperparameter pairs found in loaded runs.")

    n_plots = len(pairs)
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    colors = {"Default": "tab:blue", "Gated": "tab:orange"}

    lrs = [100, 200, 400]
    lrs_extended = [50.0] + lrs
    for ax, (rnl, lr) in zip(axes, pairs):
        for run_type in ("Default", "Gated"):
            if lr not in lrs:
                continue  # Skip cases with no learning or excessively high learning rate that causes divergence
            if run_type == "Gated":
                lr_index = np.nonzero(np.isclose(lrs, lr))[0][0]
                lr = lrs_extended[lr_index]
            key = (rnl, lr, run_type)
            if key not in grouped:
                continue
            records = grouped[key]
            if PLOT_INDIVIDUAL_RUNS:
                for record in records:
                    dW_signal, dW_noise = compute_dw_split(record)
                    ax.plot(
                        record.times,
                        dW_signal,
                        color=colors[run_type],
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.1,
                        label="_nolegend_",
                    )
                    ax.plot(
                        record.times,
                        dW_noise,
                        color=colors[run_type],
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.1,
                        label="_nolegend_",
                    )

            times, mean_signal, mean_noise, std_signal, std_noise = mean_dw_split(
                records
            )
            # total_run_time = times[-1] - times[0]
            # compression_ratio = (total_run_time / 1e-4)  / len(times) # How many samples we have vs how many steps the sim was
            compression_ratio = 1
            ax.plot(
                times,
                mean_signal * compression_ratio,
                color=colors[run_type],
                linestyle="-",
                linewidth=2.0,
                label=f"{run_type} signal",
            )
            ax.plot(
                times,
                mean_noise * compression_ratio,
                color=colors[run_type],
                linestyle="--",
                linewidth=2.0,
                label=f"{run_type} noise",
            )
            ax.fill_between(
                times,
                (mean_signal - std_signal) * compression_ratio,
                (mean_signal + std_signal) * compression_ratio,
                color=colors[run_type],
                alpha=0.2,
                label="_nolegend_",
            )
            ax.fill_between(
                times,
                (mean_noise - std_noise) * compression_ratio,
                (mean_noise + std_noise) * compression_ratio,
                color=colors[run_type],
                alpha=0.2,
                label="_nolegend_",
            )
            combined_mean, combined_std = combined(records)
            ax.plot(
                times,
                combined_mean * compression_ratio,
                color=colors[run_type],
                linestyle=":",
                linewidth=2.0,
                label=f"{run_type} combined",
            )
            ax.fill_between(
                times,
                (combined_mean - combined_std) * compression_ratio,
                (combined_mean + combined_std) * compression_ratio,
                color=colors[run_type],
                alpha=0.2,
                label="_nolegend_",
            )

        ax.set_title(f"rnl={rnl:g}, lr={lr:g}")
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    axes[0].set_ylabel("Weight")
    fig.suptitle("Gated vs eligibility (mean over key seeds)", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_environment(
    runs: list[RunRecord],
    output_path: Path | None,
    show: bool,
):
    grouped: dict[tuple[float, float, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.reward_noise_level, run.learning_rate, run.run_type)].append(run)

    pairs = sorted({(run.reward_noise_level, run.learning_rate) for run in runs})
    if not pairs:
        raise RuntimeError("No hyperparameter pairs found in loaded runs.")

    n_plots = len(pairs)
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    colors = {"Default": "tab:blue", "Gated": "tab:orange"}

    start_plotting_index = runs[0].times.searchsorted(
        100.0
    )  # Skip first 100 seconds to focus on learning phase

    lrs = [100, 200, 400]
    lrs_extended = [50.0] + lrs
    for ax, (rnl, lr) in zip(axes, pairs):
        for run_type in ("Default", "Gated"):
            if lr not in lrs:
                continue  # Skip cases with no learning or excessively high learning rate that causes divergence
            if run_type == "Gated":
                lr_index = np.nonzero(np.isclose(lrs, lr))[0][0]
                lr = lrs_extended[lr_index]
            key = (rnl, lr, run_type)
            if key not in grouped:
                continue
            records = grouped[key]
            if PLOT_INDIVIDUAL_RUNS:
                for record in records:
                    ax.plot(
                        record.times[start_plotting_index:],
                        record.state.environment_state[start_plotting_index:, 0]
                        - record.state.environment_state[start_plotting_index:, 1],
                        color=colors[run_type],
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.1,
                        label="_nolegend_",
                    )

            times = records[0].times[start_plotting_index:]
            mean_trace = np.mean(
                [
                    record.state.environment_state[start_plotting_index:, 0]
                    - record.state.environment_state[start_plotting_index:, 1]
                    for record in records
                ],
                axis=0,
            )
            std_trace = np.std(
                [
                    record.state.environment_state[start_plotting_index:, 0]
                    - record.state.environment_state[start_plotting_index:, 1]
                    for record in records
                ],
                axis=0,
            )

            ax.plot(
                times,
                mean_trace,
                color=colors[run_type],
                linestyle="-",
                linewidth=2.0,
                label=f"{run_type} (n={len(grouped[key])})",
            )
            ax.fill_between(
                times,
                mean_trace - std_trace,
                mean_trace + std_trace,
                color=colors[run_type],
                alpha=0.2,
                label="_nolegend_",
            )

        ax.set_title(f"lr={lr:g}, rnl={rnl:g}")
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    axes[0].set_ylabel("Error")
    fig.suptitle("Gated vs Default (mean over key seeds)", fontsize=14)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dw_histogram(
    runs: list[RunRecord],
    output_path: Path | None,
    show: bool,
):
    grouped: dict[tuple[float, float, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        if run.reward_noise_level == 0.0:
            continue
        grouped[(run.reward_noise_level, run.learning_rate, run.run_type)].append(run)

    pairs = sorted(
        {
            (run.reward_noise_level, run.learning_rate)
            for run in runs
            if run.reward_noise_level != 0.0
        }
    )
    if not pairs:
        raise RuntimeError("No hyperparameter pairs found in loaded runs.")

    n_plots = len(pairs)
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.0 * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    colors = {"Default": "tab:blue", "Gated": "tab:orange"}

    for ax, (rnl, lr) in zip(axes, pairs):
        for run_type in ("Default", "Gated"):
            key = (rnl, lr, run_type)
            if key not in grouped:
                continue
            records = grouped[key]
            all_dW = None
            all_dW_task = None
            all_dW_noise = None
            start = records[0].times.searchsorted(0.0)  # Focus on learning phase
            end = records[0].times.searchsorted(100.0)  # Focus on learning phase
            for record in records:
                state = record.state
                eligibility = (
                    state.agent_state.noisy_network.network_state.features.eligibility[
                        start:end, 1, -1
                    ].squeeze()
                )
                reward_noise = state.agent_state.reward_noise.squeeze()[start:end]
                RPE = state.agent_state.RPE.squeeze()[start:end]
                combined_reward = RPE + reward_noise
                dW = eligibility * combined_reward
                dW_task = eligibility * RPE
                dW_noise = eligibility * reward_noise
                all_dW = dW if all_dW is None else np.vstack((all_dW, dW))
                all_dW_task = (
                    dW_task
                    if all_dW_task is None
                    else np.vstack((all_dW_task, dW_task))
                )
                all_dW_noise = (
                    dW_noise
                    if all_dW_noise is None
                    else np.vstack((all_dW_noise, dW_noise))
                )

            flattened_dW = all_dW_task.flatten()
            dw_outliers = np.abs(flattened_dW) > 0.005
            flattened_dW = flattened_dW[~dw_outliers]

            ax.hist(
                flattened_dW,
                bins=51,
                color=colors[run_type],
                alpha=0.5,
                label=f"{run_type} combined",
                density=True,
            )
            flattened_dW_noise = all_dW_noise.flatten()
            dw_noise_outliers = np.abs(flattened_dW_noise) > 0.005
            flattened_dW_noise = flattened_dW_noise[~dw_noise_outliers]

            ax.hist(
                flattened_dW_noise,
                bins=51,
                color=colors[run_type],
                alpha=0.5,
                label=f"{run_type} noise",
                density=True,
                facecolor="none",
                linestyle="--",
                histtype="step",
            )
            ax.set_title(f"lr={lr:g}, rnl={rnl:g}")
            ax.set_xlabel("dW")
            # ax.set_ylim(0, 300)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_yscale("log")
            ax.set_xscale("linear")
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_pane(results):
    to_plot = [
        {"run_type": "Default", "lr": 400, "rnl": 2.0},
        {"run_type": "Gated", "lr": 200, "rnl": 2.0},
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for item in to_plot:
        matching_runs = [
            run
            for run in results
            if run.run_type == item["run_type"]
            and np.isclose(run.learning_rate, item["lr"])
            and np.isclose(run.reward_noise_level, item["rnl"])
        ]
        if not matching_runs:
            print(f"No runs found for {item}")
            continue

        start_plotting_index = matching_runs[0].times.searchsorted(100.0)

        mean_trace = np.mean(
            [
                np.abs(
                    record.state.environment_state[start_plotting_index:, 0]
                    - record.state.environment_state[start_plotting_index:, 1]
                )
                ** 2
                for record in matching_runs
            ],
            axis=0,
        )
        std_trace = np.std(
            [
                np.abs(
                    record.state.environment_state[start_plotting_index:, 0]
                    - record.state.environment_state[start_plotting_index:, 1]
                )
                ** 2
                for record in matching_runs
            ],
            axis=0,
        )

        times = matching_runs[0].times[start_plotting_index:]

        color = "#A5A2A2" if item["run_type"] == "Default" else "#027254"
        ax.plot(
            times,
            mean_trace,
            color=color,
            linestyle="-",
            linewidth=2.0,
            label=f"{item['run_type']} (lr={item['lr']})",
        )
        ax.fill_between(
            times,
            mean_trace - std_trace,
            mean_trace + std_trace,
            color=color,
            alpha=0.2,
            label="_nolegend_",
            linewidth=0,
        )
        ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error")
    plt.show()


def plot_delta_v_tuning(runs: list[RunRecord], output_path: Path | None, show: bool):
    grouped: dict[tuple[float, float, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.delta_v)].append(run)

    pairs = sorted({(run.delta_v) for run in runs})
    if not pairs:
        raise RuntimeError("No hyperparameter pairs found in loaded runs.")

    # n_plots = len(pairs)
    # ncols = min(4, n_plots)
    # nrows = math.ceil(n_plots / ncols)
    # fig, axes = plt.subplots(
    # 	nrows,
    # 	ncols,
    # 	figsize=(5.0 * ncols, 4.0 * nrows),
    # 	sharey=True,
    # 	constrained_layout=True,
    # )
    # axes = np.atleast_1d(axes).ravel()
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.0))

    colors = {"Gated": "#027254"}

    max_dv = 0.0025
    # for ax, (dv) in zip(axes, pairs):
    for i, dv in enumerate(pairs):
        for run_type in ["Gated"]:
            key = dv
            if key not in grouped:
                continue
            records = grouped[key]
            all_dW = None
            all_dW_noise = None
            for record in records:
                state = record.state
                eligibility = (
                    state.agent_state.noisy_network.network_state.features.eligibility[
                        :, 1, -1
                    ].squeeze()
                )
                reward_noise = state.agent_state.reward_noise.squeeze()
                RPE = state.agent_state.RPE.squeeze()
                dW = eligibility * RPE
                dW_noise = eligibility * reward_noise

                all_dW = dW if all_dW is None else np.vstack((all_dW, dW))
                all_dW_noise = (
                    dW_noise
                    if all_dW_noise is None
                    else np.vstack((all_dW_noise, dW_noise))
                )

            flattened_dW = all_dW.flatten()
            dw_outliers = np.abs(flattened_dW) > max_dv
            flattened_dW = flattened_dW[~dw_outliers]
            n, bins, patches = axs[0].hist(
                flattened_dW,
                bins=51,
                color=colors[run_type],
                alpha=1 - i * 0.15,
                density=False,
                histtype="step",
                label=r"$\Delta v={:.4f}$".format(dv),
            )
            axs[0].set_xlabel("dW")
            axs[0].set_ylabel("Count (log)")
            axs[0].set_yscale("log")
            axs[0].set_xlim(-max_dv, max_dv)
            axs[0].grid(alpha=0.3)
            axs[0].legend(fontsize=5, loc="upper right")

            # Plot noise component histogram
            flattened_dW_noise = all_dW_noise.flatten()
            dw_noise_outliers = np.abs(flattened_dW_noise) > max_dv
            flattened_dW_noise = flattened_dW_noise[~dw_noise_outliers]
            n_noise, bins_noise, patches_noise = axs[1].hist(
                flattened_dW_noise,
                bins=51,
                color=colors[run_type],
                alpha=1 - i * 0.15,
                density=False,
                histtype="step",
                label=r"$\Delta v={:.4f}$".format(dv),
            )
            axs[1].set_xlabel("dW noise component")
            axs[1].set_ylabel("Count (log)")
            axs[1].set_yscale("log")
            axs[1].set_xlim(-max_dv, max_dv)
            axs[1].grid(alpha=0.3)
            axs[1].legend(fontsize=5, loc="upper right")

            print(
                f"Task Delta_v={dv:.4f}: mean={np.mean(flattened_dW):.6g}, std={np.std(flattened_dW):.6g}, n={flattened_dW.size}"
            )
            print(
                f"Noisy Delta_v={dv:.4f}: mean={np.mean(flattened_dW_noise):.6g}, std={np.std(flattened_dW_noise):.6g}, n={flattened_dW_noise.size}"
            )
    # for ax in axes[n_plots:]:
    # 	ax.set_visible(False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to {output_path}")


def main():
    results_dir = resolve_results_dir(RESULTS_DIR)
    runs = load_runs(results_dir)

    plot_delta_v_tuning(
        runs=runs,
        output_path=OUTPUT_PATH.with_name(
            OUTPUT_PATH.stem + "_delta_v_tuning" + OUTPUT_PATH.suffix
        ),
        show=SHOW_FIGURE,
    )
    # print_parameter_search_noise_std_ratios(runs)
    # print_parameter_search_relative_SNR(runs)
    # plot_runs(runs=runs, output_path=OUTPUT_PATH, show=SHOW_FIGURE)
    # plot_dw_split(runs=runs, output_path=OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_dw_split" + OUTPUT_PATH.suffix), show=SHOW_FIGURE)
    # plot_environment(runs=runs, output_path=OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_environment_lr_matched" + OUTPUT_PATH.suffix), show=SHOW_FIGURE)
    # plot_dw_histogram(runs=runs, output_path=OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_dw_histogram" + OUTPUT_PATH.suffix), show=SHOW_FIGURE)
    # plot_single_pane(runs)
    # try:
    # 	plot_parameter_search(
    # 		runs=runs,
    # 		output_path=OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_parameter_search_environment" + OUTPUT_PATH.suffix),
    # 		show=SHOW_FIGURE,
    # 		plot_kind="environment",
    # 	)
    # 	plot_parameter_search(
    # 		runs=runs,
    # 		output_path=OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_parameter_search_dw_split" + OUTPUT_PATH.suffix),
    # 		show=SHOW_FIGURE,
    # 		plot_kind="dw_split",
    # 	)
    # 	plot_parameter_search(
    # 		runs=runs,
    # 		output_path=OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_parameter_search_weights" + OUTPUT_PATH.suffix),
    # 		show=SHOW_FIGURE,
    # 		plot_kind="weights",
    # 	)
    # except RuntimeError as exc:
    # 	print(f"Skipping parameter-search plots: {exc}")


if __name__ == "__main__":
    main()
