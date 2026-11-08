import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from adaptive_snn.utils.save_helper import load_named_result

RESULTS_DIR = "results/pendulum_main_run_final_fresh_seeds_20261104_083430/results"
FIGURES_DIR = "figures/pendulum_main_run_final_fresh_seeds_20261104_083430"

DELTA_V_ORDER = ["None", "-5", "-7", "-9", "-11", "-13", "-15", "inf"]
NOISE_FLOOR_ORDER = [2e-08, 5e-09]
NOISE_FLOOR_LABELS = {2e-08: r"$\sigma_\xi = 20 nS$", 5e-09: r"$\sigma_\xi = 5 nS$"}
NOISE_FLOOR_COLORS = {2e-08: "#2a78d6", 5e-09: "#eb6834"}
# Fixed color per delta_V so a value keeps its color across figures.
DELTA_V_COLORS = {
    "None": "#3d3d3a",
    "-15": "#0aa5b8",
    "-13": "#2a78d6",
    "-11": "#eb6834",
    "-9": "#1baf7a",
    "-7": "#eda100",
    "-5": "#e87ba4",
    "inf": "#6250d6",
}


def lr_shade(color, lr, lrs):
    """Shade of `color` for learning rate `lr`: from light (smallest lr) to
    dark (largest lr), with the base color in the middle (and for a single lr)."""
    frac = lrs.index(lr) / (len(lrs) - 1) if len(lrs) > 1 else 0.5
    mix = 0.5 - 1.0 * frac  # > 0: blend towards white, < 0: blend towards black
    if mix >= 0:
        return tuple(c + (1 - c) * mix for c in to_rgb(color))
    return tuple(c * (1 + mix) for c in to_rgb(color))


def make_lr_label(lr):
    return f"lr={lr:g}"


def make_delta_V_label(delta_v):
    if delta_v == "None":
        return "default"
    elif delta_v == "inf":
        return "spike-gated"
    else:
        return rf"$\Delta V=2^{{{delta_v}}}$"


def parse_file_params(path):
    name = Path(path).name
    delta_v = (
        m.group(1) if (m := re.search(r"_delta_V_(inf|None|-?\d+)_", name)) else None
    )
    noise_floor = (
        float(m.group(1))
        if (m := re.search(r"_noise_floor_([0-9.e+-]+)_", name))
        else None
    )
    lr = float(m.group(1)) if (m := re.search(r"_lr_([0-9.]+)_", name)) else None
    seed = int(m.group(1)) if (m := re.search(r"_seed_(\d+)", name)) else None
    return delta_v, noise_floor, lr, seed


def window_stats(run, window):
    """Average episode duration, reward, recurrent weight and hidden rate in
    non-overlapping windows of `window` seconds. The first row is the raw
    instantaneous state at t=0 (rather than the first window's average), so
    curves start at the true initial value instead of at t=window/2."""
    ts = run["ts"]
    stride = max(int(round(window / (ts[1] - ts[0]))), 1)
    idx = np.arange(0, len(ts), stride)
    n_ep = run["episode_count"].astype(np.float64).squeeze()
    len_sum = run["episode_length_sum"].astype(np.float64).squeeze()
    reward = run["mean_reward"].astype(np.float64).squeeze()
    W_recurrent = run["mean_W_actor_recurrent"].astype(np.float64).squeeze()
    rate_H = run["mean_filtered_spikes_H"].astype(np.float64).squeeze()
    balance = run["mean_balance"].astype(np.float64).squeeze()

    t = [float(ts[0])]
    episode_duration = [np.nan]
    mean_reward = [float(reward[0])]
    mean_W_actor_recurrent = [float(W_recurrent[0])]
    rate_H_list = [float(rate_H[0])]
    mean_balance = [float(balance[0])]
    for a, b in zip(idx[:-1], idx[1:]):
        d_ep = n_ep[b] - n_ep[a]
        t.append(0.5 * (ts[a] + ts[b]))
        episode_duration.append(
            (len_sum[b] - len_sum[a]) / d_ep if d_ep > 0 else np.nan
        )
        mean_reward.append(np.mean(reward[a:b]))
        mean_W_actor_recurrent.append(np.mean(W_recurrent[a:b]))
        rate_H_list.append(np.mean(rate_H[a:b]))
        mean_balance.append(np.nanmean(balance[a:b]))
    return {
        "t": np.array(t),
        "episode_duration": np.array(episode_duration),
        "mean_reward": np.array(mean_reward),
        "mean_W_actor_recurrent": np.array(mean_W_actor_recurrent),
        "rate_H": np.array(rate_H_list),
        "mean_balance": np.array(mean_balance),
    }


def load_chunked_run(paths):
    """Concatenate chunks of one run in order. Each chunk after the first starts
    with the previous chunk's final sample, so that duplicate is dropped."""
    chunks = [load_named_result(p) for p in paths]
    return {
        k: np.concatenate([c[k] if i == 0 else c[k][1:] for i, c in enumerate(chunks)])
        for k in chunks[0]
    }


def load_all_stats(results_dir, window):
    """Return stats[delta_v][noise_floor][lr] = list of per-seed window_stats dicts,
    containing only the delta_V / noise floor / lr combinations present in the data."""
    stats = {}
    runs = {}
    for path in glob.glob(str(Path(results_dir) / "*.npz")):
        m = re.match(r"(.*)_chunk_(\d+)\.npz$", path)
        runs.setdefault(m.group(1), []).append((int(m.group(2)), path))
    for run_name, chunk_paths in sorted(runs.items()):
        delta_v, noise_floor, lr, _seed = parse_file_params(run_name)
        if delta_v not in DELTA_V_ORDER or noise_floor not in NOISE_FLOOR_ORDER:
            continue
        print(f"Loading {Path(run_name).name} ({len(chunk_paths)} chunks)")
        run = load_chunked_run([p for _, p in sorted(chunk_paths)])
        stats.setdefault(delta_v, {}).setdefault(noise_floor, {}).setdefault(
            lr, []
        ).append(window_stats(run, window))
    return stats


def present_params(stats):
    """Delta_V values, noise floors and lrs present in `stats`, in plotting order.
    lrs are per delta_V, so shading only distinguishes lrs within one delta_V."""
    delta_vs = [dv for dv in DELTA_V_ORDER if dv in stats]
    noise_floors = [
        nf for nf in NOISE_FLOOR_ORDER if any(nf in stats[dv] for dv in delta_vs)
    ]
    lrs = {
        dv: sorted({lr for nf in stats[dv] for lr in stats[dv][nf]}) for dv in delta_vs
    }
    return delta_vs, noise_floors, lrs


def average_stats(stats_list, key):
    """Average a stat across seeds on the shared time grid (all runs share dt/window)."""
    values = np.stack([s[key] for s in stats_list])
    return stats_list[0]["t"], np.nanmean(values, axis=0)


def style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_summary(stats, out_dir, window):
    delta_vs, noise_floors, lrs = present_params(stats)
    panels = [
        ("episode_duration", "mean episode duration (s)"),
        ("mean_reward", "mean reward"),
    ]
    fig, axs = plt.subplots(
        len(noise_floors),
        len(panels),
        figsize=(9, 3.25 * len(noise_floors)),
        sharex=True,
        squeeze=False,
    )

    for row, noise_floor in enumerate(noise_floors):
        for col, (key, ylabel) in enumerate(panels):
            ax = axs[row, col]
            for delta_v in delta_vs:
                for lr in lrs[delta_v]:
                    seed_stats = stats[delta_v].get(noise_floor, {}).get(lr)
                    if not seed_stats:
                        continue
                    t, values = average_stats(seed_stats, key)
                    ax.plot(
                        t,
                        values,
                        lw=1.4,
                        color=lr_shade(DELTA_V_COLORS[delta_v], lr, lrs[delta_v]),
                        label=f"{make_delta_V_label(delta_v)}, {make_lr_label(lr)}",
                    )
            style(ax, ylabel)
            ax.legend(fontsize=8, frameon=False, loc="best")

    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, 2000)

    fig.suptitle(
        f"Episode duration and reward vs. $\\Delta V$ (window={window:g} s, averaged over seeds)"
    )
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "delta_V_noise_summary.pdf")
    fig.savefig(out_dir / "delta_V_noise_summary.png", dpi=200)
    plt.show()


def plot_seed_detail(stats, out_dir, window):
    """Per-seed detail: one column per delta_V, one row per quantity, with
    all seeds shown individually (colored by noise floor, not averaged)."""
    rows = [
        ("episode_duration", "mean episode duration (s)"),
        ("mean_W_actor_recurrent", "mean W recurrent"),
        ("rate_H", "mean hidden pop. rate"),
        ("mean_balance", "mean balance"),
    ]
    delta_vs, noise_floors, lrs = present_params(stats)
    fig, axs = plt.subplots(
        len(rows),
        len(delta_vs),
        figsize=(max(3.2 * len(delta_vs), 5), 2.6 * len(rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for col, delta_v in enumerate(delta_vs):
        for row, (key, ylabel) in enumerate(rows):
            ax = axs[row, col]
            is_top_row = row == 0
            for noise_floor in noise_floors:
                for lr in lrs[delta_v]:
                    color = lr_shade(NOISE_FLOOR_COLORS[noise_floor], lr, lrs[delta_v])
                    seed_stats_list = stats[delta_v].get(noise_floor, {}).get(lr, [])
                    for seed_stats in seed_stats_list:
                        ax.plot(
                            seed_stats["t"],
                            seed_stats[key],
                            color=color,
                            lw=1.0,
                            alpha=0.3 if is_top_row else 0.85,
                        )
                    if is_top_row and seed_stats_list:
                        t, mean_values = average_stats(seed_stats_list, key)
                        ax.plot(t, mean_values, color=color, lw=1.8, alpha=1.0)
            style(ax, ylabel if col == 0 else "")
        axs[0, col].set_title(make_delta_V_label(delta_v), fontsize=10)

        legend_handles = [
            plt.Line2D(
                [],
                [],
                color=lr_shade(NOISE_FLOOR_COLORS[nf], lr, lrs[delta_v]),
                lw=1.4,
                label=f"{NOISE_FLOOR_LABELS[nf]}, {make_lr_label(lr)}",
            )
            for nf in noise_floors
            for lr in lrs[delta_v]
            if stats[delta_v].get(nf, {}).get(lr)
        ]
        for ax in axs[:, col]:
            ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="best")

    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(
        f"Per-seed episode duration, recurrent weight and hidden rate (window={window:g} s)"
    )
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "delta_V_noise_seed_detail.pdf")
    fig.savefig(out_dir / "delta_V_noise_seed_detail.png", dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--out_dir", default=FIGURES_DIR)
    parser.add_argument(
        "--window", type=float, default=50.0, help="averaging window (s)"
    )
    args = parser.parse_args()

    stats = load_all_stats(args.results_dir, args.window)
    plot_summary(stats, args.out_dir, args.window)
    plot_seed_detail(stats, args.out_dir, args.window)


if __name__ == "__main__":
    main()
