import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_snn.utils.save_helper import load_named_result

PARAM_PATTERNS = {
    "lr": r"_lr_([0-9.e+-]+)_",
    "balance_rate": r"_balance_rate_([0-9.e+-]+)_",
    "gradient_clip": r"_gradient_clip_([0-9.e+-]+|inf)_",
    "tau_charge": r"_tau_charge_([0-9.e+-]+)_",
    "noise_floor": r"_noise_floor_([0-9.e+-]+)_",
    "subtract_mean_perturbation": r"_subtract_mean_perturbation_(True|False)_",
    "tau_spike_filter": r"_tau_spike_filter_([0-9.e+-]+)_",
    "init_angle_deg": r"_init_angle_deg_([0-9.e+-]+)_",
    "delta_V": r"_delta_V_([0-9.e+-]+|inf)_",
    "key_seed": r"_key_seed_(\d+)_",
}
COLORS = ["#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def parse_params(path):
    # match on the file name only: the results folder name can contain e.g. "_lr_<date>_"
    name = Path(path).name
    params = {}
    for key, pattern in PARAM_PATTERNS.items():
        if m := re.search(pattern, name):
            v = m.group(1)
            params[key] = v == "True" if v in ("True", "False") else float(v)
    return params


def parse_info(path):
    """Read scalar settings from the _info.txt written next to the npz."""
    info = {}
    info_path = re.sub(r"\.npz$", "_info.txt", path)
    try:
        text = Path(info_path).read_text()
    except FileNotFoundError:
        return info
    for key in ("actor_warmup_time", "max_episode_time"):
        if m := re.search(rf"'?{key}'?: ([0-9.]+)", text):
            info[key] = float(m.group(1))
    return info


def format_value(key, v):
    if key == "noise_floor":
        return f"{v * 1e9:g} nS"
    if key == "tau_spike_filter":
        return f"filter {v * 1e3:g} ms"
    if key == "subtract_mean_perturbation":
        return "centred" if v else "uncentred"
    if key == "init_angle_deg":
        return f"init ±{v:g}°"
    if key == "delta_V":
        if np.isinf(v):
            return "spike-gated"
        # the sweep encodes delta_V as the base-2 exponent directly (delta_V = 2**v)
        return f"$\\Delta V=2^{{{v:.0f}}}$"
    if key == "lr":
        return f"lr {v:g}"
    if key == "key_seed":
        return f"seed {v:g}"
    return f"{key} {v:g}"


def make_labels(runs):
    keys = [
        k
        for k in PARAM_PATTERNS
        if len({str(r["params"].get(k)) for r in runs.values()}) > 1
    ]
    for cfg, r in runs.items():
        parts = [format_value(k, r["params"][k]) for k in keys if k in r["params"]]
        r["label"] = f"cfg {cfg}: " + ", ".join(parts) if parts else f"cfg {cfg}"
    return keys


def find_groups(results_dir):
    """Split the npz files in results_dir into (network, delta_V, seed, ...) groups.

    A single directory can hold several sweeps run side by side (e.g. one per network /
    delta_V combination), each internally indexed by cfg_0, cfg_1, ... A "group" is
    everything sharing the filename prefix that comes before "_cfg_<n>_".
    """
    groups = {}
    for path in sorted(glob.glob(str(Path(results_dir) / "*.npz"))):
        prefix = re.split(r"_cfg_\d+_", Path(path).name, maxsplit=1)[0]
        groups.setdefault(prefix, []).append(path)
    return groups


def group_title(prefix, path):
    m = re.search(r"([A-Za-z]+Network)", prefix)
    network = m.group(1) if m else prefix
    params = parse_params(path)
    if "delta_V" in params:
        return f"{network}, {format_value('delta_V', params['delta_V'])}"
    return network


def load_runs(paths):
    """Load all chunks per configuration (within one group) and concatenate in time.
    Each chunk after the first starts with the previous chunk's final sample, so that
    duplicate is dropped."""
    by_cfg = {}
    for path in sorted(paths):
        cfg = int(m.group(1)) if (m := re.search(r"_cfg_(\d+)", path)) else 0
        chunk = int(m.group(1)) if (m := re.search(r"_chunk_(\d+)", path)) else 0
        by_cfg.setdefault(cfg, []).append((chunk, path))
    runs = {}
    for cfg, chunks in sorted(by_cfg.items()):
        chunks.sort()
        results = [load_named_result(p) for _, p in chunks]
        run = {
            k: np.concatenate(
                [
                    np.asarray(r[k]) if i == 0 else np.asarray(r[k])[1:]
                    for i, r in enumerate(results)
                ]
            )
            for k in results[0]
        }
        runs[cfg] = {
            "run": run,
            "params": parse_params(chunks[0][1]),
            "info": parse_info(chunks[0][1]),
            "path": chunks[0][1],
        }
    make_labels(runs)
    return runs


def window_stats(run, window):
    """Decode the saved quantities in non-overlapping windows of `window` seconds."""
    ts = run["ts"]
    stride = max(int(round(window / (ts[1] - ts[0]))), 1)
    idx = np.arange(0, len(ts), stride)
    n_ep = run["episode_count"].astype(np.float64).squeeze()
    len_sum = run["episode_length_sum"].astype(np.float64).squeeze()
    rows = []
    # Instantaneous values at t=0, so the curves don't start at the first window's
    # centre (e.g. t=25 for a 50s window) but actually touch the true initial state.
    rows.append(
        {
            "t": float(ts[0]),
            "episode_duration": np.nan,
            "episode_rate": np.nan,
            "mean_reward": float(run["mean_reward"][0]),
            "mean_RPE": float(run["mean_RPE"][0]),
            "var_RPE": float(run["var_RPE"][0]),
            "map_LR_std": float(np.std(run["W_input_L_minus_R_map"][0])),
            "map_H_std": float(np.std(run["W_input_H_map"][0])),
            "rate_L": float(run["mean_filtered_spikes_L"][0]),
            "rate_R": float(run["mean_filtered_spikes_R"][0]),
            "rate_H": float(run["mean_filtered_spikes_H"][0]),
            "rate_std_L": float(np.sqrt(run["var_filtered_spikes_L"][0])),
            "rate_std_R": float(np.sqrt(run["var_filtered_spikes_R"][0])),
            "rate_std_H": float(np.sqrt(run["var_filtered_spikes_H"][0])),
            "mean_balance": float(run["mean_balance"][0]),
            "mean_W_actor_input": float(run["mean_W_actor_input"][0]),
            "std_W_actor_input": float(np.sqrt(run["var_W_actor_input"][0])),
            "mean_W_actor_recurrent": float(run["mean_W_actor_recurrent"][0]),
            "std_W_actor_recurrent": float(np.sqrt(run["var_W_actor_recurrent"][0])),
            "mean_W_critic": float(run["mean_W_critic"][0]),
        }
    )
    for a, b in zip(idx[:-1], idx[1:]):
        d_ep = n_ep[b] - n_ep[a]
        mean = lambda key: float(np.mean(run[key][a:b]))
        rows.append(
            {
                "t": 0.5 * (ts[a] + ts[b]),
                "episode_duration": (len_sum[b] - len_sum[a]) / d_ep
                if d_ep > 0
                else np.nan,
                "episode_rate": d_ep / (ts[b] - ts[a]),
                "mean_reward": mean("mean_reward"),
                "mean_RPE": mean("mean_RPE"),
                "var_RPE": mean("var_RPE"),
                "map_LR_std": float(
                    np.mean(np.std(run["W_input_L_minus_R_map"][a:b], axis=(1, 2)))
                ),
                "map_H_std": float(
                    np.mean(np.std(run["W_input_H_map"][a:b], axis=(1, 2)))
                ),
                "rate_L": mean("mean_filtered_spikes_L"),
                "rate_R": mean("mean_filtered_spikes_R"),
                "rate_H": mean("mean_filtered_spikes_H"),
                "rate_std_L": float(np.sqrt(mean("var_filtered_spikes_L"))),
                "rate_std_R": float(np.sqrt(mean("var_filtered_spikes_R"))),
                "rate_std_H": float(np.sqrt(mean("var_filtered_spikes_H"))),
                "mean_balance": float(np.nanmean(run["mean_balance"][a:b])),
                "mean_W_actor_input": mean("mean_W_actor_input"),
                "std_W_actor_input": float(np.sqrt(mean("var_W_actor_input"))),
                "mean_W_actor_recurrent": mean("mean_W_actor_recurrent"),
                "std_W_actor_recurrent": float(np.sqrt(mean("var_W_actor_recurrent"))),
                "mean_W_critic": mean("mean_W_critic"),
            }
        )
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def style(ax, ylabel, log=False):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if log:
        ax.set_yscale("log")


def mark_warmup(axs, runs):
    warmups = {r["info"].get("actor_warmup_time") for r in runs.values()} - {None}
    for ax in axs.flat:
        for w in warmups:
            ax.axvline(w, color="0.6", lw=0.8, ls=":")


def plot_panels(stats, runs, panels, shape, figsize, out, name, extra=None):
    fig, axs = plt.subplots(*shape, figsize=figsize, sharex=True)
    for ax, (key, label, log) in zip(axs.flat, panels):
        for cfg, s in stats.items():
            ax.plot(
                s["t"],
                s[key],
                color=COLORS[cfg % len(COLORS)],
                lw=1.2,
                label=runs[cfg]["label"],
            )
        style(ax, label, log)
    if extra is not None:
        extra(axs)
    mark_warmup(axs, runs)
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    axs.flat[0].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.show()
    plt.close(fig)


def plot_learning(stats, runs, out):
    panels = [
        ("episode_duration", "mean episode duration (s)", False),
        ("mean_reward", "mean reward", False),
        ("map_LR_std", "std of L$-$R input map", False),
        ("map_H_std", "std of hidden input map", False),
        ("mean_RPE", "mean RPE", False),
        ("var_RPE", "variance RPE", True),
    ]

    def extra(axs):
        max_ep = {r["info"].get("max_episode_time") for r in runs.values()} - {None}
        for m in max_ep:
            axs[0, 0].axhline(m, color="0.6", lw=0.8, ls="--")
        axs[0, 0].set_ylim(bottom=0)

    plot_panels(stats, runs, panels, (3, 2), (9, 7), out, "learning", extra)


def plot_network(stats, runs, out):
    panels = [
        ("rate_L", "mean filtered rate L", False),
        ("rate_R", "mean filtered rate R", False),
        ("rate_H", "mean filtered rate H", False),
        ("rate_std_L", "std of rates within L", False),
        ("rate_std_R", "std of rates within R", False),
        ("rate_std_H", "std of rates within H", False),
        ("mean_balance", "mean E/I balance", False),
        ("mean_W_critic", "mean critic weight", False),
        ("mean_W_actor_input", "mean actor input weight", False),
        ("std_W_actor_input", "std actor input weight", False),
        ("mean_W_actor_recurrent", "mean actor recurrent weight", True),
        ("std_W_actor_recurrent", "std actor recurrent weight", True),
    ]
    plot_panels(stats, runs, panels, (6, 2), (9, 13), out, "network")


def plot_maps(runs, out):
    cfgs = list(runs)
    fig, axs = plt.subplots(2, len(cfgs), figsize=(2.3 * len(cfgs), 4.5), squeeze=False)
    maps = {}
    for cfg, r in runs.items():
        run = r["run"]
        maps[cfg] = {
            "LR": run["W_input_L_minus_R_map"][-1],
            "H": run["W_input_H_map"][-1],
        }
    rows = [
        (
            "LR",
            "final L$-$R input map\n($\\theta \\rightarrow$, $\\dot\\theta \\uparrow$)",
            "RdBu_r",
            True,
        ),
        ("H", "final hidden input map", "viridis", False),
    ]
    for i, (key, label, cmap, symmetric) in enumerate(rows):
        values = np.array([maps[c][key] for c in cfgs])
        if symmetric:
            vmax = np.nanmax(np.abs(values))
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        for j, cfg in enumerate(cfgs):
            im = axs[i, j].imshow(
                maps[cfg][key].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        axs[i, 0].set_ylabel(label, fontsize=9)
        fig.colorbar(im, ax=axs[i], shrink=0.8, pad=0.02)
    for j, cfg in enumerate(cfgs):
        axs[0, j].set_title(
            runs[cfg]["label"].replace(": ", "\n").replace(", ", "\n"), fontsize=7
        )
    fig.savefig(out / "maps.pdf", bbox_inches="tight")
    fig.savefig(out / "maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_run_dynamics(cfg, r, out):
    run = r["run"]
    ts = run["ts"]
    fig, axs = plt.subplots(5, 2, sharex=True, figsize=(9, 9))
    axs[0, 0].plot(ts, run["mean_reward"], lw=0.8)
    axs[0, 0].plot(
        ts,
        np.convolve(run["mean_reward"].squeeze(), np.ones(100) / 100, mode="same"),
        color="k",
        lw=1.5,
    )
    style(axs[0, 0], "mean reward")
    axs[1, 0].plot(ts, run["mean_RPE"], lw=0.8)
    style(axs[1, 0], "mean RPE")
    axs[2, 0].plot(ts, run["mean_W_actor_input"], lw=0.8)
    style(axs[2, 0], "mean actor input weight")
    axs[3, 0].plot(ts, run["mean_W_actor_recurrent"], lw=0.8)
    style(axs[3, 0], "mean actor recurrent weight")
    axs[4, 0].plot(ts, run["mean_W_critic"], lw=0.8)
    style(axs[4, 0], "mean critic weight")
    axs[0, 1].plot(ts, run["mean_filtered_spikes_L"], lw=0.8, label="L")
    axs[0, 1].plot(ts, -run["mean_filtered_spikes_R"], lw=0.8, label="$-$R")
    axs[0, 1].plot(
        ts,
        run["mean_filtered_spikes_L"] - run["mean_filtered_spikes_R"],
        lw=0.8,
        label="L$-$R",
    )
    axs[0, 1].legend(fontsize=7, frameon=False)
    style(axs[0, 1], "mean filtered rates")
    axs[1, 1].plot(ts, run["var_RPE"], lw=0.8)
    style(axs[1, 1], "variance RPE", log=True)
    axs[2, 1].plot(ts, run["var_W_actor_recurrent"], lw=0.8)
    style(axs[2, 1], "variance actor recurrent weight")
    axs[3, 1].plot(ts, run["agent_output"], lw=0.8)
    style(axs[3, 1], "agent output")
    axs[4, 1].plot(ts, run["mean_balance"], lw=0.8)
    axs[4, 1].fill_between(
        ts,
        run["mean_balance"] - np.sqrt(run["var_balance"]),
        run["mean_balance"] + np.sqrt(run["var_balance"]),
        alpha=0.3,
    )
    axs[4, 1].axhline(0.01, color="red", ls="--", lw=0.8)
    style(axs[4, 1], "mean E/I balance")
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(r["label"], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"dynamics_cfg_{cfg}.pdf")
    fig.savefig(out / f"dynamics_cfg_{cfg}.png", dpi=200)
    plt.close(fig)


def print_summary(stats, runs, final_window):
    keys = [
        "episode_duration",
        "episode_rate",
        "mean_reward",
        "mean_RPE",
        "map_LR_std",
        "map_H_std",
    ]
    print(f"Averages over the last {final_window:g} s")
    print(f"{'cfg':>4} " + " ".join(f"{k:>12}" for k in keys))
    for cfg, s in stats.items():
        m = s["t"] > s["t"][-1] - final_window
        print(
            f"{cfg:>4} "
            + " ".join(f"{np.nanmean(s[k][m]):>12.4g}" for k in keys)
            + f"  {runs[cfg]['label']}"
        )


def plot_lr_group_summary(group_data, final_window, out):
    """Compare final-window metrics across lr (x-axis) and group (color), e.g. per delta_V.
    Lines are the mean over seeds; dots are the individual seeds."""
    panels = [
        ("episode_duration", "mean episode duration (s)", False),
        ("mean_reward", "mean reward", False),
        ("mean_RPE", "mean RPE", False),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for gi, (label, (runs, stats)) in enumerate(group_data.items()):
        color = COLORS[gi % len(COLORS)]
        cfgs = list(runs)
        lrs = np.array([runs[cfg]["params"].get("lr", np.nan) for cfg in cfgs])
        unique_lrs = np.unique(lrs)
        for ax, (key, ylabel, log) in zip(axs.flat, panels):
            vals = []
            for cfg in cfgs:
                s = stats[cfg]
                m = s["t"] > s["t"][-1] - final_window
                vals.append(np.nanmean(s[key][m]))
            vals = np.array(vals)
            ax.scatter(lrs, vals, color=color, s=10, alpha=0.5)
            ax.plot(
                unique_lrs,
                [np.nanmean(vals[lrs == lr]) for lr in unique_lrs],
                "o-",
                color=color,
                lw=1.4,
                ms=5,
                label=label,
            )
            ax.set_xscale("log")
            style(ax, ylabel, log)
            ax.set_xlabel("learning rate")
    axs.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Final {final_window:g} s vs learning rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "lr_group_summary.pdf")
    fig.savefig(out / "lr_group_summary.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/pendulum_main_run_final_stability_20261103_170245/results",
    )
    parser.add_argument(
        "--out_dir", default=None, help="defaults to figures/<results folder name>"
    )
    parser.add_argument(
        "--window", type=float, default=50.0, help="analysis window (s)"
    )
    parser.add_argument(
        "--final_window",
        type=float,
        default=200.0,
        help="window for the maps and summary (s)",
    )
    parser.add_argument(
        "--per_run", action="store_true", help="also write the per-run dynamics figures"
    )
    args = parser.parse_args()
    base_out = Path(
        args.out_dir or Path("figures") / Path(args.results_dir).parent.name
    )
    base_out.mkdir(parents=True, exist_ok=True)

    groups = find_groups(args.results_dir)
    if not groups:
        raise SystemExit(f"No .npz files found in {args.results_dir}")

    group_data = {}
    for prefix, paths in groups.items():
        runs = load_runs(paths)
        stats = {cfg: window_stats(r["run"], args.window) for cfg, r in runs.items()}
        title = group_title(prefix, paths[0])
        out = base_out / prefix
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {title} ({prefix}) ===")
        print_summary(stats, runs, args.final_window)
        plot_learning(stats, runs, out)
        plot_network(stats, runs, out)
        plot_maps(runs, out)
        if args.per_run:
            for cfg, r in runs.items():
                plot_run_dynamics(cfg, r, out)
        group_data[title] = (runs, stats)

    if len(group_data) > 1:
        plot_lr_group_summary(group_data, args.final_window, base_out)

    print(f"\nFigures written to {base_out}")


if __name__ == "__main__":
    main()
