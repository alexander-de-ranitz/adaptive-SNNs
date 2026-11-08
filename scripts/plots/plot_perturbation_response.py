from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import ScaledTranslation
from scipy.optimize import curve_fit

from adaptive_snn.models.networks import LIFNetwork

DATA_DIRS = [Path("results/assumption_verification_20261104_112544/results")]
OUTPUT_PATH = Path("figures/perturbation_response")

V_BIN_WIDTH = 0.5e-3
MIN_TRIALS_PER_V_BIN = 100


class FixedDecimalsFormatter(ScalarFormatter):
    """ScalarFormatter with a fixed number of decimals, so tick labels of different panels align."""

    def __init__(self, decimals: int, power: int | None = None):
        super().__init__()
        self.decimals = decimals
        if power is not None:
            self.set_powerlimits((power, power))

    def _set_format(self):
        self.format = f"%1.{self.decimals}f"
        if self._usetex or self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def load_results(data_dirs: list[Path]) -> dict:
    """Load the per-seed results written by scripts/snellius/assumption_verification/run.py.

    Per-epoch sums are stacked along a leading seed axis; per-trial arrays are concatenated along the trial axis.
    """
    runs = [dict(np.load(f)) for d in data_dirs for f in sorted(d.glob("*.npz"))]
    n_trials = np.array([int(run["n_trials"]) for run in runs])
    return {
        "As": runs[0]["As"],
        "dt": float(runs[0]["dt"]),
        "pulse_offset": int(runs[0]["pulse_offset"]),
        "n_trials": n_trials,
        "y_diff_sum": np.stack(
            [run["y_diff_sum"] for run in runs]
        ),  # (seed, amplitude, step in epoch)
        "V_diff_sum": np.stack([run["V_diff_sum"] for run in runs]),
        "V_diff_sum_spike_free": np.stack(
            [run["V_diff_sum_spike_free"] for run in runs]
        ),
        "n_spike_free": np.stack(
            [run["n_spike_free"] for run in runs]
        ),  # (seed, amplitude)
        "count_change_by_time": np.stack([run["count_change_by_time"] for run in runs]),
        "applied": sum(run["applied_sum"] for run in runs)
        / n_trials.sum(),  # Mean applied amplitude
        "total_y_diff": np.concatenate([run["total_y_diff"] for run in runs], axis=1),
        "V_at_pulse": np.concatenate([run["V_at_pulse"] for run in runs]),
    }


def amplitude_pairs(As: np.ndarray) -> list[tuple[int, int]]:
    """Indices (i_pos, i_neg) of each positive amplitude and its negative counterpart."""
    return [
        (i, np.flatnonzero(np.isclose(As, -As[i], rtol=1e-6, atol=0.0))[0])
        for i in np.flatnonzero(As > 0)
    ]


def estimate(
    results: dict,
    sums: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray | None = None,
):
    """Response per nS, pooled over amplitudes as sum(w * mean response) / sum(w * applied amplitude).

    sums has shape (seed, amplitude, ...) and counts, the number of pulses summed, shape (seed, amplitude);
    by default all pulses. weights has one entry per amplitude: 1 to include it, 0 to leave it out, and -1
    for the negative amplitudes in a central difference. Pooling is valid because the response is linear.
    The SEM is computed from the spread of the per-seed estimates.
    """
    if counts is None:
        counts = np.repeat(results["n_trials"][:, None], len(results["As"]), axis=1)
    counts = counts.reshape(counts.shape + (1,) * (sums.ndim - counts.ndim))
    norm = weights @ results["applied"] * 1e9
    seed_estimates = np.tensordot(weights, sums / counts, axes=(0, 1)) / norm
    mean = (
        np.tensordot(weights, sums.sum(axis=0) / counts.sum(axis=0), axes=(0, 0)) / norm
    )
    return mean, seed_estimates.std(axis=0, ddof=1) / np.sqrt(len(sums))


def per_trial_sensitivity(results: dict) -> np.ndarray:
    """Per-trial central difference of the total spike difference, in spikes/nS, shape (n_pairs, n_trials)."""
    tot, applied = results["total_y_diff"].astype(float), results["applied"] * 1e9
    return np.stack(
        [
            (tot[p] - tot[n]) / (applied[p] - applied[n])
            for p, n in amplitude_pairs(results["As"])
        ]
    )


def gating_shape(V, scale, delta_V):
    """Shape of the gating term m(V) = (E_E - V) exp((V - V_th) / delta_V), up to a scale factor."""
    return (
        scale
        * (LIFNetwork.reversal_potential_E - V)
        * np.exp((V - LIFNetwork.firing_threshold) / delta_V)
    )


def log_exponential_fit(t, y, name: str, unit: str, t_min: float = 0.0):
    """Fit y = a * exp(-t / tau) for t >= t_min (ms) and print the coefficients."""
    after = t >= t_min
    y0 = y[after][np.argmax(np.abs(y[after]))]
    (a, tau), _ = curve_fit(
        lambda t, a, tau: a * np.exp(-t / tau), t[after], y[after], p0=(y0, 2.0)
    )
    print(f"Exponential fit, {name}: a = {a:.4g} {unit}, tau = {tau:.3f} ms")


def plot_voltage_kernel(ax, results: dict):
    """Functional derivative dV(t')/dg(t) against time since the pulse, in mV/(nS ms): the mean over amplitudes,
    with the min and max over amplitudes.

    The voltage difference per nS of pulse is divided by the pulse duration (one step, dt), which turns the
    response to a pulse of finite duration into a derivative with respect to conductance at a moment in time.

    A narrow range between them shows that the response per nS is the same for all amplitudes and both signs. The mean
    is also shown over only the epochs in which neither neuron spiked, which removes the effect of spike resets.
    """
    t = (
        (np.arange(results["V_diff_sum"].shape[-1]) - results["pulse_offset"])
        * results["dt"]
        * 1e3
    )
    scale = (
        1 / results["dt"]
    )  # V per nS of a pulse of duration dt -> V/(nS s) = mV/(nS ms)
    curves = []
    for a, A in enumerate(results["As"]):
        if A == 0:
            continue
        weights = np.zeros(len(results["As"]))
        weights[a] = 1.0
        curves.append(estimate(results, results["V_diff_sum"], weights)[0] * scale)
    ax.plot(
        t, np.min(curves, axis=0), color="grey", lw=0.5, label="Min/max over amplitudes"
    )
    ax.plot(t, np.max(curves, axis=0), color="grey", lw=0.5)
    sign = np.sign(results["As"])  # A = 0 (sanity check) gets weight 0
    m, _ = estimate(results, results["V_diff_sum"], sign)
    ax.plot(t, m * scale, color="k", label="Mean")
    log_exponential_fit(t, m * scale, "dV/dg (mean)", "mV/(nS ms)")
    m, _ = estimate(
        results, results["V_diff_sum_spike_free"], sign, results["n_spike_free"]
    )
    ax.plot(t, m * scale, color="grey", linestyle=":", label="Mean, spike-free")
    log_exponential_fit(t, m * scale, "dV/dg (mean, spike-free)", "mV/(nS ms)")
    ax.axhline(0.0, color="k", lw=0.4)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time since perturbation (ms)")
    ax.set_ylabel(r"$\delta V / \delta g$")
    ax.yaxis.set_major_formatter(FixedDecimalsFormatter(3))
    ax.legend()


def plot_count_changes(ax, results: dict):
    """Rate of added (+) and removed (-) spikes per amplitude, by the time at which the pulse first changed the spiking.

    Only pulses that change the spike count over the window contribute; pulses that only shift spikes do not.
    The last step is left out: spikes shifted across the end of the epoch look like added or removed spikes.
    An inset zooms in on the response after the pulse step, which is small compared with the pulse step itself.
    """
    n, dt, P = results["n_trials"], results["dt"], results["pulse_offset"]
    changes = results["count_change_by_time"][..., :-1]
    t = (np.arange(changes.shape[-1]) - P) * dt * 1e3
    inset = ax.inset_axes([0.38, 0.3, 0.59, 0.38])
    colors, after_pulse = {}, []
    for a, A in enumerate(results["As"]):
        if A == 0:
            continue
        mean = changes[:, a].sum(axis=0) / n.sum() / dt
        sem = (changes[:, a] / n[:, None] / dt).std(axis=0, ddof=1) / np.sqrt(len(n))
        color = colors.setdefault(abs(A), f"C{len(colors)}")
        for target in (ax, inset):
            target.plot(
                t,
                mean,
                color=color,
                linestyle="-" if A > 0 else "--",
                label=f"{A * 1e9:+.0f} nS",
            )
            target.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
        after_pulse.append(mean[P + 1 :])
        # The pulse step itself is left out of the fit: it is a separate, near-instantaneous effect
        log_exponential_fit(
            t, mean, f"added/removed spikes ({A * 1e9:+.0f} nS)", "Hz", t_min=dt * 1e3
        )
    for target in (ax, inset):
        target.axhline(0.0, color="k", lw=0.5, linestyle=":")
    ax.set_xlim(t[0], t[-1])
    inset.set_xlim(-0.2, 8)
    inset.set_ylim(1.2 * np.min(after_pulse), 1.2 * np.max(after_pulse))
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set(linewidth=0.4, alpha=0.6)
    indicator = ax.indicate_inset_zoom(inset, edgecolor="grey", lw=0.5)
    # Connectors in the order lower left, upper left, lower right, upper right
    for connector, visible in zip(indicator.connectors, (False, True, True, False)):
        connector.set_visible(visible)
    ax.set_xlabel("Time since perturbation (ms)")
    ax.set_ylabel(r"$\Delta$ firing rate (Hz)")
    ax.legend(ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2)


def plot_linearity(ax, results: dict):
    """Net effect of the pulse on the spike count over the window per nS against amplitude.

    Shown separately for positive and negative pulses, and as their central difference, in which the
    even-order (asymmetric) terms cancel.
    """
    P = results["pulse_offset"]
    total = results["y_diff_sum"][..., P:].sum(axis=-1)
    pairs = amplitude_pairs(results["As"])
    abs_As_nS = [results["As"][p] * 1e9 for p, _ in pairs]
    for label, signs, style in [
        ("Positive", (1.0, 0.0), dict(color="tab:blue", marker="o")),
        ("Negative", (0.0, 1.0), dict(color="tab:orange", marker="o")),
        ("Mean", (1.0, -1.0), dict(color="k", marker="s", linestyle="--")),
    ]:
        estimates = []
        for p, n in pairs:
            weights = np.zeros(len(results["As"]))
            weights[p], weights[n] = signs
            estimates.append(estimate(results, total, weights))
        mean, sem = np.array(estimates).T
        ax.errorbar(abs_As_nS, mean, sem, capsize=2, ms=3, label=label, **style)
    ax.set_xlabel("Perturbation amplitude (nS)")
    ax.set_xticks(abs_As_nS)
    ax.set_ylabel(r"$\Delta$ spikes / nS")
    ax.set_ylim(0, 1.15 * ax.get_ylim()[1])
    ax.yaxis.set_major_formatter(FixedDecimalsFormatter(3, power=-3))
    ax.legend()


def plot_voltage_dependence(ax, results: dict):
    """Per-trial sensitivity binned by the membrane voltage at the pulse, with a fit of the gating term."""
    sensitivity = per_trial_sensitivity(results)
    V = results["V_at_pulse"]
    pooled = sensitivity.mean(axis=0)  # Average over amplitude pairs per trial

    edges = np.arange(
        np.floor(V.min() / V_BIN_WIDTH) * V_BIN_WIDTH,
        LIFNetwork.firing_threshold + V_BIN_WIDTH / 2,
        V_BIN_WIDTH,
    )
    idx = np.digitize(V, edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.bincount(idx[idx >= 0], minlength=len(centers))[: len(centers)]
    keep = counts >= MIN_TRIALS_PER_V_BIN

    def binned(values):
        mean = np.array(
            [
                values[idx == b].mean() if keep[b] else np.nan
                for b in range(len(centers))
            ]
        )
        sem = np.array(
            [
                values[idx == b].std(ddof=1) / np.sqrt(counts[b]) if keep[b] else np.nan
                for b in range(len(centers))
            ]
        )
        return mean, sem

    # The mean over the amplitude pairs equals the mean over all amplitudes, each normalised by its amplitude
    mean, sem = binned(pooled)
    ax.plot(centers * 1e3, mean, color="k", label="Mean")
    ax.fill_between(centers * 1e3, mean - sem, mean + sem, color="k", alpha=0.2, lw=0)

    # Fit on the individual trials rather than the bins: bins in which no trial changed have zero
    # SEM and could not be weighted, but they carry the information that the sensitivity is small there.
    # The unweighted fit is dominated by trials just below threshold, where the sensitivity is largest.
    (scale, delta_V), _ = curve_fit(
        gating_shape,
        V,
        pooled,
        p0=(np.nanmax(mean), 1e-3),
        bounds=([0, 1e-5], [np.inf, 0.1]),
    )
    V_fine = np.linspace(centers[keep].min(), LIFNetwork.firing_threshold, 200)
    ax.plot(
        V_fine * 1e3,
        gating_shape(V_fine, scale, delta_V),
        color="tab:red",
        label=rf"Fit ($\Delta V$ = {delta_V * 1e3:.2f} mV)",
        linestyle="--",
    )
    print(f"Fitted delta_V = {delta_V * 1e3:.3f} mV")

    ax_hist = ax.twinx()
    ax_hist.hist(V * 1e3, bins=edges * 1e3, color="grey", alpha=0.2)
    ax_hist.set_yticks([])
    ax.set_zorder(ax_hist.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.axhline(0.0, color="k", lw=0.4)
    ax.set_xlim(-63, -49.5)
    ax.set_xticks(np.arange(-62, -49, 4))
    ax.set_xlabel(r"$V$ at perturbation onset (mV)")
    ax.set_ylabel(r"$\Delta$ spikes / nS")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend()


def plot_response(results: dict, save_path: Path = OUTPUT_PATH):
    zero = results["As"] == 0
    print(
        "A = 0 sanity check (expected 0): "
        f"{np.count_nonzero(results['y_diff_sum'][:, zero])} spike and "
        f"{np.count_nonzero(results['V_diff_sum'][:, zero])} voltage differences"
    )
    style = {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
    }
    with plt.rc_context(style):
        fig, ((ax_V, ax_count), (ax_lin, ax_m)) = plt.subplots(2, 2, figsize=(6, 4))
        plot_voltage_kernel(ax_V, results)
        plot_count_changes(ax_count, results)
        plot_linearity(ax_lin, results)
        plot_voltage_dependence(ax_m, results)
        for ax, letter in zip((ax_V, ax_count, ax_lin, ax_m), "ABCD"):
            ax.text(
                0,
                1,
                rf"\textbf{{{letter}}}",
                fontsize=10,
                va="top",
                ha="right",
                transform=ax.transAxes
                + ScaledTranslation(-0.12, 0.12, fig.dpi_scale_trans),
            )
        fig.tight_layout(pad=0.2, w_pad=2.0)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
        fig.savefig(
            save_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01
        )
        plt.close(fig)


if __name__ == "__main__":
    plot_response(load_results(DATA_DIRS))
