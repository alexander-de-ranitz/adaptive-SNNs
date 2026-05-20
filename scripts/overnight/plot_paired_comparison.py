"""Paired 4-cell comparison figure for the Single Synapse Task.

Loads results/overnight-single-synapse/raw/*.npz and produces:
- summary.csv (per (cell, dV, seed))
- paired_summary.csv (per (pair, dV, seed))
- figure_paired_comparison.pdf (2x2 grid + paired-difference + paired-ratio rows)
- figure_alexander_replication.pdf (B-PN replication, side-by-side rho/SNR vs dV)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "overnight-single-synapse"
RAW = RESULTS / "raw"

CELLS = ["A-PN", "A-PS", "B-PN", "B-PS"]
PAIRS = {"A": ("A-PN", "A-PS"), "B": ("B-PN", "B-PS")}
DV_POWS = list(range(-13, -6))  # -13..-7 inclusive (7 values)
SEEDS = [42, 43, 44, 45, 46]
DT = 1e-4
WARMUP_T = 5.0
TAU_RPE = 0.1


def _approx_rpe_from_spikes(ts, S, tau=TAU_RPE):
    """Exponentially decaying RPE proxy from instantaneous spike difference."""
    raw = S[:, 0] - S[:, 1]
    alpha = np.exp(-(ts[1] - ts[0]) / tau)
    rpe = np.zeros_like(raw, dtype=float)
    for i in range(1, len(raw)):
        rpe[i] = rpe[i - 1] * alpha + raw[i] * (1 - alpha)
    return rpe


def _compute_metrics(raw_path):
    """Compute rho and SNR for one raw npz file."""
    z = np.load(raw_path, allow_pickle=True)
    ts = z["ts"]
    S = z["S"]
    V = z["V"]
    G_learn = np.asarray(z["G_learn"]).ravel()
    noise = z["noise"]
    cell_id = str(z["cell_id"])

    # Steady-state mask
    ss = ts > WARMUP_T

    # RPE proxy from saved spikes
    rpe = _approx_rpe_from_spikes(ts, S)
    rpe_ss = rpe[ss]

    # Build the per-step gradient signal for the LEARNABLE synapse
    # (neuron 0, input 2 -> column 4 in the W matrix).
    # OUA: g_step = lr * rpe * (zeta / sigma_E) * (G_learn / w0) * gate
    # Eligibility: g_step = rpe * eligibility
    # We set lr = 1 (eta=0 means we measure the gradient signal directly).
    if cell_id.startswith("A-"):
        # noise here is the per-neuron OU (B-PN style) for A-PN or 2D for A-PS
        # We saved noise as the noise_state of the noisy wrapper.
        if noise.ndim == 2:
            # per-neuron noise: shape (T, N_neurons)
            zeta_at_learnable = noise[:, 0]  # noisy neuron's per-neuron noise
            sigma = float(z["sigma_pn"]) if "sigma_pn" in z.files else 1.0
        else:
            # per-synapse noise: shape (T, N_neurons, N+N_in)
            zeta_at_learnable = noise[:, 0, 4]
            sigma = float(z["sigma_ps"]) if "sigma_ps" in z.files else 1.0
        # G_learn already (T,) — convert to s via 1/w0 = 1e9 (synaptic_increment = 1nS)
        s_learn = G_learn / 1e-9
        # Use gate proxy = 1 here (we did not save gate values); rho/SNR computed
        # on the un-gated gradient kernel is still informative for comparisons.
        signal = rpe * (zeta_at_learnable / max(sigma, 1e-30)) * s_learn
    else:
        # Eligibility cells: saved eligibility_learnable (T,)
        elig = z["eligibility_learnable"].ravel()
        signal = rpe * elig

    signal_ss = signal[ss]
    # Gradient alignment proxy: sign of the time-averaged signal vs the
    # expected positive gradient (for SST, the noiseless twin firing more
    # than the noisy neuron when the synapse is strong enough). The true
    # sign of d<R>/dW for the learnable synapse is positive by construction:
    # increasing W -> more excitation -> noisy neuron fires more -> closer to
    # noiseless twin baseline. We define rho = sign(mean(signal)) for the
    # cell at non-zero seed magnitudes.
    mean_sig = float(np.mean(signal_ss))
    std_sig = float(np.std(signal_ss))
    rho_proxy = mean_sig / (np.abs(mean_sig) + 1e-30)  # +/- 1 sign
    snr_proxy = np.abs(mean_sig) / (std_sig + 1e-30)

    fr_noisy = float(S[ss, 0].sum() / ((ts[ss][-1] - ts[ss][0]) if ss.any() else 1))
    fr_clean = float(S[ss, 1].sum() / ((ts[ss][-1] - ts[ss][0]) if ss.any() else 1))
    v_mean_noisy = float(V[ss, 0].mean())
    v_var_noisy = float(V[ss, 0].var())

    return {
        "cell_id": cell_id,
        "delta_V": float(z["delta_V"]),
        "seed": int(z["seed"]),
        "rho": rho_proxy,
        "snr": snr_proxy,
        "mean_sig": mean_sig,
        "std_sig": std_sig,
        "fr_noisy": fr_noisy,
        "fr_clean": fr_clean,
        "v_mean_noisy": v_mean_noisy,
        "v_var_noisy": v_var_noisy,
    }


def main():
    if not RAW.exists():
        print(f"[ERROR] no raw directory at {RAW}")
        return

    rows = []
    files = sorted(RAW.glob("*.npz"))
    print(f"[plot] processing {len(files)} files")
    for fp in files:
        try:
            rows.append(_compute_metrics(fp))
        except Exception as exc:
            print(f"  [skip] {fp.name}: {exc}")

    if not rows:
        print("[plot] no rows; abort")
        return

    # Write summary CSV
    with open(RESULTS / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[plot] wrote summary.csv with {len(rows)} rows")

    # Index: dict[(cell, dV)] -> list[row]
    idx = {}
    for r in rows:
        idx.setdefault((r["cell_id"], r["delta_V"]), []).append(r)
    dV_vals = sorted({r["delta_V"] for r in rows})

    # Paired summary
    paired_rows = []
    for pair_id, (pn, ps) in PAIRS.items():
        for dV in dV_vals:
            pn_rows = idx.get((pn, dV), [])
            ps_rows = idx.get((ps, dV), [])
            seeds_common = sorted(
                {r["seed"] for r in pn_rows} & {r["seed"] for r in ps_rows}
            )
            for s in seeds_common:
                pn_r = next(r for r in pn_rows if r["seed"] == s)
                ps_r = next(r for r in ps_rows if r["seed"] == s)
                paired_rows.append({
                    "pair": pair_id,
                    "delta_V": dV,
                    "seed": s,
                    "rho_PN": pn_r["rho"],
                    "rho_PS": ps_r["rho"],
                    "diff_rho": ps_r["rho"] - pn_r["rho"],
                    "snr_PN": pn_r["snr"],
                    "snr_PS": ps_r["snr"],
                    "ratio_snr_PS_PN": ps_r["snr"] / (pn_r["snr"] + 1e-30),
                })
    if paired_rows:
        with open(RESULTS / "paired_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)
        print(f"[plot] wrote paired_summary.csv with {len(paired_rows)} rows")

    # ---- Headline figure: 2x2 + paired-difference + paired-ratio ----
    dV_vals_arr = np.array(dV_vals)
    dV_log2 = np.log2(dV_vals_arr)

    fig, axes = plt.subplots(4, 2, figsize=(9, 12), sharex="col")

    panel_titles = {
        ("A-PN", "rho"): "A-PN (per-neuron OUA): rho",
        ("A-PS", "rho"): "A-PS (per-synapse OUA, headline): rho",
        ("B-PN", "rho"): "B-PN (per-neuron Elig, Alexander): rho",
        ("B-PS", "rho"): "B-PS (per-synapse Elig, co-headline): rho",
    }

    def _agg(cell, dV, key):
        vals = [r[key] for r in idx.get((cell, dV), [])]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals) / np.sqrt(max(len(vals), 1)))

    # Top 2x2: rho + SNR bars per cell
    for col, cell_col in enumerate([("A-PN", "B-PN"), ("A-PS", "B-PS")]):
        for row, cell in enumerate(cell_col):
            ax = axes[row, col]
            means_rho = [_agg(cell, dV, "rho")[0] for dV in dV_vals]
            sems_rho = [_agg(cell, dV, "rho")[1] for dV in dV_vals]
            means_snr = [_agg(cell, dV, "snr")[0] for dV in dV_vals]
            sems_snr = [_agg(cell, dV, "snr")[1] for dV in dV_vals]
            ax2 = ax.twinx()
            ax.errorbar(dV_log2, means_rho, yerr=sems_rho, fmt="o-", color="C0", label="rho")
            ax2.errorbar(
                dV_log2,
                np.maximum(np.array(means_snr), 1e-6),
                yerr=sems_snr,
                fmt="s--",
                color="C3",
                label="SNR",
            )
            ax.set_title(cell)
            ax.set_ylabel("rho", color="C0")
            ax2.set_ylabel("SNR", color="C3")
            ax.set_ylim(-1.2, 1.2)
            if np.any(np.array(means_snr) > 0):
                ax2.set_yscale("log")
            ax.axhline(0, color="k", lw=0.5)
            ax.grid(alpha=0.3)

    # Bottom rows: paired statistics (PS - PN difference, PS/PN ratio)
    for col, pair_id in enumerate(["A", "B"]):
        pn_cell, ps_cell = PAIRS[pair_id]
        # Paired difference of rho
        ax = axes[2, col]
        diffs_mean = []
        diffs_sem = []
        for dV in dV_vals:
            paired_for_dv = [
                r for r in paired_rows if r["pair"] == pair_id and r["delta_V"] == dV
            ]
            if paired_for_dv:
                d = np.array([r["diff_rho"] for r in paired_for_dv])
                diffs_mean.append(d.mean())
                diffs_sem.append(d.std() / np.sqrt(len(d)))
            else:
                diffs_mean.append(np.nan)
                diffs_sem.append(np.nan)
        ax.errorbar(dV_log2, diffs_mean, yerr=diffs_sem, fmt="o-", color="C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"Pair {pair_id}: PS - PN rho difference")
        ax.set_ylabel("Delta rho (PS - PN)")
        ax.grid(alpha=0.3)

        # Paired SNR ratio
        ax = axes[3, col]
        ratios_mean = []
        ratios_sem = []
        for dV in dV_vals:
            paired_for_dv = [
                r for r in paired_rows if r["pair"] == pair_id and r["delta_V"] == dV
            ]
            if paired_for_dv:
                r_arr = np.array([np.log10(r["ratio_snr_PS_PN"] + 1e-30) for r in paired_for_dv])
                ratios_mean.append(r_arr.mean())
                ratios_sem.append(r_arr.std() / np.sqrt(len(r_arr)))
            else:
                ratios_mean.append(np.nan)
                ratios_sem.append(np.nan)
        ax.errorbar(dV_log2, ratios_mean, yerr=ratios_sem, fmt="o-", color="C4")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"Pair {pair_id}: log10(SNR_PS / SNR_PN)")
        ax.set_ylabel("log10(SNR_PS / SNR_PN)")
        ax.set_xlabel("log2(Delta_V) [V]")
        ax.grid(alpha=0.3)

    fig.suptitle("Paired 4-cell comparison: Single Synapse Task (eta=0)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS / "figure_paired_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")

    # ---- Supporting: Alexander replication (B-PN cell) ----
    fig2, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    means_rho = [_agg("B-PN", dV, "rho")[0] for dV in dV_vals]
    sems_rho = [_agg("B-PN", dV, "rho")[1] for dV in dV_vals]
    means_snr = [_agg("B-PN", dV, "snr")[0] for dV in dV_vals]
    sems_snr = [_agg("B-PN", dV, "snr")[1] for dV in dV_vals]
    ax[0].errorbar(dV_log2, means_rho, yerr=sems_rho, fmt="o-", color="C0")
    ax[0].set_title("B-PN replication: rho vs Delta_V")
    ax[0].set_xlabel("log2(Delta_V) [V]")
    ax[0].set_ylabel("rho")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].grid(alpha=0.3)

    ax[1].errorbar(
        dV_log2,
        np.maximum(np.array(means_snr), 1e-6),
        yerr=sems_snr,
        fmt="o-",
        color="C3",
    )
    ax[1].set_title("B-PN replication: SNR vs Delta_V")
    ax[1].set_xlabel("log2(Delta_V) [V]")
    ax[1].set_ylabel("SNR")
    if np.any(np.array(means_snr) > 0):
        ax[1].set_yscale("log")
    ax[1].grid(alpha=0.3)
    fig2.tight_layout()
    out2 = RESULTS / "figure_alexander_replication.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"[plot] wrote {out2}")


if __name__ == "__main__":
    main()
