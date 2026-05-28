"""Analysis for run_paired_comparison_v2 outputs.

Implements Alexander Eqs. 33 and 34 properly:
  rho(cell, dV, seed) = ∫ e(t) δ_r^task(t) dt / ∫ |e(t) δ_r^task(t)| dt
  SNR(cell, dV, seed) = ∫ e(t) δ_r^task(t) dt / ∫ |e(t) δ_r^noise(t)| dt

where:
  e(t)       = eligibility (saved for B cells; reconstructed offline for A cells
               as instantaneous OUA excursion ζ * s * γ).
  δ_r^task   = exp-filter(spike_noisy - spike_noiseless, τ=100ms).
  δ_r^noise  = exp-filter(Poisson(1 Hz, unit jumps), τ=100ms), generated from
               a seeded reproducible draw.

Also computes:
  - firing rates (noisy / noiseless), V mean & variance, γ-engagement
  - paired statistics across (PN, PS) twins at matched seed
  - paired CSVs and a headline figure

Outputs go to results/overnight-single-synapse-v2/.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "overnight-single-synapse-v2"
RAW = RESULTS / "raw"

# Task constants — must match run_paired_comparison_v2.py
TAU_RPE = 0.1
W0 = 1e-9
E_E = 0.0
V_TH = -50e-3
V_REST = -70e-3
WARMUP_T = 10.0
RPE_NOISE_RATE = 1.0   # Hz
RPE_NOISE_STD = 1.0    # jump std

CELLS = ["A-PN", "A-PS", "A0-PN", "A0-PS", "B-PN", "B-PS"]
PAIRS = {
    "A": ("A-PN", "A-PS"),
    "A0": ("A0-PN", "A0-PS"),
    "B": ("B-PN", "B-PS"),
}


def _voltage_gate(V, delta_V):
    default_area = (V_TH - V_REST)
    driving_force = E_E - V
    integral = lambda V_: (E_E + delta_V - V_) * -np.exp((V_ - V_TH) / delta_V)
    area = integral(V_REST) - integral(V_TH)
    return (driving_force / delta_V * np.exp((V - V_TH) / delta_V)) / (area / default_area)


def _exp_filter(x, dt, tau):
    """Discrete-time exponential filter approximation of dy/dt = (-y + x)/tau."""
    alpha = np.exp(-dt / tau)
    y = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        y[i] = y[i - 1] * alpha + x[i] * (1 - alpha)
    return y


def _rpe_task(ts, S_noisy, S_noiseless, tau=TAU_RPE):
    """Eq. 5.2 — δ_r^task = sign(S0 - S1), filtered."""
    raw = (S_noisy > 0).astype(float) - (S_noiseless > 0).astype(float)
    dt = float(ts[1] - ts[0])
    return _exp_filter(raw, dt, tau)


def _rpe_noise(ts, seed, rate=RPE_NOISE_RATE, std=RPE_NOISE_STD, tau=TAU_RPE):
    """Independent Poisson-jump 'reward noise' process per Eq. 5.3.

    Implemented offline: at each timestep, draw a Poisson jump count with
    probability rate*dt; jumps are Gaussian with std=1.0; filter with τ_RPE.
    Reproducible from the seed.
    """
    rng = np.random.default_rng(seed * 10007 + 1)  # decorrelate from spike seed
    dt = float(ts[1] - ts[0])
    n_jumps_per_step = rng.poisson(rate * dt, size=len(ts))
    jump_signs = rng.normal(0.0, std, size=len(ts))
    raw = n_jumps_per_step.astype(float) * jump_signs
    return _exp_filter(raw, dt, tau)


def _eligibility_oua(z_learn, G_learn, V_noisy, delta_V, use_gate):
    """OUA's instantaneous eligibility-equivalent at the learnable synapse.

    e_ij(t) = ζ_ij(t) * s_ij(t) * γ(V_i(t))   (τ_e → 0 limit, multiplicative-on-weight)
    With s = G / w0; γ = 1 when use_gate is False.
    """
    s_learn = G_learn / W0
    if use_gate:
        gate = _voltage_gate(V_noisy, delta_V)
    else:
        gate = np.ones_like(V_noisy)
    return z_learn * s_learn * gate


def _compute_metrics(path):
    z = np.load(path, allow_pickle=True)
    ts = z["ts"]
    S = z["S"]
    V = z["V"]
    G_learn = np.asarray(z["G_learnable"]).ravel()
    noise = z["noise_state"]
    cell_id = str(z["cell_id"])
    delta_V = float(z["delta_V"])
    seed = int(z["seed"])

    # Reconstruct eligibility e(t).
    if cell_id.startswith("B-"):
        e = np.asarray(z["eligibility_learnable"]).ravel()
    else:
        use_gate = cell_id in ("A-PN", "A-PS")
        if noise.ndim == 2:
            # per-neuron noise broadcast: zeta_i for noisy neuron is noise[:, 0]
            # in per-neuron geometry the learnable synapse receives the per-neuron
            # excursion directly (alpha=0 endpoint, no |E_i| scaling here because
            # we treat the per-neuron noise as the perturbation seen at this
            # synapse — variance match handled separately in calibration).
            zeta_at_learn = noise[:, 0]
            sigma_norm = float(z["sigma_pn"])
        else:
            zeta_at_learn = noise[:, 0, 4]
            sigma_norm = float(z["sigma_ps"])
        # Normalise zeta by sigma so the eligibility is dimensionless O(1)
        # (matches the existing repo convention noise/noise_std in eligibility_LIF).
        zeta_rel = zeta_at_learn / max(sigma_norm, 1e-30)
        # Convert G to s (dimensionless) since W is dimensionless in repo
        e = _eligibility_oua(zeta_rel, G_learn, V[:, 0], delta_V, use_gate)

    # δ_r task and noise
    rpe_task = _rpe_task(ts, S[:, 0], S[:, 1])
    rpe_noise = _rpe_noise(ts, seed)

    # Apply steady-state mask
    ss = ts > WARMUP_T
    e_ss = e[ss]
    rt = rpe_task[ss]
    rn = rpe_noise[ss]

    # Eq. 33 and 34
    num = float(np.trapezoid(e_ss * rt, ts[ss]))
    denom_rho = float(np.trapezoid(np.abs(e_ss * rt), ts[ss]))
    denom_snr = float(np.trapezoid(np.abs(e_ss * rn), ts[ss]))
    rho = num / denom_rho if denom_rho > 1e-30 else 0.0
    snr = num / denom_snr if denom_snr > 1e-30 else 0.0

    # Additional sanity metrics
    fr_n = float(S[ss, 0].sum() / (ts[ss][-1] - ts[ss][0]))
    fr_c = float(S[ss, 1].sum() / (ts[ss][-1] - ts[ss][0]))
    v_mean = float(V[ss, 0].mean())
    v_var = float(V[ss, 0].var())
    rpe_task_var = float(np.var(rt))
    rpe_noise_var = float(np.var(rn))
    elig_var = float(np.var(e_ss))

    return {
        "cell_id": cell_id,
        "delta_V": delta_V,
        "dV_pow": int(round(np.log2(delta_V))),
        "seed": seed,
        "rho": rho,
        "snr": snr,
        "fr_noisy": fr_n,
        "fr_clean": fr_c,
        "v_mean_noisy": v_mean,
        "v_var_noisy": v_var,
        "rpe_task_var": rpe_task_var,
        "rpe_noise_var": rpe_noise_var,
        "elig_var": elig_var,
        "num_e_rtask": num,
    }


def main():
    files = sorted(RAW.glob("*.npz"))
    print(f"[analyze] {len(files)} raw files")
    if not files:
        return

    rows = []
    for fp in files:
        try:
            rows.append(_compute_metrics(fp))
        except Exception as e:
            print(f"  [skip] {fp.name}: {type(e).__name__}: {str(e)[:100]}")

    with open(RESULTS / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[analyze] summary.csv ({len(rows)} rows)")

    # Paired CSV
    idx = defaultdict(list)
    for r in rows:
        idx[(r["cell_id"], r["delta_V"])].append(r)
    dV_vals = sorted({r["delta_V"] for r in rows})
    paired_rows = []
    for pair_id, (pn, ps) in PAIRS.items():
        for dV in dV_vals:
            pns = idx.get((pn, dV), [])
            pss = idx.get((ps, dV), [])
            common = sorted({r["seed"] for r in pns} & {r["seed"] for r in pss})
            for s in common:
                pr = next(r for r in pns if r["seed"] == s)
                pp = next(r for r in pss if r["seed"] == s)
                paired_rows.append({
                    "pair": pair_id,
                    "delta_V": dV,
                    "dV_pow": int(round(np.log2(dV))),
                    "seed": s,
                    "rho_PN": pr["rho"],
                    "rho_PS": pp["rho"],
                    "diff_rho": pp["rho"] - pr["rho"],
                    "snr_PN": pr["snr"],
                    "snr_PS": pp["snr"],
                    "log10_ratio_SNR": (
                        np.log10(abs(pp["snr"]) + 1e-30)
                        - np.log10(abs(pr["snr"]) + 1e-30)
                    ),
                    "fr_PN": pr["fr_noisy"],
                    "fr_PS": pp["fr_noisy"],
                })
    if paired_rows:
        with open(RESULTS / "paired_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(paired_rows[0].keys()))
            w.writeheader()
            w.writerows(paired_rows)
        print(f"[analyze] paired_summary.csv ({len(paired_rows)} rows)")

    # Headline plot: 6 cells × dV with rho and SNR overlays
    def _agg(cell, dV, key):
        vs = [r[key] for r in idx.get((cell, dV), [])]
        if not vs:
            return np.nan, np.nan
        return float(np.mean(vs)), float(np.std(vs) / np.sqrt(max(len(vs), 1)))

    dV_arr = np.array(dV_vals)
    dV_log2 = np.log2(dV_arr)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    cell_order = ["A-PN", "B-PN", "A0-PN", "A-PS", "B-PS", "A0-PS"]
    titles = {
        "A-PN": "A-PN  (per-neuron OUA, gate ON, HEADLINE)",
        "A-PS": "A-PS  (per-synapse OUA, gate ON, HEADLINE)",
        "A0-PN": "A0-PN (per-neuron OUA, gate OFF, control)",
        "A0-PS": "A0-PS (per-synapse OUA, gate OFF, control)",
        "B-PN": "B-PN  (per-neuron Elig, gate ON — Alexander)",
        "B-PS": "B-PS  (per-synapse Elig, gate ON, co-headline)",
    }
    for ax, cell in zip(axes.ravel(), cell_order):
        means_rho = [_agg(cell, dV, "rho")[0] for dV in dV_vals]
        sems_rho = [_agg(cell, dV, "rho")[1] for dV in dV_vals]
        means_snr = [_agg(cell, dV, "snr")[0] for dV in dV_vals]
        sems_snr = [_agg(cell, dV, "snr")[1] for dV in dV_vals]
        ax.errorbar(dV_log2, means_rho, yerr=sems_rho, fmt="o-", color="C0", label=r"$\rho$ (Eq.33)")
        ax2 = ax.twinx()
        ax2.errorbar(dV_log2, means_snr, yerr=sems_snr, fmt="s--", color="C3", label="SNR (Eq.34)")
        ax.set_title(titles.get(cell, cell), fontsize=9)
        ax.set_xlabel("log2(ΔV)")
        ax.set_ylabel(r"$\rho$", color="C0")
        ax2.set_ylabel("SNR", color="C3")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.2, 1.2)
    fig.suptitle(
        "Single Synapse Task — gradient alignment ρ (Eq. 33) and SNR (Eq. 34)\n"
        f"6 cells, 7 ΔV × 5 seeds (η=0, T=60s, noise_scale=2.0, balance=0.0)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / "figure_v2_6cell_rho_snr.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] {out}")

    # Paired comparison panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for col, (pair_id, (pn, ps)) in enumerate(PAIRS.items()):
        # row 0: paired difference of rho
        diffs_mean = []
        diffs_sem = []
        for dV in dV_vals:
            ds = [r["diff_rho"] for r in paired_rows if r["pair"] == pair_id and r["delta_V"] == dV]
            if ds:
                diffs_mean.append(float(np.mean(ds)))
                diffs_sem.append(float(np.std(ds) / np.sqrt(len(ds))))
            else:
                diffs_mean.append(np.nan)
                diffs_sem.append(np.nan)
        ax = axes[0, col]
        ax.errorbar(dV_log2, diffs_mean, yerr=diffs_sem, fmt="o-", color="C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"Pair {pair_id}: ρ(PS) − ρ(PN)")
        ax.set_ylabel("Δρ")
        ax.grid(alpha=0.3)

        # row 1: log10(SNR_PS / SNR_PN)
        rs_mean = []
        rs_sem = []
        for dV in dV_vals:
            rs = [r["log10_ratio_SNR"] for r in paired_rows if r["pair"] == pair_id and r["delta_V"] == dV]
            if rs:
                rs_mean.append(float(np.mean(rs)))
                rs_sem.append(float(np.std(rs) / np.sqrt(len(rs))))
            else:
                rs_mean.append(np.nan)
                rs_sem.append(np.nan)
        ax = axes[1, col]
        ax.errorbar(dV_log2, rs_mean, yerr=rs_sem, fmt="o-", color="C4")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"Pair {pair_id}: log10(SNR_PS / SNR_PN)")
        ax.set_xlabel("log2(ΔV)")
        ax.set_ylabel("log10 ratio")
        ax.grid(alpha=0.3)
    fig.suptitle("Paired per-neuron vs per-synapse comparison (matched seed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out2 = RESULTS / "figure_v2_paired.pdf"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] {out2}")


if __name__ == "__main__":
    main()
