"""Analysis for Phase C (learning dynamics) and Phase E (Tier-2 patterns).

Produces:
- results/overnight-learning-dynamics/figure_W_trajectories.pdf
- results/overnight-learning-dynamics/summary.csv
- results/overnight-tier2-patterns/figure_readout_fr.pdf
- results/overnight-tier2-patterns/summary.csv
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


# --------------------------- Phase C: learning dynamics ----------------------
def analyze_learning_dynamics():
    raw = REPO / "results" / "overnight-learning-dynamics" / "raw"
    out_dir = raw.parent
    if not raw.exists() or not list(raw.glob("*.npz")):
        print("[learning] no data in", raw)
        return
    files = sorted(raw.glob("*.npz"))
    print(f"[learning] {len(files)} files")

    rows = []
    by_key = defaultdict(list)  # (cell, dV) -> list of (ts, W) trajectories
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        cell = str(z["cell_id"])
        dV = float(z["delta_V"])
        seed = int(z["seed"])
        ts = z["ts"]
        W = np.asarray(z["W_learn"]).ravel()
        S = np.asarray(z["S"])
        fr_n = float(S[:, 0].sum() / float(ts[-1]))
        fr_c = float(S[:, 1].sum() / float(ts[-1]))
        W_init = float(W[0]); W_final = float(W[-1])
        dW_dt = (W_final - W_init) / float(ts[-1])
        rows.append({
            "cell": cell, "delta_V": dV, "dV_pow": int(round(np.log2(dV))), "seed": seed,
            "W_init": W_init, "W_final": W_final, "dW_dt": dW_dt,
            "fr_noisy": fr_n, "fr_clean": fr_c,
        })
        by_key[(cell, dV)].append((ts, W))

    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[learning] wrote summary.csv ({len(rows)} rows)")

    # Aggregate plot: W(t) per cell, mean across seeds + shading
    cells = sorted({r["cell"] for r in rows})
    dVs = sorted({r["delta_V"] for r in rows})

    fig, axes = plt.subplots(len(dVs), len(cells), figsize=(2.0 * len(cells), 2.3 * len(dVs)), squeeze=False)
    for i, dV in enumerate(dVs):
        for j, cell in enumerate(cells):
            ax = axes[i, j]
            trajs = by_key.get((cell, dV), [])
            if not trajs:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                continue
            ts = trajs[0][0]
            Ws = np.stack([w for _, w in trajs])
            mean_W = np.mean(Ws, axis=0)
            std_W = np.std(Ws, axis=0)
            ax.plot(ts, mean_W, color="C0")
            ax.fill_between(ts, mean_W - std_W, mean_W + std_W, alpha=0.2, color="C0")
            ax.set_title(f"{cell} dV=2^{int(round(np.log2(dV)))}", fontsize=8)
            ax.axhline(0, color="k", lw=0.3)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel("W_learnable")
            if i == len(dVs) - 1:
                ax.set_xlabel("time (s)")
    fig.suptitle("Phase C — η > 0 closed-loop weight trajectories", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "figure_W_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[learning] wrote {out}")


# --------------------------- Phase E: Tier-2 -----------------------------------
def analyze_tier2():
    raw = REPO / "results" / "overnight-tier2-patterns" / "raw"
    out_dir = raw.parent
    if not raw.exists() or not list(raw.glob("*.npz")):
        print("[tier2] no data in", raw)
        return
    files = sorted(raw.glob("*.npz"))
    print(f"[tier2] {len(files)} files")

    rows = []
    by_cell = defaultdict(list)  # cell -> list of (ts, fr_readout, W_in)
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        cell = str(z["cell_id"])
        ts = z["ts"]
        fr = np.asarray(z["fr_readout"]).ravel()
        W_in = np.asarray(z["W_readout_inputs"])  # (T, N_INPUTS)
        npp = int(z["N_inputs"]) // int(z["n_patterns"])
        # On-pattern (pattern 0) input weights mean
        W_target_mean = W_in[:, :npp].mean(axis=1)
        W_off_mean = W_in[:, npp:].mean(axis=1)
        rows.append({
            "cell": cell, "seed": int(z["seed"]),
            "fr_readout_init": float(fr[:50].mean()),
            "fr_readout_final": float(fr[-50:].mean()),
            "W_target_init": float(W_target_mean[0]),
            "W_target_final": float(W_target_mean[-1]),
            "W_off_init": float(W_off_mean[0]),
            "W_off_final": float(W_off_mean[-1]),
            "selectivity_final": float(W_target_mean[-1] - W_off_mean[-1]),
        })
        by_cell[cell].append((ts, fr, W_target_mean, W_off_mean))

    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[tier2] wrote summary.csv ({len(rows)} rows)")

    cells = sorted(by_cell.keys())
    fig, axes = plt.subplots(2, len(cells), figsize=(3 * len(cells), 5), squeeze=False)
    for j, cell in enumerate(cells):
        ax0 = axes[0, j]
        ax1 = axes[1, j]
        for ts, fr, Wt, Wo in by_cell[cell]:
            ax0.plot(ts, fr, alpha=0.6)
            ax1.plot(ts, Wt, color="C2", alpha=0.6, label="W_target")
            ax1.plot(ts, Wo, color="C1", alpha=0.6, label="W_off-pattern")
        ax0.set_title(f"{cell} — readout fr")
        ax0.set_ylabel("Hz")
        ax0.grid(alpha=0.3)
        ax1.set_title(f"{cell} — input weight selectivity")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("W")
        ax1.grid(alpha=0.3)
        if j == 0:
            ax1.legend(["target", "off"], fontsize=7)
    fig.suptitle("Phase E — Tier-2 pattern discrimination", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "figure_tier2.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[tier2] wrote {out}")


if __name__ == "__main__":
    analyze_learning_dynamics()
    analyze_tier2()
