"""Plot Phase D theory validation outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "results" / "overnight-theory-validation"


def plot_tau_e_equivalence():
    fp = DIR / "tau_e_equivalence.npz"
    if not fp.exists():
        print("[plot] no tau_e_equivalence.npz")
        return
    z = np.load(fp, allow_pickle=True)
    ts = z["ts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cells = ["A-PS", "B-PS_tau6ms", "B-PS_tau30ms", "B-PS_tau100ms"]
    colors = ["k", "C0", "C1", "C2"]
    labels = ["A-PS (OUA, τ_e=0)", "B-PS τ_e=6ms (=τ_E)", "B-PS τ_e=30ms", "B-PS τ_e=100ms"]
    for cell, color, label in zip(cells, colors, labels):
        W = np.asarray(z[f"{cell}__W_learn"]).ravel()
        axes[0].plot(ts, W, color=color, label=label)
    axes[0].set_title("τ_e → 0 equivalence (Q9): W_learnable(t)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("W")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale("symlog", linthresh=1e-3)

    # Eligibility comparison if available
    for cell, color, label in zip(cells, colors, labels):
        if f"{cell}__elig_learn" in z.files:
            e = np.asarray(z[f"{cell}__elig_learn"]).ravel()
            axes[1].plot(ts, e, color=color, label=label, alpha=0.7)
    axes[1].set_title("Eligibility trace at learnable synapse")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("e_ij")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = DIR / "figure_tau_e_equivalence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_lyapunov():
    fp = DIR / "lyapunov.npz"
    if not fp.exists():
        print("[plot] no lyapunov.npz")
        return
    z = np.load(fp, allow_pickle=True)
    ts = z["ts"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for dV_pow in [-13, -11, -9]:
        W = np.asarray(z[f"dV{dV_pow}__W_learn"]).ravel()
        ax.plot(ts, W, label=f"ΔV = 2^{dV_pow}")
    ax.set_title("Lyapunov check (Q5): A-PN W_learnable(t) under γ-clip")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("W")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DIR / "figure_lyapunov.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_stein_bias():
    import csv
    fp = DIR / "stein_bias.csv"
    if not fp.exists():
        print("[plot] no stein_bias.csv")
        return
    rows = list(csv.DictReader(open(fp)))
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    cells = sorted({r["file"].split("_dV")[0] for r in rows})
    for cell in cells:
        cells_rows = [r for r in rows if r["file"].startswith(cell)]
        biases = [float(r["bias_proxy"]) for r in cells_rows]
        ax.scatter([cell] * len(biases), biases, label=cell)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Stein-lemma bias: E[δ_r|V∈gate] − E[δ_r] at ΔV=2^-9")
    ax.set_ylabel("bias proxy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = DIR / "figure_stein_bias.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    plot_tau_e_equivalence()
    plot_lyapunov()
    plot_stein_bias()
