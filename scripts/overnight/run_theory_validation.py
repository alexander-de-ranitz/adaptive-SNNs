"""Phase D — Theory validation experiments.

Three checks:

1. **τ_e → 0 equivalence (theorist Q9; unified §4.5)**
   Run B-PS (PerSynapseGatedEligibilityLIFNetwork) with τ_e ∈
   {τ_E, 3τ_E, 100ms} and compare W trajectories to A-PS (OUA mean-reversion,
   which is the τ_e → 0 limit). With τ_e = τ_E the two should be statistically
   indistinguishable on short windows; with τ_e = 100 ms eligibility wins on
   delayed-reward regimes (no explicit reward delay here so the gap is from
   the integral kernel itself).

2. **OUA Lyapunov check (theorist Q5)**
   Run A-PN with η > 0 at very small ΔV (where the gate spikes) — verify the
   γ-clip prevents W blow-up. Report `clip_engagement_fraction = fraction of
   timesteps where clip_factor < 1`.

3. **Stein-lemma bias diagnostic (theorist Q4)**
   From the v2 raw data, compute E[δ_r^task(t) | V_noisy(t) ∈ gate-active
   region]. If non-zero, the gated estimator is biased per Stein's lemma.

Output: results/overnight-theory-validation/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dfx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_SNN.models.networks.learning_agent import LearningAgent  # noqa: E402
from adaptive_SNN.models.networks.oua_LIF import OUAMeanReversionLIFNetwork  # noqa: E402
from adaptive_SNN.models.networks.oua_eligibility_LIF import (  # noqa: E402
    PerSynapseGatedEligibilityLIFNetwork,
)
from adaptive_SNN.models.networks.per_synapse_noisy_network import (  # noqa: E402
    PerSynapseNoisyNetwork,
)
from adaptive_SNN.models.networks.noisy_network import NoisyNetwork  # noqa: E402
from adaptive_SNN.models.noise.oup import NeuralNoiseOUP  # noqa: E402
from adaptive_SNN.models.noise.per_synapse_oup import PerSynapseOUP  # noqa: E402
from adaptive_SNN.solver import solve_ODE  # noqa: E402

MASTER_SEED = 42
N_NEURONS, N_INPUTS = 2, 3
INPUT_RATES = jnp.array([5000.0, 1250.0, 10.0])
INPUT_TYPES = jnp.array([True, False, True])
NOISE_SCALE_HYPERPARAM = 2.0
BALANCE_TARGET = 0.0
T_TOTAL = 30.0
DT = 1e-4
N_SAVE = 800

RESULTS_DIR = REPO_ROOT / "results" / "overnight-theory-validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_initial_weights():
    return jnp.tile(jnp.array([jnp.nan, jnp.nan, 1.1, 11.0, 0.0]), (N_NEURONS, 1))


def _input_spike_fn(spike_key):
    rates = INPUT_RATES

    def fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint(t / DT), dtype=jnp.int64)
        spikes_1d = jr.poisson(jr.fold_in(spike_key, step_idx), rates * DT, shape=(1, N_INPUTS))
        return jnp.tile(spikes_1d, (N_NEURONS, 1)).astype(jnp.float64)

    return fn


def _make_oua_per_synapse(delta_V, sigma_ps, key, use_gating=True, update_clip=1.0):
    net = OUAMeanReversionLIFNetwork(
        dt=DT, N_neurons=N_NEURONS, N_inputs=N_INPUTS,
        initial_weight_matrix=make_initial_weights(),
        input_types=INPUT_TYPES, fully_connected_input=True, key=key,
        delta_V=float(delta_V), use_gating=use_gating, update_clip=update_clip,
    )
    oup = PerSynapseOUP(N_neurons=N_NEURONS, N_inputs=N_INPUTS, excitatory_mask=net.excitatory_mask,
                        tau=net.tau_E, noise_std=sigma_ps)
    return PerSynapseNoisyNetwork(net, oup, min_noise_std=5e-9), net


def _make_oua_per_neuron(delta_V, sigma_pn, key, use_gating=True, update_clip=1.0):
    net = OUAMeanReversionLIFNetwork(
        dt=DT, N_neurons=N_NEURONS, N_inputs=N_INPUTS,
        initial_weight_matrix=make_initial_weights(),
        input_types=INPUT_TYPES, fully_connected_input=True, key=key,
        delta_V=float(delta_V), use_gating=use_gating, update_clip=update_clip,
    )
    return NoisyNetwork(net, NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn), min_noise_std=5e-9), net


def _make_elig_per_synapse(delta_V, sigma_ps, key, tau_eligibility=0.1):
    net = PerSynapseGatedEligibilityLIFNetwork(
        dt=DT, N_neurons=N_NEURONS, N_inputs=N_INPUTS,
        initial_weight_matrix=make_initial_weights(),
        input_types=INPUT_TYPES, fully_connected_input=True, key=key,
        tau_eligibility=tau_eligibility, delta_V=float(delta_V),
    )
    oup = PerSynapseOUP(N_neurons=N_NEURONS, N_inputs=N_INPUTS, excitatory_mask=net.excitatory_mask,
                        tau=net.tau_E, noise_std=sigma_ps)
    return PerSynapseNoisyNetwork(net, oup, min_noise_std=5e-9), net


def _build_args(spike_key, rpe_key, sigma_ps, lr):
    return {
        "get_input_spikes": _input_spike_fn(spike_key),
        "get_learning_rate": lambda t, x, a: jnp.array(lr),
        "get_desired_balance": lambda t, x, a: jnp.array(BALANCE_TARGET),
        "noise_scale_hyperparam": NOISE_SCALE_HYPERPARAM,
        "use_noise": jnp.array([True, False]),
        "per_synapse_noise_std_target": sigma_ps,
        "rpe_noise_key": rpe_key,
    }


def _save_fn(cell_id):
    def fn(t, y, args):
        inner = y.network_state.network_state
        return {
            "V": inner.V,
            "S": inner.S,
            "W_learn": jnp.atleast_1d(inner.W[0, 4]),
            "G_learn": jnp.atleast_1d(inner.G[0, 4]),
            "rpe": jnp.atleast_1d(y.rpe),
        }
    return fn


def run_one(name, agent, args, T):
    save_ts = jnp.linspace(0.0, T, N_SAVE)
    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=_save_fn(name)))
    key = jr.PRNGKey(7)
    return solve_ODE(agent, dfx.EulerHeun(), 0.0, T, DT, agent.initial,
                              save_at=saveat, args=args, key=key)


def experiment_tau_e_equivalence(sigma_pn, sigma_ps, seed=42, T=15.0, lr=50.0):
    """B-PS at τ_e ∈ {6ms, 30ms, 100ms} vs A-PS reference."""
    print(f"\n[exp1] τ_e equivalence: A-PS vs B-PS at varying τ_e (T={T}s, lr={lr})")
    base_key = jr.PRNGKey(seed)
    cell_key, spike_key, brown_key, rpe_key = jr.split(base_key, 4)
    delta_V = 2.0 ** -9
    results = {}
    # A-PS (OUA reference)
    noisy_aps, _ = _make_oua_per_synapse(delta_V, sigma_ps, cell_key, use_gating=True)
    args = _build_args(spike_key, rpe_key, sigma_ps, lr)
    agent = LearningAgent(noisy_aps)
    sol = run_one("A-PS", agent, args, T)
    results["A-PS"] = {k: np.asarray(v) for k, v in sol.ys.items()}
    results["ts"] = np.asarray(sol.ts)
    # B-PS at three τ_e values (same key, same setup)
    for tau_e_ms in [6, 30, 100]:
        tau_e = tau_e_ms * 1e-3
        noisy_bps, _ = _make_elig_per_synapse(delta_V, sigma_ps, cell_key, tau_eligibility=tau_e)
        agent = LearningAgent(noisy_bps)
        args = _build_args(spike_key, rpe_key, sigma_ps, lr)
        sol = run_one(f"B-PS-tau{tau_e_ms}ms", agent, args, T)
        results[f"B-PS_tau{tau_e_ms}ms"] = {k: np.asarray(v) for k, v in sol.ys.items()}
    np.savez(RESULTS_DIR / "tau_e_equivalence.npz", **{
        "ts": results["ts"],
        **{f"{name}__{k}": v for name in results if name != "ts" for k, v in results[name].items()}
    })
    print(f"  Saved {RESULTS_DIR / 'tau_e_equivalence.npz'}")
    # Quick summary: W_learn at end
    for name in ["A-PS", "B-PS_tau6ms", "B-PS_tau30ms", "B-PS_tau100ms"]:
        W = results[name]["W_learn"].ravel()
        print(f"  {name}: W[0]={W[0]:.3e} W[-1]={W[-1]:.3e}")


def experiment_lyapunov(sigma_pn, sigma_ps, seed=42, T=20.0, lr=20.0):
    """A-PN at very small ΔV — clip should engage; W should stay bounded."""
    print(f"\n[exp2] Lyapunov check: A-PN at tiny ΔV (T={T}s, lr={lr})")
    base_key = jr.PRNGKey(seed)
    cell_key, spike_key, brown_key, rpe_key = jr.split(base_key, 4)
    results = {}
    for dV_pow in [-13, -11, -9]:
        delta_V = 2.0 ** dV_pow
        noisy, _ = _make_oua_per_neuron(delta_V, sigma_pn, cell_key, use_gating=True, update_clip=0.5)
        agent = LearningAgent(noisy)
        args = _build_args(spike_key, rpe_key, sigma_ps, lr)
        sol = run_one(f"A-PN_dV2^{dV_pow}", agent, args, T)
        ys = {k: np.asarray(v) for k, v in sol.ys.items()}
        results[f"dV{dV_pow}"] = ys
        W = ys["W_learn"].ravel()
        print(f"  dV=2^{dV_pow}: W[0]={W[0]:.3e} W[-1]={W[-1]:.3e} max|W|={np.max(np.abs(W)):.3e}")
    np.savez(RESULTS_DIR / "lyapunov.npz", ts=np.asarray(sol.ts),
             **{f"{name}__{k}": v for name, ys in results.items() for k, v in ys.items()})
    print(f"  Saved {RESULTS_DIR / 'lyapunov.npz'}")


def experiment_stein_bias():
    """Compute E[δ_r^task | V in gate-active region] from v2 raw data."""
    print("\n[exp3] Stein-lemma bias diagnostic (from v2 data)")
    v2_raw = REPO_ROOT / "results" / "overnight-single-synapse-v2" / "raw"
    files = sorted(v2_raw.glob("A-PS_dV2^-9_seed*.npz")) + sorted(v2_raw.glob("B-PN_dV2^-9_seed*.npz"))
    if not files:
        print(f"  No v2 files found in {v2_raw} — skipping bias diagnostic")
        return
    rows = []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        ts = z["ts"]
        V = z["V"][:, 0]
        S = z["S"]
        delta_V = float(z["delta_V"])
        # Gate-active region: V in [V_th - 5*delta_V, V_th + 2*delta_V]
        V_th = -50e-3
        gate_active = (V > V_th - 5 * delta_V) & (V < V_th + 2 * delta_V)
        # δ_r^task = exp-filter of spike diff
        raw_rpe = (S[:, 0] > 0).astype(float) - (S[:, 1] > 0).astype(float)
        dt = float(ts[1] - ts[0])
        alpha = np.exp(-dt / 0.1)
        rpe = np.zeros_like(raw_rpe)
        for i in range(1, len(rpe)):
            rpe[i] = rpe[i - 1] * alpha + raw_rpe[i] * (1 - alpha)
        # Conditional mean of δ_r given V in gate-active
        if gate_active.sum() > 10:
            cond_mean = float(np.mean(rpe[gate_active]))
            marg_mean = float(np.mean(rpe))
        else:
            cond_mean = float("nan")
            marg_mean = float("nan")
        rows.append({
            "file": fp.name,
            "delta_V": delta_V,
            "n_gate_active_samples": int(gate_active.sum()),
            "cond_E_rpe_given_gate": cond_mean,
            "marg_E_rpe": marg_mean,
            "bias_proxy": cond_mean - marg_mean,
        })
    import csv
    with open(RESULTS_DIR / "stein_bias.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {RESULTS_DIR / 'stein_bias.csv'} ({len(rows)} files)")
    for r in rows[:6]:
        print(f"  {r['file']}: gate-active n={r['n_gate_active_samples']}, "
              f"E[δ_r|V in gate]={r['cond_E_rpe_given_gate']:.3e}, "
              f"E[δ_r]={r['marg_E_rpe']:.3e}, bias={r['bias_proxy']:.3e}")


def main():
    from run_paired_comparison_v2 import calibrate
    calib = calibrate()
    sigma_pn = calib["sigma_per_neuron"]
    sigma_ps = calib["sigma_per_synapse"]
    with open(RESULTS_DIR / "calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)

    experiment_tau_e_equivalence(sigma_pn, sigma_ps, T=15.0)
    experiment_lyapunov(sigma_pn, sigma_ps, T=20.0)
    experiment_stein_bias()


if __name__ == "__main__":
    main()
