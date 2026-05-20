"""Paired comparison v2 — Single Synapse Task with Alexander Eq. 33/34 metrics.

Improvements over v1:
- 6 cells (adds gate-off OUA control pair A0-PN, A0-PS).
- noise_scale_hyperparam = 2.0 (drives firing rate closer to Alexander Table 1 target).
- balance = 1.0 (heterosynaptic rebalancing enabled per Alexander §2.5 / Eq. 32).
- Saves the noiseless-twin spike train and full V[:, 0] trajectory for offline
  ρ / SNR computation per Alexander Eq. 33 / 34.
- Saves enough state to recompute the OUA eligibility-equivalent
  e_ij(t) = ζ_ij(t) · s_ij(t) · γ(V_i(t)) offline.

Cells:
  A-PN  : OUAMeanReversionLIFNetwork + NoisyNetwork (per-neuron OU)        gate ON
  A-PS  : OUAMeanReversionLIFNetwork + PerSynapseNoisyNetwork              gate ON
  A0-PN : OUAMeanReversionLIFNetwork + NoisyNetwork                        gate OFF (control)
  A0-PS : OUAMeanReversionLIFNetwork + PerSynapseNoisyNetwork              gate OFF (control)
  B-PN  : GatedLIFNetwork + NoisyNetwork                                   gate ON (Alexander's cell)
  B-PS  : PerSynapseGatedEligibilityLIFNetwork + PerSynapseNoisyNetwork    gate ON (co-headline)
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

from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork  # noqa: E402
from adaptive_SNN.models.networks.noisy_network import NoisyNetwork  # noqa: E402
from adaptive_SNN.models.networks.oua_LIF import OUAMeanReversionLIFNetwork  # noqa: E402
from adaptive_SNN.models.networks.oua_eligibility_LIF import (  # noqa: E402
    PerSynapseGatedEligibilityLIFNetwork,
)
from adaptive_SNN.models.networks.per_synapse_noisy_network import (  # noqa: E402
    PerSynapseNoisyNetwork,
)
from adaptive_SNN.models.noise.oup import NeuralNoiseOUP  # noqa: E402
from adaptive_SNN.models.noise.per_synapse_oup import PerSynapseOUP  # noqa: E402
from adaptive_SNN.solver import simulate_noisy_SNN  # noqa: E402

# ---------------------------------------------------------------------------
MASTER_SEED = 42
N_NEURONS = 2
N_INPUTS = 3
INPUT_RATES = jnp.array([5000.0, 1250.0, 10.0])
INPUT_TYPES = jnp.array([True, False, True])  # E, I, E
NOISE_SCALE_HYPERPARAM = 2.0  # bumped from Alexander's default 0.0; with min=5nS gives σ≈13nS
BALANCE_TARGET = 0.0          # match Alexander's SST config (initial weights pre-tuned)
TAU_RPE = 0.1
T_TOTAL = 60.0                # 60s window (Alexander uses 300s; we trade off budget)
DT = 1e-4
WARMUP_T = 10.0
N_SAVE = 2400                 # save every 25ms for fine-enough resolution

DELTA_V_SWEEP = jnp.array([2.0 ** -k for k in range(7, 14)])  # 2^-7..2^-13
SEEDS = [42, 43, 44, 45, 46]
CELL_IDS = ["A-PN", "A-PS", "A0-PN", "A0-PS", "B-PN", "B-PS"]
RESULTS_DIR = REPO_ROOT / "results" / "overnight-single-synapse-v2"
RAW_DIR = RESULTS_DIR / "raw"


def make_initial_weights():
    return jnp.tile(jnp.array([jnp.nan, jnp.nan, 1.1, 11.0, 0.0]), (N_NEURONS, 1))


def _input_spike_fn_factory(spike_key):
    rates = INPUT_RATES

    def fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint(t / DT), dtype=jnp.int64)
        spikes_1d = jr.poisson(
            jr.fold_in(spike_key, step_idx), rates * DT, shape=(1, N_INPUTS)
        )
        return jnp.tile(spikes_1d, (N_NEURONS, 1)).astype(jnp.float64)

    return fn


def _make_base_network(model_cls, delta_V, key, use_gating=True):
    kwargs = dict(
        dt=DT,
        N_neurons=N_NEURONS,
        N_inputs=N_INPUTS,
        initial_weight_matrix=make_initial_weights(),
        input_types=INPUT_TYPES,
        fully_connected_input=True,
        key=key,
    )
    if model_cls is GatedLIFNetwork:
        net = model_cls(**kwargs)
        object.__setattr__(net, "delta_V", float(delta_V))
        return net
    if model_cls is OUAMeanReversionLIFNetwork:
        return model_cls(
            **kwargs,
            delta_V=float(delta_V),
            use_gating=use_gating,
            update_clip=1.0,
        )
    if model_cls is PerSynapseGatedEligibilityLIFNetwork:
        return model_cls(**kwargs, delta_V=float(delta_V))
    raise ValueError(f"Unknown model_cls {model_cls}")


def _make_cell(cell_id, delta_V, sigma_pn, sigma_ps, key):
    net_key, _ = jr.split(key)
    if cell_id in ("A-PN", "A0-PN"):
        use_gating = (cell_id == "A-PN")
        net = _make_base_network(OUAMeanReversionLIFNetwork, delta_V, net_key, use_gating=use_gating)
        noise = NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn)
        return NoisyNetwork(net, noise, min_noise_std=5e-9), "PN"
    if cell_id in ("A-PS", "A0-PS"):
        use_gating = (cell_id == "A-PS")
        net = _make_base_network(OUAMeanReversionLIFNetwork, delta_V, net_key, use_gating=use_gating)
        noise = PerSynapseOUP(
            N_neurons=N_NEURONS, N_inputs=N_INPUTS,
            excitatory_mask=net.excitatory_mask,
            tau=net.tau_E, noise_std=sigma_ps,
        )
        return PerSynapseNoisyNetwork(net, noise, min_noise_std=5e-9), "PS"
    if cell_id == "B-PN":
        net = _make_base_network(GatedLIFNetwork, delta_V, net_key)
        noise = NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn)
        return NoisyNetwork(net, noise, min_noise_std=5e-9), "PN"
    if cell_id == "B-PS":
        net = _make_base_network(PerSynapseGatedEligibilityLIFNetwork, delta_V, net_key)
        noise = PerSynapseOUP(
            N_neurons=N_NEURONS, N_inputs=N_INPUTS,
            excitatory_mask=net.excitatory_mask,
            tau=net.tau_E, noise_std=sigma_ps,
        )
        return PerSynapseNoisyNetwork(net, noise, min_noise_std=5e-9), "PS"
    raise ValueError(f"Unknown cell {cell_id}")


def _make_save_fn(cell_id):
    def fn(t, y, args):
        inner = y.network_state
        common = {
            "V": inner.V,
            "S": inner.S,
            "G_learnable": jnp.atleast_1d(inner.G[0, 4]),
            "G_bg_E": jnp.atleast_1d(inner.G[0, 2]),    # 5 kHz E input
            "W_learnable": jnp.atleast_1d(inner.W[0, 4]),
            "noise_state": y.noise_state,
            "var_E_n": jnp.atleast_1d(inner.var_E_conductance[0]),
            "mean_E_n": jnp.atleast_1d(inner.mean_E_conductance[0]),
        }
        if cell_id.startswith("B-"):
            common["eligibility_learnable"] = jnp.atleast_1d(inner.features.eligibility[0, 4])
        return common

    return fn


def _build_args(spike_key, sigma_ps, learning_rate=0.0):
    return {
        "get_input_spikes": _input_spike_fn_factory(spike_key),
        "get_learning_rate": lambda t, x, a: jnp.array(learning_rate),
        "get_desired_balance": lambda t, x, a: jnp.array(BALANCE_TARGET),
        "noise_scale_hyperparam": NOISE_SCALE_HYPERPARAM,
        "use_noise": jnp.array([True, False]),
        "RPE": jnp.array(0.0),
        "per_synapse_noise_std_target": sigma_ps,
    }


def run_cell(cell_id, delta_V, seed, sigma_pn, sigma_ps, T=T_TOTAL, learning_rate=0.0):
    base_key = jr.PRNGKey(seed)
    cell_key, spike_key, brown_key = jr.split(base_key, 3)
    wrap, _ = _make_cell(cell_id, delta_V, sigma_pn, sigma_ps, cell_key)
    args = _build_args(spike_key, sigma_ps, learning_rate=learning_rate)
    save_ts = jnp.linspace(0.0, T, N_SAVE)
    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=_make_save_fn(cell_id)))
    return simulate_noisy_SNN(
        wrap, dfx.EulerHeun(), 0.0, T, DT, wrap.initial,
        save_at=saveat, args=args, key=brown_key,
    )


def calibrate(T_warm=20.0, seed=MASTER_SEED):
    """Variance-match calibration using B-PN at the desired operating regime."""
    key = jr.PRNGKey(seed)
    net_key, spike_key, brown_key = jr.split(key, 3)
    net = GatedLIFNetwork(
        N_neurons=N_NEURONS, N_inputs=N_INPUTS, dt=DT,
        initial_weight_matrix=make_initial_weights(), input_types=INPUT_TYPES,
        fully_connected_input=True, key=net_key,
    )
    object.__setattr__(net, "delta_V", 1e-3)
    noise = NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=5e-9)
    wrap = NoisyNetwork(net, noise, min_noise_std=5e-9)
    args = _build_args(spike_key, 0.0, learning_rate=0.0)
    save_ts = jnp.linspace(WARMUP_T, T_warm, 400)

    def save_fn(t, y, args):
        return {
            "G_E1": jnp.atleast_1d(y.network_state.G[0, 2]),
            "G_E2": jnp.atleast_1d(y.network_state.G[0, 4]),
            "var_E": jnp.atleast_1d(y.network_state.var_E_conductance[0]),
            "V": jnp.atleast_1d(y.network_state.V[0]),
            "S": jnp.atleast_1d(y.network_state.S[0]),
        }

    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=save_fn))
    sol = simulate_noisy_SNN(
        wrap, dfx.EulerHeun(), 0.0, T_warm, DT, wrap.initial,
        save_at=saveat, args=args, key=brown_key,
    )
    G_exc = np.column_stack([
        np.asarray(sol.ys["G_E1"]).ravel(),
        np.asarray(sol.ys["G_E2"]).ravel(),
    ]) / float(net.synaptic_increment)
    s2_mean = float(np.mean(G_exc ** 2))
    var_E_ss = float(np.mean(np.asarray(sol.ys["var_E"]).ravel()[-100:]))
    fr = float(np.mean(np.asarray(sol.ys["S"]).ravel()) / (T_warm - WARMUP_T) * len(np.asarray(sol.ys["S"]).ravel()))
    V_mean = float(np.mean(np.asarray(sol.ys["V"]).ravel()))
    sigma_pn = float(np.sqrt(var_E_ss) * NOISE_SCALE_HYPERPARAM + 5e-9)
    abs_E_i = 2.0
    sigma_ps = float(sigma_pn / (net.synaptic_increment * np.sqrt(abs_E_i * s2_mean)))
    return {
        "sigma_per_neuron": sigma_pn,
        "sigma_per_synapse": sigma_ps,
        "abs_E_i": abs_E_i,
        "s2_mean": s2_mean,
        "var_E_ss": var_E_ss,
        "fr_noisy_calib_warmup": fr,
        "V_mean_noisy_calib": V_mean,
        "noise_scale_hyperparam": NOISE_SCALE_HYPERPARAM,
        "balance_target": BALANCE_TARGET,
        "T_warm": T_warm,
        "master_seed": MASTER_SEED,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("OUA P3-v2 — Single Synapse Task, 6 cells, Eq. 33/34 metrics")
    print(f"noise_scale={NOISE_SCALE_HYPERPARAM}, balance={BALANCE_TARGET}, T={T_TOTAL}s, dt={DT}s")
    print(f"DV: {[float(x) for x in DELTA_V_SWEEP]}")
    print(f"Cells: {CELL_IDS}")
    print("=" * 78)

    t0 = time.time()
    calib = calibrate()
    print(f"[calib] {time.time()-t0:.1f}s")
    for k, v in calib.items():
        print(f"  {k}: {v}")
    with open(RESULTS_DIR / "calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)
    sigma_pn = calib["sigma_per_neuron"]
    sigma_ps = calib["sigma_per_synapse"]

    n_total = len(CELL_IDS) * len(DELTA_V_SWEEP) * len(SEEDS)
    print(f"[run] {n_total} simulations")
    elapsed = 0.0
    done = 0
    for cell_id in CELL_IDS:
        for k, dV in enumerate(DELTA_V_SWEEP):
            dV_pow = -7 - k
            for seed in SEEDS:
                tag = f"{cell_id}_dV2^{dV_pow}_seed{seed}"
                save_path = RAW_DIR / f"{tag}.npz"
                if save_path.exists():
                    done += 1
                    continue
                t0 = time.time()
                try:
                    sol = run_cell(cell_id, float(dV), seed, sigma_pn, sigma_ps, T=T_TOTAL)
                except Exception as exc:
                    print(f"[ERR] {tag}: {type(exc).__name__}: {str(exc)[:150]}")
                    continue
                wall = time.time() - t0
                elapsed += wall
                done += 1
                ts = np.asarray(sol.ts)
                ys = {k: np.asarray(v) for k, v in sol.ys.items()}
                np.savez(
                    save_path,
                    ts=ts, **ys,
                    cell_id=cell_id, delta_V=float(dV), seed=seed,
                    sigma_pn=sigma_pn, sigma_ps=sigma_ps,
                    master_seed=MASTER_SEED,
                    noise_scale_hyperparam=NOISE_SCALE_HYPERPARAM,
                    balance_target=BALANCE_TARGET,
                )
                fr = float(ys["S"][:, 0].sum() / T_TOTAL)
                print(f"[done {done}/{n_total}] {tag} wall={wall:.1f}s fr={fr:.2f}Hz")
    print(f"[total] {elapsed:.1f}s for new runs")


if __name__ == "__main__":
    main()
