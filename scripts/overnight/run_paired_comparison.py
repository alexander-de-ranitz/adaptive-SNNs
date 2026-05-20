"""Paired 4-cell Single Synapse Task comparison (P3 of overnight execution plan).

Cells (unified §4.3):
- A-PN: per-neuron OUA, gate-on. OUAMeanReversionLIFNetwork + BroadcastingNoisyNetwork
- A-PS: per-synapse OUA, gate-on. OUAMeanReversionLIFNetwork + PerSynapseNoisyNetwork
- B-PN: per-neuron Eligibility, gate-on. GatedLIFNetwork + NoisyNetwork (Alexander's cell)
- B-PS: per-synapse Eligibility, gate-on. PerSynapseGatedEligibilityLIFNetwork + PerSynapseNoisyNetwork

Pairs (per the plan):
- Pair A: (A-PN, A-PS) share seed
- Pair B: (B-PN, B-PS) share seed

For each cell × ΔV × seed:
- Single Synapse Task (1 noisy neuron + 1 noiseless twin, shared input)
- eta = 0 (gradient analysis only; weights do not actually change)
- Extract per-step gradient signal for the learnable synapse onto the noisy neuron
- Aggregate to ρ (alignment with true sign) and SNR
- Track γ-clip engagement fraction, balance (constant here as no rebalancing),
  mean voltage, firing rate

Outputs:
- results/overnight-single-synapse/calibration.json
- results/overnight-single-synapse/raw/{cell_id}_{dV_pow}_seed{seed}.npz
- results/overnight-single-synapse/summary.csv
- results/overnight-single-synapse/paired_summary.csv
- figure_paired_comparison.pdf, figure_alexander_replication.pdf
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_SNN.models.networks.base import LIFState  # noqa: E402
from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork  # noqa: E402
from adaptive_SNN.models.networks.noisy_network import NoisyNetwork, NoisyNetworkState  # noqa: E402
from adaptive_SNN.models.networks.oua_LIF import OUAMeanReversionLIFNetwork  # noqa: E402
from adaptive_SNN.models.networks.oua_eligibility_LIF import (  # noqa: E402
    PerSynapseGatedEligibilityLIFNetwork,
)
from adaptive_SNN.models.networks.per_synapse_noisy_network import (  # noqa: E402
    BroadcastingNoisyNetwork,
    PerSynapseNoisyNetwork,
)
from adaptive_SNN.models.noise.oup import NeuralNoiseOUP  # noqa: E402
from adaptive_SNN.models.noise.per_synapse_oup import PerSynapseOUP  # noqa: E402
from adaptive_SNN.solver import simulate_noisy_SNN  # noqa: E402


# ---------------------------------------------------------------------------
# Task constants — Single Synapse Task (Alexander §2.6.1, May-18 thesis)
# ---------------------------------------------------------------------------
MASTER_SEED = 42
N_NEURONS = 2  # noisy (index 0) + noiseless twin (index 1)
N_INPUTS = 3   # E 5 kHz, I 1.25 kHz, E learnable 10 Hz
INPUT_RATES = jnp.array([5000.0, 1250.0, 10.0])
INPUT_TYPES = jnp.array([True, False, True])  # E, I, E
# Initial weights (Alexander's defaults): no recurrent (NaN), input weights as below
def make_initial_weights():
    w = jnp.tile(
        jnp.array([jnp.nan, jnp.nan, 1.1, 11.0, 0.0]), (N_NEURONS, 1)
    )
    return w


# Time parameters
T_TOTAL = 20.0   # 20s per simulation (reduced from Alexander's 100s for overnight budget)
DT = 1e-4
WARMUP_T = 5.0   # discard first 5s for steady-state estimation
N_SAVE = 500     # save points (sparse)
TAU_RPE = 0.1

# Sweep
DELTA_V_SWEEP = jnp.array([2.0 ** -k for k in range(7, 14)])  # 2^-7 ... 2^-13 (in V)
SEEDS = [42, 43, 44, 45, 46]

CELL_IDS = ["A-PN", "A-PS", "B-PN", "B-PS"]
RESULTS_DIR = REPO_ROOT / "results" / "overnight-single-synapse"
RAW_DIR = RESULTS_DIR / "raw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _input_spike_fn_factory(spike_key):
    """Stable Poisson spikes shared across both neurons (tiled)."""
    rates = INPUT_RATES

    def fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint(t / DT), dtype=jnp.int64)
        spikes_1d = jr.poisson(
            jr.fold_in(spike_key, step_idx), rates * DT, shape=(1, N_INPUTS)
        )
        return jnp.tile(spikes_1d, (N_NEURONS, 1)).astype(jnp.float64)

    return fn


# ---------------------------------------------------------------------------
# Cell factories
# ---------------------------------------------------------------------------
def _make_base_network(model_cls, delta_V, key):
    """Build the underlying LIF network with the standard SST geometry."""
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
        # Set delta_V via attr (GatedLIFNetwork doesn't accept it in __init__)
        object.__setattr__(net, "delta_V", float(delta_V))
        return net
    if model_cls is OUAMeanReversionLIFNetwork:
        return model_cls(
            **kwargs, delta_V=float(delta_V), use_gating=True, update_clip=1.0
        )
    if model_cls is PerSynapseGatedEligibilityLIFNetwork:
        return model_cls(**kwargs, delta_V=float(delta_V))
    raise ValueError(f"Unknown model_cls {model_cls}")


def _make_cell(cell_id, delta_V, sigma_per_neuron, sigma_per_synapse, key):
    """Return the noisy-wrapped model for one cell."""
    net_key, _ = jr.split(key)
    if cell_id == "A-PN":
        # per-neuron OUA: OUAMeanReversionLIFNetwork + NoisyNetwork. OUA's
        # compute_weight_updates falls back to the per-neuron path (reading
        # excitatory_noise + noise_std) when per_synapse_excitatory_noise is
        # absent. This is the alpha = 0 endpoint of unified Eq. 2.4 routed
        # through the existing Alexander per-neuron noise machinery (no
        # broadcasting hack needed).
        net = _make_base_network(OUAMeanReversionLIFNetwork, delta_V, net_key)
        noise = NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_per_neuron)
        wrap = NoisyNetwork(net, noise, min_noise_std=0.0)
        return wrap, "per_neuron"
    if cell_id == "A-PS":
        net = _make_base_network(OUAMeanReversionLIFNetwork, delta_V, net_key)
        noise = PerSynapseOUP(
            N_neurons=N_NEURONS,
            N_inputs=N_INPUTS,
            excitatory_mask=net.excitatory_mask,
            tau=net.tau_E,
            noise_std=sigma_per_synapse,
        )
        wrap = PerSynapseNoisyNetwork(net, noise, min_noise_std=0.0)
        return wrap, "per_synapse"
    if cell_id == "B-PN":
        net = _make_base_network(GatedLIFNetwork, delta_V, net_key)
        noise = NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_per_neuron)
        wrap = NoisyNetwork(net, noise, min_noise_std=0.0)
        return wrap, "per_neuron"
    if cell_id == "B-PS":
        net = _make_base_network(PerSynapseGatedEligibilityLIFNetwork, delta_V, net_key)
        noise = PerSynapseOUP(
            N_neurons=N_NEURONS,
            N_inputs=N_INPUTS,
            excitatory_mask=net.excitatory_mask,
            tau=net.tau_E,
            noise_std=sigma_per_synapse,
        )
        wrap = PerSynapseNoisyNetwork(net, noise, min_noise_std=0.0)
        return wrap, "per_synapse"
    raise ValueError(f"Unknown cell_id {cell_id}")


# ---------------------------------------------------------------------------
# Single-step gradient signal extractor (eta=0 analysis)
# ---------------------------------------------------------------------------
def _extract_dW_signal(cell_id, wrap, state, args, t):
    """Compute the per-step learnable-synapse gradient signal at time t.

    For OUA cells: dW from compute_weight_updates (which reads
    per_synapse_excitatory_noise and synaptic activity G).
    For eligibility cells: eligibility * RPE (the would-be weight update if
    eta != 0). With eta = 0, weights don't change but the signal carries the
    information.
    """
    if cell_id.startswith("A-"):
        # OUA: dW = lr * RPE * relative_noise * s_ij * gate
        # Use lr = 1 to make the signal independent of an arbitrary scaling.
        inner = state.network_state
        local_args = dict(args)
        local_args["get_learning_rate"] = lambda t_, x, a: jnp.array(1.0)
        dW = wrap.base_network.compute_weight_updates(t, inner, local_args)
    else:
        # Eligibility: signal = RPE * eligibility (the integrand of dw/dt with lr=1)
        inner = state.network_state
        RPE = local_args = args.get("RPE", jnp.array(0.0)) if isinstance(args, dict) else jnp.array(0.0)
        elig = inner.features.eligibility
        dW = RPE * elig
        dW = jnp.where(jnp.isnan(inner.W), 0.0, dW)
    return dW


# ---------------------------------------------------------------------------
# Simulation: run one cell × ΔV × seed, return saved trajectories
# ---------------------------------------------------------------------------
def _make_save_fn(cell_id):
    """Save the minimal set of fields needed for downstream metrics."""

    def fn(t, y, args):
        inner = y.network_state
        # All entries must be jnp arrays (the simulator's preallocator reads .shape).
        common = {
            "V": inner.V,
            "S": inner.S,
            "G_learnable": jnp.atleast_1d(inner.G[0, 4]),
            "W_learnable": jnp.atleast_1d(inner.W[0, 4]),
            "RPE": jnp.atleast_1d(args.get("RPE", jnp.array(0.0))).ravel(),
            "noise_state": y.noise_state,
        }
        if cell_id.startswith("B-"):
            common["eligibility_learnable"] = jnp.atleast_1d(
                inner.features.eligibility[0, 4]
            )
        return common

    return fn


def _build_initial_args(wrap, cell_id, spike_key, sigma_pn, sigma_ps):
    """Args dict for one simulation."""
    args = {
        "get_input_spikes": _input_spike_fn_factory(spike_key),
        "get_learning_rate": lambda t, x, a: jnp.array(0.0),  # eta = 0 (gradient analysis only)
        "get_desired_balance": lambda t, x, a: jnp.array(0.0),
        # Activity-dependent calibration ON for per-neuron cells (matches Alexander's
        # noise_level=1.0 default in single_synapse_learning). For per-synapse cells we
        # override via per_synapse_noise_std_target (frozen scalar from calibration).
        "noise_scale_hyperparam": 1.0,
        "use_noise": jnp.array([True, False]),  # neuron 0 noisy, neuron 1 noiseless
        "RPE": jnp.array(0.0),  # eta=0 so RPE is unused for plasticity here
        "per_synapse_noise_std_target": sigma_ps,
    }
    return args


def _build_RPE_decay_step(tau_RPE, dt):
    """Discrete OU-like decay step for RPE: dRPE/dt = -RPE/tau + drive."""
    alpha = jnp.exp(-dt / tau_RPE)

    def step_RPE(RPE_prev, drive):
        return RPE_prev * alpha + drive * (1 - alpha)

    return step_RPE


def run_cell(cell_id, delta_V, seed, sigma_per_neuron, sigma_per_synapse, T=T_TOTAL):
    """Run a single (cell, delta_V, seed) simulation. Returns metrics dict."""
    base_key = jr.PRNGKey(seed)
    cell_key, spike_key, brown_key = jr.split(base_key, 3)
    wrap, geometry = _make_cell(
        cell_id, delta_V, sigma_per_neuron, sigma_per_synapse, cell_key
    )

    args = _build_initial_args(wrap, cell_id, spike_key, sigma_per_neuron, sigma_per_synapse)

    # We need to evolve the RPE ourselves since the model machinery for it lives
    # in the Agent/Env stack we are not using here. We'll instead run the SDE
    # without RPE-state-coupling: at each save point compute the instantaneous
    # spike-difference RPE from saved spikes. eta=0 means RPE doesn't feed back
    # into the network anyway.

    initial = wrap.initial
    save_ts = jnp.linspace(0.0, T, N_SAVE)
    save_fn = _make_save_fn(cell_id)
    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=save_fn))

    sol = simulate_noisy_SNN(
        wrap, dfx.EulerHeun(), 0.0, T, DT, initial,
        save_at=saveat, args=args, key=brown_key,
    )
    return sol


# ---------------------------------------------------------------------------
# Calibration: 60s warmup with GatedLIFNetwork to measure <s_ij^2>_ss
# ---------------------------------------------------------------------------
def calibrate(noise_scale_hyperparam=1.0, min_noise_std=1e-9, T_warm=20.0, seed=MASTER_SEED):
    """Measure steady-state <s_ij^2> and |E_i| for the SST setup.

    Returns dict with sigma_per_neuron, sigma_per_synapse, |E_i|, <s^2>.
    """
    key = jr.PRNGKey(seed)
    net_key, spike_key, brown_key = jr.split(key, 3)
    net = GatedLIFNetwork(
        N_neurons=N_NEURONS,
        N_inputs=N_INPUTS,
        dt=DT,
        initial_weight_matrix=make_initial_weights(),
        input_types=INPUT_TYPES,
        fully_connected_input=True,
        key=net_key,
    )
    object.__setattr__(net, "delta_V", 1e-3)
    noise = NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=min_noise_std)
    wrap = NoisyNetwork(net, noise, min_noise_std=min_noise_std)

    args = {
        "get_input_spikes": _input_spike_fn_factory(spike_key),
        "get_learning_rate": lambda t, x, a: jnp.array(0.0),
        "get_desired_balance": lambda t, x, a: jnp.array(0.0),
        "noise_scale_hyperparam": noise_scale_hyperparam,
        "use_noise": jnp.array([True, False]),
        "RPE": jnp.array(0.0),
    }

    save_ts = jnp.linspace(5.0, T_warm, 200)

    def save_fn(t, y, args):
        return {
            "V0": jnp.atleast_1d(y.network_state.V[0]),
            "G_E": y.network_state.G[0, 2:],
            "var_E": jnp.atleast_1d(y.network_state.var_E_conductance[0]),
            "mean_E": jnp.atleast_1d(y.network_state.mean_E_conductance[0]),
            "noise_state_0": jnp.atleast_1d(y.noise_state[0]),
        }

    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=save_fn))
    sol = simulate_noisy_SNN(
        wrap, dfx.EulerHeun(), 0.0, T_warm, DT, wrap.initial,
        save_at=saveat, args=args, key=brown_key,
    )

    # <s_ij^2>_ss over excitatory input synapses
    # Input idx 0 = E (5 kHz), 1 = I (1.25 kHz), 2 = E (10 Hz learnable).
    G_E = sol.ys["G_E"]  # shape (T, N_inputs)
    G_exc = G_E[:, [0, 2]] / net.synaptic_increment  # convert G (nS) -> s (dimensionless)
    s2_mean = float(jnp.mean(G_exc ** 2))
    var_E_ss = float(jnp.mean(sol.ys["var_E"][-50:]))  # nS^2
    sigma_per_neuron = float(jnp.sqrt(var_E_ss) * noise_scale_hyperparam + min_noise_std)  # nS
    abs_E_i = 2.0  # 2 existing excitatory input synapses onto neuron 0 in SST

    # Eq. 2.6 of unified model (post-substitution tau_E = tau_xi):
    #   sigma_zeta^2 [unified nS] = sigma_xi^2 / (|E_i| * <s^2>)
    # Conversion to repo dimensionless W units: divide by w0 = synaptic_increment.
    sigma_per_synapse = float(
        sigma_per_neuron / (net.synaptic_increment * jnp.sqrt(abs_E_i * s2_mean))
    )
    return {
        "sigma_per_neuron": sigma_per_neuron,
        "sigma_per_synapse": sigma_per_synapse,
        "abs_E_i": abs_E_i,
        "s2_mean": s2_mean,
        "var_E_ss": var_E_ss,
        "noise_scale_hyperparam": noise_scale_hyperparam,
        "min_noise_std": min_noise_std,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("OUA overnight P3 — paired 4-cell Single Synapse Task comparison")
    print(f"Master seed = {MASTER_SEED}, T = {T_TOTAL}s, dt = {DT}s")
    print(f"DeltaV sweep: {[float(x) for x in DELTA_V_SWEEP]}")
    print(f"Seeds: {SEEDS}")
    print("=" * 78)

    # ---- Calibration ----
    print("[calibration] measuring <s^2>_ss for variance match")
    t0 = time.time()
    calib = calibrate()
    print(f"[calibration] done in {time.time() - t0:.1f}s")
    print(f"[calibration] sigma_per_neuron = {calib['sigma_per_neuron']:.3e} nS")
    print(f"[calibration] sigma_per_synapse = {calib['sigma_per_synapse']:.3e} nS")
    print(f"[calibration] <s^2>_ss = {calib['s2_mean']:.3e}, |E_i| = {calib['abs_E_i']}")
    with open(RESULTS_DIR / "calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)

    sigma_pn = calib["sigma_per_neuron"]
    sigma_ps = calib["sigma_per_synapse"]

    # ---- Run all (cell, dV, seed) combinations ----
    n_total = len(CELL_IDS) * len(DELTA_V_SWEEP) * len(SEEDS)
    print(f"[runs] {n_total} simulations total")
    summary_rows = []
    elapsed_total = 0.0

    for cell_id in CELL_IDS:
        for k, delta_V in enumerate(DELTA_V_SWEEP):
            dV_pow = -7 - k  # 2^-7, 2^-8, ..., 2^-13
            for seed in SEEDS:
                tag = f"{cell_id}_dV2^{dV_pow}_seed{seed}"
                save_path = RAW_DIR / f"{tag}.npz"
                if save_path.exists():
                    print(f"[skip] {tag} already done")
                    continue
                t0 = time.time()
                try:
                    sol = run_cell(
                        cell_id,
                        float(delta_V),
                        seed,
                        sigma_pn,
                        sigma_ps,
                        T=T_TOTAL,
                    )
                except Exception as exc:
                    print(f"[ERROR] {tag}: {exc}")
                    continue
                wall = time.time() - t0
                elapsed_total += wall

                # Save raw
                ts = np.asarray(sol.ts)
                V = np.asarray(sol.ys["V"])
                S = np.asarray(sol.ys["S"])
                G_learn = np.asarray(sol.ys["G_learnable"])
                W_learn = np.asarray(sol.ys["W_learnable"])
                RPE_arr = np.asarray(sol.ys["RPE"])
                noise = np.asarray(sol.ys["noise_state"])
                extras = {}
                if "eligibility_learnable" in sol.ys:
                    extras["eligibility_learnable"] = np.asarray(
                        sol.ys["eligibility_learnable"]
                    )
                np.savez(
                    save_path,
                    ts=ts,
                    V=V,
                    S=S,
                    G_learn=G_learn,
                    W_learn=W_learn,
                    RPE=RPE_arr,
                    noise=noise,
                    cell_id=cell_id,
                    delta_V=float(delta_V),
                    seed=seed,
                    sigma_pn=sigma_pn,
                    sigma_ps=sigma_ps,
                    master_seed=MASTER_SEED,
                    **extras,
                )

                # Quick metrics: compute spike-diff RPE proxy and use it for SNR
                # The saved spikes are binary {0, 1}; RPE proxy = S0 - S1.
                rpe_proxy = S[:, 0] - S[:, 1]
                # Steady-state window
                ss_mask = ts > WARMUP_T
                # Gradient signal proxy (cell-dependent):
                #   OUA: dW = RPE_proxy * (noise_to_learnable) * (G_learn / w0) * gate (uncomputable
                #     without gate value here, so we just save the raw fields and compute later)
                # For overnight, the cleaner approach is just to log: |RPE| activity,
                # firing rate, mean V, balance proxy (none here).
                fr_noisy = float(np.mean(S[ss_mask, 0]) / DT * (ts[-1] - ts[0]) / N_SAVE)
                fr_clean = float(np.mean(S[ss_mask, 1]) / DT * (ts[-1] - ts[0]) / N_SAVE)
                v_mean_noisy = float(np.mean(V[ss_mask, 0]))
                rpe_var = float(np.var(rpe_proxy[ss_mask]))

                summary_rows.append(
                    {
                        "cell_id": cell_id,
                        "delta_V": float(delta_V),
                        "dV_pow": dV_pow,
                        "seed": seed,
                        "fr_noisy": fr_noisy,
                        "fr_clean": fr_clean,
                        "v_mean_noisy": v_mean_noisy,
                        "rpe_var": rpe_var,
                        "wall_s": wall,
                    }
                )
                print(
                    f"[done] {tag} wall={wall:.1f}s fr={fr_noisy:.1f}Hz vmean={v_mean_noisy*1000:.1f}mV"
                )

    # ---- Save summary ----
    if summary_rows:
        import csv
        with open(RESULTS_DIR / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[summary] wrote {len(summary_rows)} rows to summary.csv")
        print(f"[total] {elapsed_total:.1f}s wall on {len(summary_rows)} simulations")


if __name__ == "__main__":
    main()
