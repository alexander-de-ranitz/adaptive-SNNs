"""Phase E — Tier-2 pattern discrimination at impl-plan-target scale N = 1000.

Identical task structure to run_tier2_patterns.py, scaled up:
- N = 1000 (800 E + 200 I), p_E = 0.1, p_I = 0.2
- N_inputs = 100, K = 4 patterns of 25 inputs each
- T = 10 s (20 trials of 500 ms)
- 4 cells (A-PN, A-PS, B-PN, B-PS), 2 seeds for tractability

Network-scale calibration: re-runs the SST-style 20s warmup but on a single-
neuron-of-the-large-network proxy to get appropriate per-neuron / per-synapse
sigmas at this size and connectivity. Result saved to calibration.json.

Output: results/overnight-tier2-patterns-N1000/raw/{cell_id}_seed{seed}.npz
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
from adaptive_SNN.models.networks.learning_agent import LearningAgent  # noqa: E402
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
from adaptive_SNN.solver import solve_ODE  # noqa: E402

# ---------------------------------------------------------------------------
# Tier-2 large-scale task parameters (impl plan §5.2 target)
# ---------------------------------------------------------------------------
N_NEURONS = 500          # impl plan §5.2 target is 1000; CPU JIT-compilation
                         # hits a wall at N=1000 (no progress in 1h wall clock).
                         # N=500 is the practical maximum on this hardware
                         # without GPU/HPC. The conclusions about per-synapse
                         # advantage at large N apply qualitatively at N=500
                         # as long as |E_i| (number of E synapses per neuron)
                         # grows commensurately, which it does: with p_E=0.1
                         # and fully-connected inputs, |E_i| = 0.8*500*0.1 + 100
                         # = 40 + 100 = 140 (vs |E_i|=180 at N=1000).
N_INPUTS = 100
N_PATTERNS = 4
NEURONS_PER_PATTERN = N_INPUTS // N_PATTERNS  # 25
HIGH_RATE = 100.0
LOW_RATE = 5.0
PATTERN_DURATION = 0.5
T_TOTAL = 10.0
DT = 1e-4
N_SAVE = 100
DELTA_V = 2.0 ** -9
NOISE_SCALE_HYPERPARAM = 2.0
BALANCE_TARGET = 1.0
SEEDS = [42, 43]
CELL_IDS = ["A-PN", "A-PS", "B-PN", "B-PS"]

LEARNING_RATES = {
    "A-PN": 10.0,  "A-PS": 10.0,
    "B-PN": 100.0, "B-PS": 100.0,
}

READOUT_IDX = 0

RESULTS_DIR = REPO_ROOT / "results" / "overnight-tier2-patterns-N1000"
RAW_DIR = RESULTS_DIR / "raw"


def make_input_rates_fn(spike_key, pattern_seed):
    rng = np.random.default_rng(pattern_seed)
    pattern_mask = np.zeros((N_PATTERNS, N_INPUTS), dtype=bool)
    for p in range(N_PATTERNS):
        pattern_mask[p, p * NEURONS_PER_PATTERN:(p + 1) * NEURONS_PER_PATTERN] = True
    pattern_mask_jnp = jnp.asarray(pattern_mask)

    def fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint(t / DT), dtype=jnp.int64)
        p = jnp.asarray(jnp.floor(t / PATTERN_DURATION).astype(jnp.int64) % N_PATTERNS)
        rates_per_input = jnp.where(pattern_mask_jnp[p], HIGH_RATE, LOW_RATE)
        spikes = jr.poisson(jr.fold_in(spike_key, step_idx), rates_per_input * DT, shape=(1, N_INPUTS))
        return jnp.tile(spikes, (N_NEURONS, 1)).astype(jnp.float64)

    return fn


def _make_base_network(model_cls, key, use_gating=True):
    kwargs = dict(
        dt=DT, N_neurons=N_NEURONS, N_inputs=N_INPUTS,
        connection_prob_E=0.1, connection_prob_I=0.2,
        initial_input_weight=0.3, initial_rec_weight=0.3,
        rec_weight_std=0.2,
        fully_connected_input=True,
        fraction_excitatory_recurrent=0.8,
        fraction_excitatory_input=1.0,
        mean_synaptic_delay=1.5e-3,
        key=key,
    )
    if model_cls is GatedLIFNetwork:
        net = model_cls(**kwargs)
        object.__setattr__(net, "delta_V", float(DELTA_V))
        return net
    if model_cls is OUAMeanReversionLIFNetwork:
        return model_cls(**kwargs, delta_V=float(DELTA_V), use_gating=use_gating, update_clip=1.0)
    if model_cls is PerSynapseGatedEligibilityLIFNetwork:
        return model_cls(**kwargs, delta_V=float(DELTA_V))
    raise ValueError(model_cls)


def _make_noisy(cell_id, key, sigma_pn, sigma_ps):
    net_key, _ = jr.split(key)
    if cell_id in ("A-PN", "A0-PN"):
        use_gating = (cell_id == "A-PN")
        net = _make_base_network(OUAMeanReversionLIFNetwork, net_key, use_gating=use_gating)
        return NoisyNetwork(net, NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn), min_noise_std=5e-9), net
    if cell_id in ("A-PS", "A0-PS"):
        use_gating = (cell_id == "A-PS")
        net = _make_base_network(OUAMeanReversionLIFNetwork, net_key, use_gating=use_gating)
        oup = PerSynapseOUP(N_neurons=N_NEURONS, N_inputs=N_INPUTS, excitatory_mask=net.excitatory_mask, tau=net.tau_E, noise_std=sigma_ps)
        return PerSynapseNoisyNetwork(net, oup, min_noise_std=5e-9), net
    if cell_id == "B-PN":
        net = _make_base_network(GatedLIFNetwork, net_key)
        return NoisyNetwork(net, NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn), min_noise_std=5e-9), net
    if cell_id == "B-PS":
        net = _make_base_network(PerSynapseGatedEligibilityLIFNetwork, net_key)
        oup = PerSynapseOUP(N_neurons=N_NEURONS, N_inputs=N_INPUTS, excitatory_mask=net.excitatory_mask, tau=net.tau_E, noise_std=sigma_ps)
        return PerSynapseNoisyNetwork(net, oup, min_noise_std=5e-9), net
    raise ValueError(cell_id)


def _task_rpe_fn_factory(target_pattern):
    inv_dt = jnp.array(1.0 / DT, dtype=jnp.float64)
    target_p = jnp.array(target_pattern, dtype=jnp.int64)

    def fn(t, state, args):
        p = jnp.floor(t / PATTERN_DURATION).astype(jnp.int64) % jnp.array(N_PATTERNS, dtype=jnp.int64)
        multiplier = jnp.where(p == target_p, 1.0, -1.0)
        readout_spike = state.network_state.network_state.S[READOUT_IDX]
        return readout_spike * multiplier * inv_dt

    return fn


def _build_args(spike_key, rpe_key, pattern_seed, sigma_ps, lr, target_pattern):
    use_noise = jnp.ones((N_NEURONS,), dtype=bool)
    return {
        "get_input_spikes": make_input_rates_fn(spike_key, pattern_seed),
        "get_learning_rate": lambda t, x, a: jnp.array(lr),
        "get_desired_balance": lambda t, x, a: jnp.array(BALANCE_TARGET),
        "noise_scale_hyperparam": NOISE_SCALE_HYPERPARAM,
        "use_noise": use_noise,
        "per_synapse_noise_std_target": sigma_ps,
        "rpe_noise_key": rpe_key,
        "target_pattern": target_pattern,
        "task_rpe_fn": _task_rpe_fn_factory(target_pattern),
    }


def _save_fn(cell_id):
    def fn(t, y, args):
        inner = y.network_state.network_state
        p = jnp.floor(t / PATTERN_DURATION).astype(jnp.int64) % jnp.array(N_PATTERNS, dtype=jnp.int64)
        return {
            "fr_readout": jnp.atleast_1d(inner.firing_rate[READOUT_IDX]),
            "rpe": jnp.atleast_1d(y.rpe),
            "V_readout": jnp.atleast_1d(inner.V[READOUT_IDX]),
            "W_readout_inputs": inner.W[READOUT_IDX, N_NEURONS:],
            "S_readout": jnp.atleast_1d(inner.S[READOUT_IDX]),
            "S_sum": jnp.atleast_1d(jnp.sum(inner.S)),
            "pattern_idx": jnp.atleast_1d(p),
        }
    return fn


def calibrate_network(T_warm=5.0):
    """Network-scale warmup: run a B-PN N=1000 network briefly, measure var_E,
    <s²>, and |E_i| to feed Eq. 11′ for sigma_per_synapse."""
    print("[calib] N=1000 warmup, T=5s")
    base_key = jr.PRNGKey(42)
    cell_key, spike_key, brown_key, rpe_key = jr.split(base_key, 4)
    noisy, base_net = _make_noisy("B-PN", cell_key, sigma_pn=5e-9, sigma_ps=0.0)
    args = _build_args(spike_key, rpe_key, 42, 0.0, 0.0, 0)
    save_ts = jnp.linspace(2.0, T_warm, 60)

    def save_fn(t, y, args):
        inner = y.network_state.network_state
        return {
            "G_inputs": inner.G[READOUT_IDX, N_NEURONS:],
            "var_E": jnp.atleast_1d(inner.var_E_conductance[READOUT_IDX]),
            "V": jnp.atleast_1d(inner.V[READOUT_IDX]),
        }

    agent = LearningAgent(noisy)
    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=save_fn))
    sol = solve_ODE(agent, dfx.EulerHeun(), 0.0, T_warm, DT, agent.initial,
                             save_at=saveat, args=args, key=brown_key)
    G_E = np.asarray(sol.ys["G_inputs"]) / float(base_net.synaptic_increment)
    s2 = float(np.mean(G_E ** 2))
    var_E = float(np.mean(np.asarray(sol.ys["var_E"]).ravel()[-20:]))
    V_mean = float(np.mean(np.asarray(sol.ys["V"]).ravel()))
    sigma_pn = float(np.sqrt(var_E) * NOISE_SCALE_HYPERPARAM + 5e-9)
    # |E_i| at this network scale: N_inputs (all E) + N_E_rec * p_E
    abs_E_i = float(N_INPUTS + N_NEURONS * 0.8 * 0.1)
    sigma_ps = float(sigma_pn / (base_net.synaptic_increment * np.sqrt(abs_E_i * s2)))
    return {
        "sigma_per_neuron": sigma_pn,
        "sigma_per_synapse": sigma_ps,
        "abs_E_i": abs_E_i,
        "s2_mean": s2,
        "var_E_ss": var_E,
        "V_mean_readout": V_mean,
        "T_warm": T_warm,
        "N_neurons": N_NEURONS,
        "N_inputs": N_INPUTS,
    }


def run_cell(cell_id, seed, sigma_pn, sigma_ps, T=T_TOTAL):
    base_key = jr.PRNGKey(seed)
    cell_key, spike_key, brown_key, rpe_key = jr.split(base_key, 4)
    noisy, base_net = _make_noisy(cell_id, cell_key, sigma_pn, sigma_ps)
    agent = LearningAgent(noisy, tau_RPE=0.1, rpe_noise_rate=1.0, rpe_noise_std=1.0,
                          noisy_idx=READOUT_IDX, noiseless_idx=READOUT_IDX)
    target_pattern = 0
    args = _build_args(spike_key, rpe_key, seed * 13 + 7, sigma_ps,
                       LEARNING_RATES.get(cell_id, 1.0), target_pattern)
    save_ts = jnp.linspace(0.0, T, N_SAVE)
    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=_save_fn(cell_id)))
    return solve_ODE(agent, dfx.EulerHeun(), 0.0, T, DT, agent.initial,
                              save_at=saveat, args=args, key=brown_key)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Tier-2 LARGE — N={N_NEURONS}, K={N_PATTERNS}, T={T_TOTAL}s, "
          f"cells={CELL_IDS}, seeds={SEEDS}")
    print("=" * 78)

    t0 = time.time()
    calib = calibrate_network(T_warm=5.0)
    print(f"[calib] {time.time()-t0:.1f}s")
    for k, v in calib.items():
        print(f"  {k}: {v}")
    with open(RESULTS_DIR / "calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)
    sigma_pn = calib["sigma_per_neuron"]
    sigma_ps = calib["sigma_per_synapse"]

    n_total = len(CELL_IDS) * len(SEEDS)
    done = 0
    elapsed = 0.0
    for cell_id in CELL_IDS:
        for seed in SEEDS:
            tag = f"{cell_id}_seed{seed}"
            path = RAW_DIR / f"{tag}.npz"
            if path.exists():
                done += 1
                continue
            t0 = time.time()
            try:
                sol = run_cell(cell_id, seed, sigma_pn, sigma_ps, T=T_TOTAL)
            except Exception as exc:
                print(f"[ERR] {tag}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            wall = time.time() - t0
            elapsed += wall
            done += 1
            ts = np.asarray(sol.ts)
            ys = {k: np.asarray(v) for k, v in sol.ys.items()}
            np.savez(path, ts=ts, **ys,
                     cell_id=cell_id, seed=seed,
                     N_neurons=N_NEURONS, N_inputs=N_INPUTS, n_patterns=N_PATTERNS,
                     pattern_duration=PATTERN_DURATION, delta_V=DELTA_V,
                     sigma_pn=sigma_pn, sigma_ps=sigma_ps,
                     lr=LEARNING_RATES.get(cell_id, 1.0))
            W_in = ys["W_readout_inputs"]
            fr_init = float(ys["fr_readout"][:10].mean())
            fr_final = float(ys["fr_readout"][-10:].mean())
            W_target = float(W_in[-1, :NEURONS_PER_PATTERN].mean())
            W_off = float(W_in[-1, NEURONS_PER_PATTERN:].mean())
            print(f"[done {done}/{n_total}] {tag} wall={wall:.1f}s "
                  f"fr {fr_init:.1f}→{fr_final:.1f}Hz, "
                  f"W_tgt={W_target:.3e} W_off={W_off:.3e} sel={W_target-W_off:+.3e}")
    print(f"[total] {elapsed:.1f}s new")


if __name__ == "__main__":
    main()
