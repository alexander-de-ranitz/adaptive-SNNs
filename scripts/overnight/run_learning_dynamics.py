"""Phase C — Learning dynamics with η > 0.

Closed-loop η > 0 training on the Single Synapse Task using `LearningAgent`
to carry a filtered scalar RPE state through the integrator. Tracks the
learnable weight W_learnable(t) plus the RPE trace, the eligibility (for
B cells), and the noisy/noiseless spike statistics.

We use 3 ΔV points and 5 seeds per cell, T = 30 s (sufficient to see W
trajectory diverge from initial value 0).
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
from adaptive_SNN.solver import simulate_noisy_SNN  # noqa: E402

MASTER_SEED = 42
N_NEURONS = 2
N_INPUTS = 3
INPUT_RATES = jnp.array([5000.0, 1250.0, 10.0])
INPUT_TYPES = jnp.array([True, False, True])
NOISE_SCALE_HYPERPARAM = 2.0
BALANCE_TARGET = 0.0
T_TOTAL = 30.0
DT = 1e-4
N_SAVE = 600
TAU_RPE = 0.1
RPE_NOISE_RATE = 1.0
RPE_NOISE_STD = 1.0

DELTA_V_SWEEP = jnp.array([2.0 ** -11, 2.0 ** -9, 2.0 ** -7])
SEEDS = [42, 43, 44, 45, 46]
CELL_IDS = ["A-PN", "A-PS", "A0-PN", "A0-PS", "B-PN", "B-PS"]

LEARNING_RATES = {
    "A-PN": 50.0,  "A0-PN": 50.0,
    "A-PS": 50.0,  "A0-PS": 50.0,
    "B-PN": 500.0, "B-PS": 500.0,
}

RESULTS_DIR = REPO_ROOT / "results" / "overnight-learning-dynamics"
RAW_DIR = RESULTS_DIR / "raw"


def make_initial_weights():
    return jnp.tile(jnp.array([jnp.nan, jnp.nan, 1.1, 11.0, 0.0]), (N_NEURONS, 1))


def _input_spike_fn(spike_key):
    rates = INPUT_RATES

    def fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint(t / DT), dtype=jnp.int64)
        spikes_1d = jr.poisson(jr.fold_in(spike_key, step_idx), rates * DT, shape=(1, N_INPUTS))
        return jnp.tile(spikes_1d, (N_NEURONS, 1)).astype(jnp.float64)

    return fn


def _make_base_network(model_cls, delta_V, key, use_gating=True):
    kwargs = dict(dt=DT, N_neurons=N_NEURONS, N_inputs=N_INPUTS,
                  initial_weight_matrix=make_initial_weights(),
                  input_types=INPUT_TYPES, fully_connected_input=True, key=key)
    if model_cls is GatedLIFNetwork:
        net = model_cls(**kwargs)
        object.__setattr__(net, "delta_V", float(delta_V))
        return net
    if model_cls is OUAMeanReversionLIFNetwork:
        return model_cls(**kwargs, delta_V=float(delta_V), use_gating=use_gating, update_clip=1.0)
    if model_cls is PerSynapseGatedEligibilityLIFNetwork:
        return model_cls(**kwargs, delta_V=float(delta_V))
    raise ValueError(model_cls)


def _make_noisy(cell_id, delta_V, sigma_pn, sigma_ps, key):
    net_key, _ = jr.split(key)
    if cell_id in ("A-PN", "A0-PN"):
        use_gating = (cell_id == "A-PN")
        net = _make_base_network(OUAMeanReversionLIFNetwork, delta_V, net_key, use_gating=use_gating)
        return NoisyNetwork(net, NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn), min_noise_std=5e-9)
    if cell_id in ("A-PS", "A0-PS"):
        use_gating = (cell_id == "A-PS")
        net = _make_base_network(OUAMeanReversionLIFNetwork, delta_V, net_key, use_gating=use_gating)
        oup = PerSynapseOUP(N_neurons=N_NEURONS, N_inputs=N_INPUTS, excitatory_mask=net.excitatory_mask, tau=net.tau_E, noise_std=sigma_ps)
        return PerSynapseNoisyNetwork(net, oup, min_noise_std=5e-9)
    if cell_id == "B-PN":
        net = _make_base_network(GatedLIFNetwork, delta_V, net_key)
        return NoisyNetwork(net, NeuralNoiseOUP(tau=net.tau_E, dim=N_NEURONS, noise_std=sigma_pn), min_noise_std=5e-9)
    if cell_id == "B-PS":
        net = _make_base_network(PerSynapseGatedEligibilityLIFNetwork, delta_V, net_key)
        oup = PerSynapseOUP(N_neurons=N_NEURONS, N_inputs=N_INPUTS, excitatory_mask=net.excitatory_mask, tau=net.tau_E, noise_std=sigma_ps)
        return PerSynapseNoisyNetwork(net, oup, min_noise_std=5e-9)
    raise ValueError(cell_id)


def _save_fn(cell_id):
    def fn(t, y, args):
        inner = y.network_state.network_state
        common = {
            "V": inner.V,
            "S": inner.S,
            "G_learn": jnp.atleast_1d(inner.G[0, 4]),
            "W_learn": jnp.atleast_1d(inner.W[0, 4]),
            "rpe": jnp.atleast_1d(y.rpe),
        }
        if cell_id.startswith("B-"):
            common["elig_learn"] = jnp.atleast_1d(inner.features.eligibility[0, 4])
        return common
    return fn


def _build_args(spike_key, rpe_noise_key, sigma_ps, lr):
    return {
        "get_input_spikes": _input_spike_fn(spike_key),
        "get_learning_rate": lambda t, x, a: jnp.array(lr),
        "get_desired_balance": lambda t, x, a: jnp.array(BALANCE_TARGET),
        "noise_scale_hyperparam": NOISE_SCALE_HYPERPARAM,
        "use_noise": jnp.array([True, False]),
        "per_synapse_noise_std_target": sigma_ps,
        "rpe_noise_key": rpe_noise_key,
    }


def run_cell(cell_id, delta_V, seed, sigma_pn, sigma_ps, T=T_TOTAL, lr=None):
    base_key = jr.PRNGKey(seed)
    cell_key, spike_key, brown_key, rpe_key = jr.split(base_key, 4)
    noisy = _make_noisy(cell_id, delta_V, sigma_pn, sigma_ps, cell_key)
    agent = LearningAgent(noisy, tau_RPE=TAU_RPE, rpe_noise_rate=RPE_NOISE_RATE, rpe_noise_std=RPE_NOISE_STD)
    args = _build_args(spike_key, rpe_key, sigma_ps, lr if lr is not None else LEARNING_RATES.get(cell_id, 1.0))
    save_ts = jnp.linspace(0.0, T, N_SAVE)
    saveat = dfx.SaveAt(subs=dfx.SubSaveAt(ts=save_ts, fn=_save_fn(cell_id)))
    return simulate_noisy_SNN(agent, dfx.EulerHeun(), 0.0, T, DT, agent.initial, save_at=saveat, args=args, key=brown_key)


def main():
    from run_paired_comparison_v2 import calibrate
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Phase C — η>0 learning dynamics (closed-loop via LearningAgent)")
    print(f"T={T_TOTAL}s, dt={DT}s, cells={CELL_IDS}")
    print("=" * 78)

    calib = calibrate()
    sigma_pn = calib["sigma_per_neuron"]
    sigma_ps = calib["sigma_per_synapse"]
    with open(RESULTS_DIR / "calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)
    print(f"[calib] sigma_pn={sigma_pn:.2e}, sigma_ps={sigma_ps:.3f}")

    n_total = len(CELL_IDS) * len(DELTA_V_SWEEP) * len(SEEDS)
    elapsed = 0.0
    done = 0
    for cell_id in CELL_IDS:
        for dV in DELTA_V_SWEEP:
            dV_pow = int(round(np.log2(float(dV))))
            for seed in SEEDS:
                tag = f"{cell_id}_dV2^{dV_pow}_seed{seed}"
                path = RAW_DIR / f"{tag}.npz"
                if path.exists():
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
                np.savez(path, ts=ts, **ys, cell_id=cell_id, delta_V=float(dV), seed=seed,
                         sigma_pn=sigma_pn, sigma_ps=sigma_ps,
                         lr=LEARNING_RATES.get(cell_id, 1.0))
                W_final = float(ys["W_learn"][-1])
                fr_n = float(ys["S"][:, 0].sum() / T_TOTAL)
                print(f"[done {done}/{n_total}] {tag} wall={wall:.1f}s W_final={W_final:.3e} fr_n={fr_n:.2f}Hz")
    print(f"[total] {elapsed:.1f}s new")


if __name__ == "__main__":
    main()
