import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import argparse
import time

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt

from adaptive_snn.models import SystemState
from adaptive_snn.models.networks import LIFNetwork
from adaptive_snn.simulation_configs.single_synapse_config import (
    create_single_synapse_config,
)
from adaptive_snn.solver import solve_ODE
from adaptive_snn.utils.metrics import compute_CV_ISI
from adaptive_snn.utils.runner import setup_simulation


def make_save_fn(network: LIFNetwork):
    def save_fn(t, x: SystemState, args):
        state = x.agent_state.network_state
        charge_in = state.charge_in
        charge_out = state.charge_out
        balance = jnp.where(
            (charge_in != 0.0) | (charge_out != 0.0),
            (charge_in + charge_out) / (jnp.abs(charge_in) + jnp.abs(charge_out)),
            jnp.nan,
        )
        # Synaptic conductances as computed in the network's drift (excluding the perturbations)
        weighted_conductances = jnp.where(~jnp.isnan(state.W), state.W, 0.0) * state.G
        E_conductance = weighted_conductances @ network.excitatory_mask
        I_conductance = weighted_conductances @ network.inhibitory_mask
        return (
            state.V,
            state.S.astype(jnp.bool_),
            balance,
            E_conductance,
            I_conductance,
        )

    return save_fn


def main():
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file", type=str, default=None, help="Output file name"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=1e-9,
        help="Min noise std for the synaptic noise",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=jnp.nan,
        help="Balance parameter for the network",
    )
    parser.add_argument(
        "--initial_weight",
        type=float,
        default=0.0,
        help="Initial weight for the background E synapse",
    )
    parser.add_argument("--key_seed", type=int, default=0, help="Random key seed")
    parser.add_argument(
        "--t1", type=float, default=2500.0, help="End time of the simulation"
    )
    parser.add_argument(
        "--start_save",
        type=float,
        default=500.0,
        help="Time from which to compute the firing statistics",
    )
    parser.add_argument(
        "--chunk_length",
        type=float,
        default=100.0,
        help="Simulation is run in chunks of this length to reduce memory usage. Statistics are also stored per chunk.",
    )

    args = parser.parse_args()

    # Background E synapse of weight args.initial_weight (and I synapse 8 times larger), synapse to be learned at 0
    cfg = create_single_synapse_config(
        jr.PRNGKey(args.key_seed),
        network_cls=LIFNetwork,
        background_weight=args.initial_weight,
        learned_weight=0.0,
        noise_floor=args.noise_level,
        balance=args.balance,
    )

    # Set up once and call solve_ODE per chunk, such that model/args (and thus the jit cache) are reused
    model, sim_args, simulation_key = setup_simulation(cfg)
    solver = dfx.EulerHeun()
    save_fn = make_save_fn(model.agent.network)
    steps_per_chunk = int(round(args.chunk_length / cfg.dt))
    n_chunks = int(round(args.t1 / args.chunk_length))
    first_saved_chunk = int(round(args.start_save / args.chunk_length))

    # Spikes are kept for all saved chunks (bool, cheap), so that ISIs spanning chunk boundaries are included.
    # V and balance are reduced per chunk.
    spikes, spike_ts = [], []
    sum_V, sum_balance, n_balance = [], [], []
    sum_E_conductance, sum_I_conductance = [], []
    chunk_firing_rate, chunk_CV_ISI = [], []

    y = model.initial
    for chunk in range(n_chunks):
        t0 = jnp.asarray(chunk * args.chunk_length, dtype=jnp.float64)
        t1 = jnp.asarray((chunk + 1) * args.chunk_length, dtype=jnp.float64)
        # Save after every step (save indices 1..steps_per_chunk)
        save_at = SaveAt(
            ts=t0 + cfg.dt * jnp.arange(1, steps_per_chunk + 1), fn=save_fn
        )
        sol = solve_ODE(
            model,
            solver,
            t0,
            t1,
            cfg.dt,
            y,
            save_at=save_at,
            args=sim_args,
            return_final_state=True,
            key=simulation_key,
        )
        (V, S, balance, E_conductance, I_conductance), y = sol.ys

        if chunk >= first_saved_chunk:
            spikes.append(S)
            spike_ts.append(sol.ts)
            sum_V.append(jnp.sum(V, axis=0))
            sum_balance.append(jnp.nansum(balance, axis=0))
            n_balance.append(jnp.sum(~jnp.isnan(balance), axis=0))
            sum_E_conductance.append(jnp.sum(E_conductance, axis=0))
            sum_I_conductance.append(jnp.sum(I_conductance, axis=0))
            chunk_firing_rate.append(jnp.sum(S, axis=0) / args.chunk_length)
            chunk_CV_ISI.append(compute_CV_ISI(S, sol.ts))

    spikes = jnp.concatenate(spikes)
    spike_ts = jnp.concatenate(spike_ts)
    sum_V, sum_balance, n_balance = (
        jnp.stack(sum_V),
        jnp.stack(sum_balance),
        jnp.stack(n_balance),
    )
    sum_E_conductance = jnp.stack(sum_E_conductance)
    sum_I_conductance = jnp.stack(sum_I_conductance)
    n_steps = spikes.shape[0]

    CV_ISI = compute_CV_ISI(spikes, spike_ts)
    mean_V = jnp.sum(sum_V) / (n_steps * sum_V.shape[1])
    firing_rate = jnp.sum(spikes) / (args.t1 - args.start_save)
    mean_balance = jnp.sum(sum_balance) / jnp.sum(n_balance)
    mean_E_conductance = jnp.sum(sum_E_conductance) / (
        n_steps * sum_E_conductance.shape[1]
    )
    mean_I_conductance = jnp.sum(sum_I_conductance) / (
        n_steps * sum_I_conductance.shape[1]
    )

    jnp.savez(
        args.output_file,
        CV_ISI=CV_ISI,
        mean_V=mean_V,
        firing_rate=firing_rate,
        mean_balance=mean_balance,
        mean_E_conductance=mean_E_conductance,  # Siemens, excluding perturbations
        mean_I_conductance=mean_I_conductance,
        chunk_firing_rate=jnp.stack(chunk_firing_rate),
        chunk_CV_ISI=jnp.stack(chunk_CV_ISI),
        chunk_mean_V=sum_V / steps_per_chunk,
        chunk_mean_balance=sum_balance / n_balance,
        chunk_mean_E_conductance=sum_E_conductance / steps_per_chunk,
        chunk_mean_I_conductance=sum_I_conductance / steps_per_chunk,
    )

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
