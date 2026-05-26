import jax

jax.config.update("jax_enable_x64", True)

import argparse
import time

from diffrax import SaveAt
from jax import numpy as jnp
from jax import random as jr

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks.eligibility_LIF import EligibilityLIFNetwork
from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_SNN.simulation_configs.biofeedback_experiment import create_config
from adaptive_SNN.utils.runner import run_simulation


def main():
    print(f"Jax is using device: {jax.devices()[0]}")

    parser = argparse.ArgumentParser(
        description="Run network simulation with specified parameters"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/biofeedback_experiment.npz",
        help="Path to save the simulation results",
    )
    parser.add_argument(
        "--model", type=str, default="gated", help="Type of model to simulate"
    )
    parser.add_argument(
        "--key_seed", type=int, default=0, help="Seed for random number generation"
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, help="Learning rate for synaptic updates"
    )

    args = parser.parse_args()
    if args.model == "gated":
        model_cls = GatedLIFNetwork
    elif (args.model == "eligibility") or (args.model == "default"):
        model_cls = EligibilityLIFNetwork
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    cfg = create_config(
        model_cls=model_cls, N_neurons=1000, key=jr.PRNGKey(args.key_seed)
    )
    cfg.connection_prob_E = 0.1
    cfg.t1 = 600
    cfg.initial_rec_weight = 1.0
    cfg.initial_input_weight = 1.0
    cfg.save_file = args.output_file
    cfg.balance = 0.5
    # Set learning rates for recurrent and input weights. Only recurrent weights are plastic
    cfg.lr = args.lr * jnp.hstack(
        (
            jnp.ones((cfg.N_neurons, cfg.N_neurons)),
            jnp.zeros((cfg.N_neurons, cfg.N_inputs)),
        )
    )
    cfg.warmup_time = 5.0

    def save(t, x: SystemState, args):
        return (
            x.environment_state.astype(jnp.float32),
            x.agent_state.network_state.network_state.W[0].astype(jnp.float32),
            jnp.nanmean(x.agent_state.network_state.network_state.W),
            jnp.nanstd(x.agent_state.network_state.network_state.W),
        )

    cfg.save_at = SaveAt(ts=jnp.linspace(cfg.t0, cfg.t1, 5000), fn=save)

    N_E_in = 150
    cfg.args.update({"N_simulated_E_inputs": N_E_in})

    input_rate = jnp.array([N_E_in * 10])  # High frequency background input

    spike_key = jr.fold_in(cfg.key, 1337)

    def input_spike_fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint((t - cfg.t0) / cfg.dt), dtype=jnp.int64)
        return jr.poisson(
            jr.fold_in(spike_key, step_idx),
            input_rate * cfg.dt,
            shape=(cfg.N_neurons, cfg.N_inputs),
        )

    cfg.input_spike_fn = input_spike_fn

    start = time.time()
    sol, model = run_simulation(cfg, save_results=True)
    end = time.time()
    print(f"Simulation took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
