import jax

jax.config.update("jax_enable_x64", True)

import argparse
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import SaveAt

from adaptive_snn.models.networks import LIFNetwork, LIFState
from adaptive_snn.simulation_configs.single_synapse_config import (
    create_single_synapse_config,
)
from adaptive_snn.utils.runner import run_batched_simulation, run_simulation
from adaptive_snn.utils.save_helper import save_part_of_state

# Neuron 0 is an unperturbed reference; neuron i + 1 receives pulses of AMPLITUDES[i]
AMPLITUDES = np.array([5e-9, 10e-9, 20e-9, 0.0, -5e-9, -10e-9, -20e-9])
N_NEURONS = len(AMPLITUDES) + 1
EPOCH_STEPS = 155  # 15.5 ms per epoch
PULSE_OFFSET = (
    5  # The pulse acts during step 5 of each epoch, 0.5 ms after the epoch starts
)
N_SETTLE_EPOCHS = 7  # Discarded (~0.1 s), to forget the shared warmed-up starting state


class NoisePulseNetwork(LIFNetwork):
    """LIF network of identical, unconnected neurons with identical input, where neurons 1..N receive pulses.

    At the start of every epoch, all neurons are reset to the state of the reference neuron 0. During step
    PULSE_OFFSET of the epoch, neuron i receives a conductance pulse of args["pulse_amps"][i]. The pulse is
    applied after this step's input spikes are added to G, so the clipping of the total E conductance at
    zero is resolved here and the stored perturbation is the applied one.
    """

    def compute_desired_noise_std(self, t, state: LIFState, args):
        return jnp.zeros((self.N_neurons,))

    def pre_step_update(self, t, state: LIFState, args, input_spikes):
        state = super().pre_step_update(t, state, args, input_spikes)
        step = jnp.rint(t / self.dt).astype(int) % EPOCH_STEPS  # Assumes t0 = 0

        # The spike buffer is not reset: it only feeds recurrent connections, which do not exist here
        state = eqx.tree_at(
            lambda s: (
                s.V,
                s.S,
                s.G,
                s.time_since_last_spike,
                s.filtered_spike_trains,
                s.mean_V,
                s.mean_E_conductance,
                s.mean_I_conductance,
                s.var_E_conductance,
                s.charge_in,
                s.charge_out,
            ),
            state,
            replace_fn=lambda x: jnp.where(step == 0, x[:1], x),
        )

        W_clean = jnp.where(jnp.isnan(state.W), 0.0, state.W)
        synaptic_E_conductances = (W_clean * state.G) @ self.excitatory_mask
        requested = jnp.where(step == PULSE_OFFSET, args["pulse_amps"], 0.0)
        applied = (
            jnp.maximum(synaptic_E_conductances + requested, 0.0)
            - synaptic_E_conductances
        )
        return eqx.tree_at(lambda s: s.perturbations, state, applied)

    def drift(self, t, state: LIFState, args, RPE=jnp.zeros(1)):
        d_state = super().drift(t, state, args, RPE)
        # Keep the perturbations fixed within a step, so the saved value is the applied one
        return eqx.tree_at(
            lambda s: s.perturbations, d_state, jnp.zeros_like(state.perturbations)
        )


def make_config(key, t1, pulse_amps, balance, save_at):
    # The config draws the input spike trains from the input key in args, so that every simulation in a batch
    # receives its own input. All neurons of a simulation receive identical inputs.
    config = create_single_synapse_config(
        key,
        network_cls=NoisePulseNetwork,
        N_neurons=N_NEURONS,
        background_weight=1.0,
        learned_weight=1.0,
        noise_floor=0.0,
        balance=balance,
        t1=t1,
    )
    config.save_at = save_at
    config.additional_args["pulse_amps"] = jnp.asarray(pulse_amps)
    return config


def compute_perturbation_response(
    output_file: Path, n_sims=500, n_epochs=70, warmup_time=100.0, seed=9999
):
    """Estimate dy(t)/dg and dV(t)/dg from the difference between pulsed neurons and the reference neuron.

    Runs n_sims simulations in parallel, each with its own input and n_epochs epochs of one pulse per
    perturbed neuron. Saved per amplitude: the sums over epochs of the spike and voltage differences at each
    step of the epoch, and the summed applied amplitude. The voltage differences are also summed over only
    the epochs in which neither the reference nor the perturbed neuron spiked, to show the contribution of
    spike resets. Saved per epoch: the reference voltage at the start of the pulse step and the total spike
    difference after the pulse, to test the voltage dependence of m(t).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pulse_amps = np.concatenate([[0.0], AMPLITUDES])
    key = jr.PRNGKey(seed)
    warmup_key, sims_key = jr.split(key)

    # Warm up the network to reach a steady state (I weights balanced, filters settled)
    warmup_config = make_config(
        warmup_key,
        t1=warmup_time,
        pulse_amps=np.zeros(N_NEURONS),
        balance=0.01,
        save_at=SaveAt(t1=True, fn=lambda t, x, args: save_part_of_state(x, S=True)),
    )
    sol, _ = run_simulation(
        warmup_config, save_results=False, load_if_exists=False, return_final_state=True
    )
    _, warmed_up_state = sol.ys

    def save_fn(t, x, args):
        s = x.agent_state.network_state
        return (
            (s.S[1:] - s.S[0]).astype(jnp.int8),  # -1, 0 or +1
            (s.V[1:] - s.V[0]).astype(jnp.float32),
            s.V[0].astype(jnp.float32),
            s.perturbations[1:].astype(jnp.float32),
            s.S[0].astype(jnp.int8),
        )

    dt = warmup_config.dt
    n_steps = (N_SETTLE_EPOCHS + n_epochs) * EPOCH_STEPS
    configs = []
    for sim_key in jr.split(sims_key, n_sims):
        configs.append(
            make_config(
                sim_key,
                t1=n_steps * dt,
                pulse_amps=pulse_amps,
                balance=jnp.nan,  # Freeze I weights so the neurons only differ by the pulses
                save_at=SaveAt(steps=True, fn=save_fn),
            )
        )
    sols, _ = run_batched_simulation(
        configs,
        save_results=False,
        load_if_exists=False,
        y0s=[warmed_up_state] * n_sims,
    )

    # Saved index k + 1 is the state after step k. Reshape to (epochs, step in epoch, ...) for all simulations.
    y_diff, V_diff, V_ref, perturbations, S_ref = (
        np.asarray(leaf)[:, 1 : n_steps + 1]
        .reshape(n_sims, -1, EPOCH_STEPS, *leaf.shape[2:])[:, N_SETTLE_EPOCHS:]
        .reshape(n_sims * n_epochs, EPOCH_STEPS, *leaf.shape[2:])
        for leaf in sols.ys
    )
    applied_amps = perturbations[:, PULSE_OFFSET]
    spike_free = ~S_ref.any(axis=1)[:, None] & ~y_diff.any(axis=1)  # (epoch, amplitude)
    # Classify each pulse by its net effect on the spike count, and bin it by the step at which the spike
    # trains first differ: the moment the pulse first changed the spiking
    net_count_change = y_diff[:, PULSE_OFFSET:].sum(
        axis=1, dtype=np.int64
    )  # (epoch, amplitude)
    first_divergence = np.argmax(y_diff != 0, axis=1)
    count_change_by_time = np.stack(
        [
            np.bincount(first_divergence[:, a], net_count_change[:, a], EPOCH_STEPS)
            for a in range(len(AMPLITUDES))
        ]
    )

    np.savez(
        output_file.with_suffix(".npz"),
        y_diff_sum=y_diff.sum(axis=0).T,  # (amplitude, step in epoch)
        V_diff_sum=V_diff.sum(
            axis=0, dtype=np.float64
        ).T,  # float32 sums are inaccurate
        V_diff_sum_spike_free=np.einsum(
            "esa,ea->as", V_diff, spike_free, dtype=np.float64
        ),
        n_spike_free=spike_free.sum(axis=0),
        count_change_by_time=count_change_by_time,  # (amplitude, step of first divergence)
        applied_sum=applied_amps.sum(axis=0, dtype=np.float64),
        total_y_diff=net_count_change.T.astype(np.int8),  # (amplitude, epoch)
        V_at_pulse=V_ref[
            :, PULSE_OFFSET - 1
        ],  # Reference voltage at the start of the pulse step
        As=AMPLITUDES,
        n_trials=n_sims * n_epochs,
        dt=dt,
        pulse_offset=PULSE_OFFSET,
    )
    for A, applied in zip(AMPLITUDES, applied_amps.T):
        print(f"A={A * 1e9:+.0f} nS: mean applied {applied.mean() * 1e9:+.2f} nS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/assumption_verification/perturbation_response_summary",
        help="Output path without extension; the summary is saved as <output_file>.npz",
    )
    parser.add_argument("--key_seed", type=int, default=9999, help="Random key seed")
    parser.add_argument(
        "--n_sims", type=int, default=500, help="Simulations run in parallel"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=70, help="Pulses per neuron per simulation"
    )
    parser.add_argument("--warmup_time", type=float, default=100.0)
    args = parser.parse_args()
    compute_perturbation_response(
        Path(args.output_file),
        n_sims=args.n_sims,
        n_epochs=args.n_epochs,
        warmup_time=args.warmup_time,
        seed=args.key_seed,
    )
