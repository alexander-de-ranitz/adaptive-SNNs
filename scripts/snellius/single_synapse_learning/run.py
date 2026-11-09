import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import argparse
import time

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import SaveAt
from jaxtyping import Array

from adaptive_snn.models import SystemState
from adaptive_snn.models.networks import GatedLIFNetwork
from adaptive_snn.models.networks.gated_LIF import GatedEligibility, GatedLIFState
from adaptive_snn.simulation_configs.single_synapse_config import (
    create_single_synapse_config,
)
from adaptive_snn.solver.solver import solve_ODE_batched, stack_pytrees
from adaptive_snn.utils.runner import setup_simulation

# Many independent trials of the task are simulated in parallel. Each trial consists of a perturbed
# neuron and its unperturbed twin with identical inputs, weights and initial state. Both are rewarded with an
# exponentially decaying trace per spike, and the trial's RPE is the difference between the two rewards. The RPE thus
# equals the causal effect of the perturbations on the reward: emitted and prevented spikes produce RPEs of equal size,
# and reward that would also have occurred without perturbations cancels. Every trial acts as one neuron of a network
# for every other trial: the external RPE noise of a trial is the sum of the RPEs of all other trials. Since weights
# are frozen and the RPE does not feed back into the dynamics, the trials can be simulated independently and combined
# per chunk.

# Gating steepnesses to evaluate, each with and without mean subtraction (centering). Four more traces are appended:
#  - spike gate (binary gate: only the step after the neuron spiked), with and without mean subtraction
#  - no gating, with and without mean subtraction (the latter is the EligibilityLIFNetwork rule)
# Centering ("_centered") subtracts the running mean of the gated perturbation per neuron. This leaves the covariance
# between the gated perturbation and the synaptic trace G in the eligibility, which causes drift under a non-zero mean
# RPE. Therefore, every rule is also evaluated with per-synapse centering ("_syncentered"), which subtracts the running
# mean of the full eligibility input (gated perturbation * G) per synapse instead. For the ungated rule, this mean is
# expected to be zero (the perturbation is independent of G); it is included as a control.
# delta_V = 2^-15 is not included, since the gate is not properly resolved at dt = 1e-4.
DELTA_VS = jnp.power(2.0, -jnp.arange(5, 14, 2, dtype=jnp.float64))
TRACE_LABELS = (
    [f"gated_{dv}_centered" for dv in DELTA_VS.tolist()]
    + [f"gated_{dv}" for dv in DELTA_VS.tolist()]
    + ["spike_centered", "spike", "ungated_centered", "ungated"]
    + [f"gated_{dv}_syncentered" for dv in DELTA_VS.tolist()]
    + ["spike_syncentered", "ungated_syncentered"]
)
N_TRACES = len(TRACE_LABELS)
SUBTRACT_MEAN = jnp.array([label.endswith("_centered") for label in TRACE_LABELS])
SUBTRACT_SYNAPSE_MEAN = jnp.array(
    [label.endswith("_syncentered") for label in TRACE_LABELS]
)
TRIAL_WINDOW_LENGTH = 10.0  # Weight updates are summed over windows of this length


class MultiTraceEligibility(GatedEligibility):
    mean_eligibility_input: Array  # Running mean of the eligibility input per synapse, (N_TRACES, N_neurons, N_syn)


class MultiDeltaVGatedLIFNetwork(GatedLIFNetwork):
    """GatedLIFNetwork that tracks one eligibility trace per delta_V in DELTA_VS, plus spike-gated and ungated traces.

    Only valid with frozen weights: the traces never feed back into the dynamics,
    so all of them see the exact same trajectory, as if run separately with the same seed.
    """

    def init_features(self) -> MultiTraceEligibility:
        assert not self.learn_I_weights, "Not implemented for learn_I_weights=True"
        eligibility_shape = (N_TRACES, self.N_neurons, self.N_neurons + self.N_inputs)
        return MultiTraceEligibility(
            eligibility=jnp.zeros(eligibility_shape),
            mean_scaled_perturbations=jnp.zeros((N_TRACES, self.N_neurons)),
            mean_eligibility_input=jnp.zeros(eligibility_shape),
        )

    def noise_shape_features(self) -> MultiTraceEligibility:
        return MultiTraceEligibility(
            eligibility=None,
            mean_scaled_perturbations=None,
            mean_eligibility_input=None,
        )

    def compute_feature_drift(
        self, t, state: GatedLIFState, args
    ) -> MultiTraceEligibility:
        noise_std = self.compute_desired_noise_std(t, state, args)
        normalized_perturbations = jnp.where(
            noise_std != 0.0, state.perturbations / noise_std, 0.0
        )  # (N_neurons,)

        # Spike gate: state.S holds the spikes of the previous step (the drift is evaluated before this
        # step's threshold crossing is known).
        spike_gate = state.S[: self.N_neurons]
        voltage_gate = self.gating_function(state.V[None, :], DELTA_VS[:, None])
        gating = jnp.concatenate(
            [
                voltage_gate,  # Centered
                voltage_gate,  # Uncentered
                spike_gate[None, :],
                spike_gate[None, :],
                jnp.ones((2, self.N_neurons)),
                voltage_gate,  # Per-synapse centered
                spike_gate[None, :],  # Per-synapse centered
                jnp.ones((1, self.N_neurons)),  # Per-synapse centered
            ]
        )  # (N_TRACES, N_neurons)

        scaled_perturbations = normalized_perturbations[None, :] * gating
        d_mean_scaled_perturbations = (
            scaled_perturbations - state.features.mean_scaled_perturbations
        ) / self.tau_mean_scaled_noise

        scaled_perturbations = jnp.where(
            SUBTRACT_MEAN[:, None],
            scaled_perturbations - state.features.mean_scaled_perturbations,
            scaled_perturbations,
        )

        eligibility_input = (
            scaled_perturbations[:, :, None]
            / self.synaptic_increment
            * self.excitatory_mask[None, None, :]
            * state.G[None]
        )  # (N_TRACES, N_neurons, N_syn)
        d_mean_eligibility_input = (
            eligibility_input - state.features.mean_eligibility_input
        ) / self.tau_mean_scaled_noise
        eligibility_input = jnp.where(
            SUBTRACT_SYNAPSE_MEAN[:, None, None],
            eligibility_input - state.features.mean_eligibility_input,
            eligibility_input,
        )

        d_eligibility = (
            eligibility_input - state.features.eligibility / self.tau_eligibility
        )

        return MultiTraceEligibility(
            eligibility=d_eligibility,
            mean_scaled_perturbations=d_mean_scaled_perturbations,
            mean_eligibility_input=d_mean_eligibility_input,
        )

    def compute_weight_updates(
        self, t, state: GatedLIFState, args, RPE: Array
    ) -> Array:
        # Weights are frozen; the eligibility has an extra trace dimension so the default rule does not apply
        return jnp.zeros_like(state.W)


def make_config(key, noise_level, initial_weight):
    # Perturbed neuron and its unperturbed twin (balancing disabled, such that the twin cannot drift away). The
    # background E synapse and the synapse to be learned both have weight initial_weight. Every trial receives its own
    # input spike trains, drawn from the input key in its args.
    config = create_single_synapse_config(
        key,
        network_cls=MultiDeltaVGatedLIFNetwork,
        twin=True,
        background_weight=initial_weight,
        learned_weight=initial_weight,
        noise_floor=noise_level,
    )
    config.base_network_kwargs["tau_eligibility"] = 0.01
    return config


def save_fn(t, x: SystemState, args):
    # Eligibility of the to-be-learned synapse (last input) of the perturbed neuron for every trace, and the RPE
    return (
        x.agent_state.network_state.features.eligibility[:, 0, -1],
        x.agent_state.RPE[0],
    )


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
        "--initial_weight",
        type=float,
        default=1.0,
        help="Initial weight for the background E synapse",
    )
    parser.add_argument("--key_seed", type=int, default=0, help="Random key seed")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1000,
        help="Number of trials simulated in parallel",
    )
    parser.add_argument(
        "--t1", type=float, default=150.0, help="End time of the simulation"
    )
    parser.add_argument(
        "--start_save",
        type=float,
        default=50.0,
        help="Time from which to accumulate weight updates",
    )
    parser.add_argument(
        "--chunk_length",
        type=float,
        default=0.25,
        help="Simulation is run in chunks of this length to reduce memory usage.",
    )

    args = parser.parse_args()
    assert TRIAL_WINDOW_LENGTH % args.chunk_length == 0, (
        "TRIAL_WINDOW_LENGTH must be divisible by chunk_length"
    )
    assert args.start_save % TRIAL_WINDOW_LENGTH == 0, (
        "start_save must be divisible by TRIAL_WINDOW_LENGTH"
    )

    B = args.n_trials
    configs = [
        make_config(key, args.noise_level, args.initial_weight)
        for key in jr.split(jr.PRNGKey(args.key_seed), B)
    ]
    assert jnp.all(configs[0].lr == 0.0), "Multi Delta_V requires frozen weights"

    # Set up and stack once, then call solve_ODE_batched per chunk, carrying over the (stacked) final states.
    # Re-stacking the per-trial models, args and states every chunk makes host memory grow steadily.
    models, sim_args, sim_keys = zip(*(setup_simulation(config) for config in configs))
    y0s = stack_pytrees([model.initial for model in models])
    models, sim_args, sim_keys = (
        stack_pytrees(list(models)),
        stack_pytrees(list(sim_args)),
        jnp.stack(sim_keys),
    )
    solver = dfx.EulerHeun()
    dt = configs[0].dt
    steps_per_chunk = int(round(args.chunk_length / dt))
    n_chunks = int(round(args.t1 / args.chunk_length))
    first_saved_chunk = int(round(args.start_save / args.chunk_length))
    chunks_per_window = int(round(TRIAL_WINDOW_LENGTH / args.chunk_length))

    # Per-chunk sums, shape (n_saved_chunks, B, N_TRACES)
    task_updates = []
    noise_updates = []
    eligibility_sums = []  # Summed eligibility, to subtract the mean RPE from the external noise post hoc
    rpe_sum = (
        0.0  # Summed RPE over all trials and saved steps, to estimate the mean RPE
    )

    for chunk in range(n_chunks):
        t0 = chunk * args.chunk_length
        save_at = SaveAt(ts=t0 + dt * jnp.arange(1, steps_per_chunk + 1), fn=save_fn)
        sols = solve_ODE_batched(
            models=models,
            solver=solver,
            # Pass times as arrays rather than floats, such that they are not static and the jit cache is reused
            t0=jnp.asarray(t0, dtype=jnp.float64),
            t1=jnp.asarray(t0 + args.chunk_length, dtype=jnp.float64),
            dt0=dt,
            y0s=y0s,
            save_at=save_at,
            return_final_state=True,
            args=sim_args,
            keys=sim_keys,
        )
        (eligibility, rpe), y0s = (
            sols.ys
        )  # (B, steps, N_TRACES), (B, steps), stacked final states

        if chunk < first_saved_chunk:
            continue

        # External noise of each trial: the sum of the RPEs of all other trials. Its mean ((B - 1) times the mean RPE) is
        # kept, such that the effect of centering can be shown. It can be removed post hoc using the eligibility sums.
        external_noise = rpe.sum(0, keepdims=True) - rpe
        # einsum sums over steps without materializing the (B, steps, N_TRACES) products, saving memory
        task_updates.append(jnp.einsum("bsk,bs->bk", eligibility, rpe))
        noise_updates.append(jnp.einsum("bsk,bs->bk", eligibility, external_noise))
        eligibility_sums.append(eligibility.sum(1))
        rpe_sum += rpe.sum()

    # Sum the chunks into windows of TRIAL_WINDOW_LENGTH: shape (n_windows, B, N_TRACES)
    to_windows = (
        lambda x: jnp.stack(x).reshape(-1, chunks_per_window, B, N_TRACES).sum(1)
    )

    jnp.savez(
        args.output_file,
        delta_Vs=DELTA_VS,
        trace_labels=np.array(TRACE_LABELS),
        task_updates=to_windows(task_updates),
        noise_updates=to_windows(noise_updates),
        eligibility_sums=to_windows(eligibility_sums),
        mean_rpe=rpe_sum / (len(task_updates) * B * steps_per_chunk),
    )

    end = time.time()
    print(f"Simulation completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
