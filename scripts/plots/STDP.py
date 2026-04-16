import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

# Allow running this file directly while importing from the repository modules.
_THIS_FILE = Path(__file__).resolve()
_SCRIPTS_DIR = _THIS_FILE.parents[1]
_REPO_ROOT = _THIS_FILE.parents[2]
for _path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from diffrax import SaveAt

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks.eligibility_LIF import (
    ElibilityState,
    Eligibility,
    EligibilityLIFNetwork,
)
from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_SNN.simulation_configs.single_neuron_simulation import (
    create_single_neuron_config_extra_synapse,
)
from adaptive_SNN.utils.runner import run_simulation

MAX_TIME_DIFF = 0.1
WINDOW_BUFFER = 10e-3


class NoDecayGatedLIFNetwork(GatedLIFNetwork):
    def compute_feature_drift(self, t, state: ElibilityState, args) -> Eligibility:
        noise_std = args.get("noise_std", 0.0)
        noise_conductance = args.get("excitatory_noise", jnp.zeros((self.N_neurons,)))

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set relative noise strength to zero
        relative_noise_strength = jnp.where(
            noise_std != 0.0, noise_conductance / noise_std, 0.0
        )

        # Map the relative noise strength to each excitatory synapse
        noise_per_synapse = jnp.outer(relative_noise_strength, self.excitatory_mask)

        delta_V = args.get("delta_V", self.delta_V)

        synaptic_traces = state.G
        d_eligibility = (
            noise_per_synapse
            * synaptic_traces
            / self.synaptic_increment
            * self.gating_function(state.V, delta_V)[:, None]
        )
        return Eligibility(eligibility=d_eligibility)


class NoDecayEligibilityLIFNetwork(EligibilityLIFNetwork):
    def compute_feature_drift(self, t, state, args):
        noise_std = args.get("noise_std", 0.0)
        noise_conductance = args.get("excitatory_noise", jnp.zeros((self.N_neurons,)))

        # To decouple the absolute noise level from the synaptic weight changes, we normalize the noise by the desired noise std
        # In case the noise std is zero (no noise), avoid division by zero and set relative noise strength to zero
        relative_noise_strength = jnp.where(
            noise_std != 0.0, noise_conductance / noise_std, 0.0
        )

        # Map the relative noise strength to each excitatory synapse
        noise_per_synapse = jnp.outer(relative_noise_strength, self.excitatory_mask)

        synaptic_traces = state.G
        d_eligibility = noise_per_synapse * synaptic_traces / self.synaptic_increment
        return Eligibility(eligibility=d_eligibility)


def compute_eligibility_changes_around_spikes(sol) -> list[tuple[float, float]]:
    result = sol.ys
    pre_synaptic_spikes, post_synaptic_spikes, eligibility = result

    # Plot how the eligibility changes over time and how it relates to pre- and post-synaptic spikes
    post_synaptic_spike_times = sol.ts[post_synaptic_spikes == 1]
    pre_synaptic_spike_times = sol.ts[pre_synaptic_spikes == 1]

    spike_time_diff = (
        pre_synaptic_spike_times[:, None] - post_synaptic_spike_times[None, :]
    )

    mask = (spike_time_diff >= -MAX_TIME_DIFF) & (spike_time_diff <= MAX_TIME_DIFF)
    coincidences = jnp.sum(mask, axis=0)

    # Get the post-synaptic spike times that have at least one pre-synaptic spike within the window
    post_spikes_w_pre = post_synaptic_spike_times[coincidences > 0]

    # For each of these post-synaptic spikes, get the time differences to the nearest pre-synaptic spike
    paired_pre_indices = jnp.argmin(
        jnp.abs(spike_time_diff[:, coincidences > 0]), axis=0
    )
    paired_pre_spike_times = pre_synaptic_spike_times[paired_pre_indices]
    time_diffs = paired_pre_spike_times - post_spikes_w_pre

    e_changes = []

    skipped_spikes = 0
    for i, post_spike_time in enumerate(post_spikes_w_pre):
        # if i < len(post_spikes_w_pre)-2 and (post_spikes_w_pre[i + 1] - post_spikes_w_pre[i]) < 0.1:
        #     skipped_spikes += 1
        #     continue
        pre_spike_time = paired_pre_spike_times[i]
        time_diff = time_diffs[i]
        window = (
            min(pre_spike_time, post_spike_time) - WINDOW_BUFFER,
            max(pre_spike_time, post_spike_time) + WINDOW_BUFFER,
        )
        # window = (pre_spike_time, pre_spike_time + MAX_TIME_DIFF * 1.1)
        nearest_time_indices = (
            jnp.argmin(jnp.abs(sol.ts - window[0])),
            jnp.argmin(jnp.abs(sol.ts - window[1])),
        )
        delta_e = (
            eligibility[nearest_time_indices[1]] - eligibility[nearest_time_indices[0]]
        )
        e_changes.append((time_diff, delta_e))
    print(f"Skipped {skipped_spikes}/{len(post_spikes_w_pre)}")
    return e_changes


def run_sim(id, key):
    config = create_single_neuron_config_extra_synapse(N_neurons=1, key=key)
    config.t1 = 30
    config.base_network_cls = NoDecayGatedLIFNetwork
    config.min_noise_std = 0.0
    config.noise_level = 0.3

    def save_fn(t, x: SystemState, args):
        pre_synaptic_spikes = args["get_input_spikes"](t, None, None)[
            :, 2
        ].squeeze()  # Get the spikes from the third input
        post_synaptic_spikes = x.agent_state.noisy_network.network_state.S[0].squeeze()
        eligibility = x.agent_state.noisy_network.network_state.features.eligibility[
            0, 3
        ].squeeze()  # Eligibility for the synapse from input 2 to neuron 0
        return (pre_synaptic_spikes, post_synaptic_spikes, eligibility)

    save_at = SaveAt(steps=True, fn=lambda t, x, args: save_fn(t, x, args))

    config.save_at = save_at

    config.args["use_noise"] = jnp.array([True])
    config.save_file = f"results/STDP_plot/STDP_no_decay_default_simulation_{id}"
    return run_simulation(config, save_results=True)


def plot_STDP():
    key = jr.PRNGKey(2001)
    all_e_changes = []
    for i in range(20):
        start = time.time()
        sol, _ = run_sim(i, jr.fold_in(key, i))
        e_changes = compute_eligibility_changes_around_spikes(sol)
        end = time.time()
        print(
            f"Simulation {i} completed in {end - start:.2f} seconds. Found {len(e_changes)} coincidences.",
            end="\r",
        )
        all_e_changes.extend(e_changes)

    print("\n")
    print(f"Total coincidences across all simulations: {len(all_e_changes)}")
    plt.scatter(*zip(*all_e_changes), c="#027254", alpha=0.7)
    plt.xlabel("Time difference (pre - post) in seconds")
    plt.ylabel("Change in eligibility")
    plt.xlim(-MAX_TIME_DIFF, MAX_TIME_DIFF)
    plt.hlines(0, -MAX_TIME_DIFF, MAX_TIME_DIFF, color="k", linestyle="--")

    ymin, ymax = plt.ylim()
    ymin, ymax = -max(abs(ymin), abs(ymax)), max(abs(ymin), abs(ymax))
    plt.vlines(0, ymin, ymax, color="k", linestyle="--")
    plt.ylim(ymin, ymax)
    plt.show()


if __name__ == "__main__":
    plot_STDP()
