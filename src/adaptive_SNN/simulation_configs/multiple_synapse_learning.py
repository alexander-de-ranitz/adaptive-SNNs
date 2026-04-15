import jax

jax.config.update("jax_enable_x64", True)
from diffrax import SaveAt
from jax import numpy as jnp
from jax import random as jr

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks.coupled import (
    CoupledWeightGatedLIFNetwork,
)
from adaptive_SNN.models.noise.poisson_jump import PoissonJumpProcess
from adaptive_SNN.models.reward import StudentRewardModel
from adaptive_SNN.utils.config import SimulationConfig
from adaptive_SNN.utils.save_helper import save_part_of_state


def create_multiple_synapse_learning_config(
    lr: float,
    noise_level: float,
    reward_noise_rate: float,
    key=jr.PRNGKey(0),
    N_groups: int = 1,
) -> SimulationConfig:
    # Define network parameters
    t0 = 0.0
    t1 = 1000
    dt = 1e-4
    N_students = N_groups
    N_neurons = 3 * N_groups
    N_background_input_per_group = 2
    N_background_input = N_groups * N_background_input_per_group
    N_signal_input_per_group = 4
    N_signal_input = N_groups * N_signal_input_per_group
    N_inputs = N_background_input + N_signal_input

    TEACHER_INDICES = jnp.arange(0, N_neurons, 3)
    STUDENT_INDICES = TEACHER_INDICES + 1
    REFERENCE_INDICES = TEACHER_INDICES + 2

    # Define input signal parameters
    key, spike_key, reward_noise_key = jr.split(key, 3)
    INPUT_TYPES = jnp.array(
        [1] * N_signal_input + ([1, 0] * N_groups)
    )  # Group-specific signal inputs (E) and group-specific background inputs (E/I)
    BACKGROUND_RATES_PER_GROUP = jnp.array(
        [5000, 1250]
    )  # Constant background rates per group

    # Define target and initial weights
    BACKGROUND_WEIGHTS = jnp.array([1.1, 11])

    def _group_signal_slice(group_idx: int):
        start = group_idx * N_signal_input_per_group
        end = (group_idx + 1) * N_signal_input_per_group
        return start, end

    def _group_background_slice(group_idx: int):
        start = N_signal_input + group_idx * N_background_input_per_group
        end = start + N_background_input_per_group
        return start, end

    def _build_initial_weight_matrix():
        initial_weight_matrix = jnp.full((N_neurons, N_neurons + N_inputs), -jnp.inf)

        group_ids = jnp.arange(N_groups)[:, None]
        feature_ids = jnp.arange(N_signal_input_per_group)[None, :]
        teacher_signal_weights = 15.0 + 10.0 * jnp.sin(
            0.7 * group_ids + 0.9 * feature_ids
        )
        teacher_signal_weights = jnp.clip(teacher_signal_weights, a_min=0.0)

        for group_idx in range(N_groups):
            teacher_idx = int(TEACHER_INDICES[group_idx])
            student_idx = int(STUDENT_INDICES[group_idx])
            reference_idx = int(REFERENCE_INDICES[group_idx])
            signal_start, signal_end = _group_signal_slice(group_idx)
            background_start, background_end = _group_background_slice(group_idx)

            teacher_input_weights = jnp.full((N_inputs,), -jnp.inf)
            teacher_input_weights = teacher_input_weights.at[
                signal_start:signal_end
            ].set(teacher_signal_weights[group_idx])
            teacher_input_weights = teacher_input_weights.at[
                background_start:background_end
            ].set(BACKGROUND_WEIGHTS)

            student_input_weights = jnp.full((N_inputs,), -jnp.inf)
            student_input_weights = student_input_weights.at[
                signal_start:signal_end
            ].set(0.0)
            student_input_weights = student_input_weights.at[
                background_start:background_end
            ].set(BACKGROUND_WEIGHTS)

            initial_weight_matrix = initial_weight_matrix.at[
                teacher_idx, N_neurons:
            ].set(teacher_input_weights)
            initial_weight_matrix = initial_weight_matrix.at[
                student_idx, N_neurons:
            ].set(student_input_weights)
            initial_weight_matrix = initial_weight_matrix.at[
                reference_idx, N_neurons:
            ].set(student_input_weights)

        return initial_weight_matrix

    INITIAL_WEIGHT_MATRIX = _build_initial_weight_matrix()

    LEARNABLE_WEIGHT_MASK = jnp.zeros_like(INITIAL_WEIGHT_MATRIX)
    for group_idx in range(N_groups):
        student_idx = int(STUDENT_INDICES[group_idx])
        signal_start, signal_end = _group_signal_slice(group_idx)
        LEARNABLE_WEIGHT_MASK = LEARNABLE_WEIGHT_MASK.at[
            student_idx, N_neurons + signal_start : N_neurons + signal_end
        ].set(1.0)

    def generate_rates(t, state: SystemState, args):
        group_ids = jnp.arange(N_groups)
        rate_1 = (
            1 + jnp.sin(2 * jnp.pi * (0.2 + 0.02 * group_ids) * t + 0.4 * group_ids)
        ) * 15
        rate_2 = (
            jnp.cos(2 * jnp.pi * (0.8 + 0.03 * group_ids) * t + 0.3 * group_ids) * 12.0
            + 3.0
        )
        rate_3 = (
            jnp.sin(
                2 * jnp.pi * (0.1 + 0.01 * group_ids) * t
                + (0.8 + 0.1 * group_ids) * jnp.pi
            )
            * jnp.cos(2 * jnp.pi * 0.2 * t)
            * 15.0
            + 5.0
        )
        rate_4 = (
            jnp.cos(
                2 * jnp.pi * (0.25 + 0.02 * group_ids) * t
                + (0.5 + 0.1 * group_ids) * jnp.pi
            )
            * 10.0
            + 5.0
        )
        signal_rates = jnp.stack([rate_1, rate_2, rate_3, rate_4], axis=1)
        signal_rates = 10 + jnp.clip(
            signal_rates, a_min=0.0
        )  # Ensure rates are non-negative
        background_rates = jnp.tile(BACKGROUND_RATES_PER_GROUP, (N_groups, 1))
        rates = jnp.concatenate([signal_rates, background_rates], axis=1).reshape(-1)
        return rates

    def generate_spikes(t, state: SystemState, args):
        base_key = jr.fold_in(spike_key, jnp.rint(t / dt))
        spikes = jnp.zeros((N_neurons, N_inputs))
        all_group_rates = generate_rates(t, state, args).reshape(N_groups, -1)

        for group_idx in range(N_groups):
            teacher_idx = TEACHER_INDICES[group_idx]
            student_idx = STUDENT_INDICES[group_idx]
            reference_idx = REFERENCE_INDICES[group_idx]
            signal_start, signal_end = _group_signal_slice(group_idx)
            background_start, background_end = _group_background_slice(group_idx)

            group_rates = all_group_rates[group_idx]
            group_key = jr.fold_in(base_key, group_idx)
            group_spikes = jr.poisson(group_key, lam=group_rates * dt)

            spikes = spikes.at[teacher_idx, signal_start:signal_end].set(
                group_spikes[:N_signal_input_per_group]
            )
            spikes = spikes.at[student_idx, signal_start:signal_end].set(
                group_spikes[:N_signal_input_per_group]
            )
            spikes = spikes.at[reference_idx, signal_start:signal_end].set(
                group_spikes[:N_signal_input_per_group]
            )

            spikes = spikes.at[teacher_idx, background_start:background_end].set(
                group_spikes[N_signal_input_per_group:]
            )
            spikes = spikes.at[student_idx, background_start:background_end].set(
                group_spikes[N_signal_input_per_group:]
            )
            spikes = spikes.at[reference_idx, background_start:background_end].set(
                group_spikes[N_signal_input_per_group:]
            )

        return spikes

    save_at = SaveAt(
        ts=jnp.linspace(t0, t1, 5000),
        fn=lambda t, state, args: save_part_of_state(
            state,
            environment_state=True,
            reward_noise=True,
            W=True,
            V=True,
            G=True,
            reward_signal=True,
            predicted_reward=True,
            eligibility=True,
            noise_state=True,
            mean_E_conductance=True,
            var_E_conductance=True,
        ),
    )

    cfg = SimulationConfig(
        base_network_cls=CoupledWeightGatedLIFNetwork,
        base_network_kwargs={
            "weight_coupling_indices": (REFERENCE_INDICES, STUDENT_INDICES)
        },
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        connection_prob=0.0,
        fully_connected_input=True,
        balance=0.0,
        mean_synaptic_delay=0.0,
        input_types=INPUT_TYPES,
        t0=t0,
        t1=t1,
        initial_weight_matrix=INITIAL_WEIGHT_MATRIX,
        lr=lr * LEARNABLE_WEIGHT_MASK,
        noise_level=jnp.zeros(N_neurons).at[STUDENT_INDICES].set(noise_level),
        min_noise_std=0.0,
        warmup_time=100,
        args={
            "use_noise": jnp.array([False] * N_neurons).at[STUDENT_INDICES].set(True)
        },
        key=0,
        save_at=save_at,
        network_output_fn=lambda t, agent_state, args: jnp.squeeze(
            agent_state.noisy_network.network_state.S
        ),
        input_spike_fn=generate_spikes,
        reward_fn=lambda t, x, args: jnp.array([0.0]),
        reward_model=StudentRewardModel,
        reward_kwargs={"N_neurons": N_neurons, "N_students": N_students},
        reward_noise_model=PoissonJumpProcess,
        reward_noise_kwargs={
            "jump_rate": reward_noise_rate,
            "jump_mean": 0.0,
            "jump_std": 1.0,
            "dim": 1,
            "dt": dt,
            "tau": 0.05,
            "key": reward_noise_key,
        },
        environment_kwargs={"dim": N_neurons, "rate": 10},
    )

    return cfg


if __name__ == "__main__":
    cfg = create_multiple_synapse_learning_config(
        lr=0.01, noise_level=0.1, reward_noise_rate=1.0, N_groups=3
    )
# def plot_teacher_simulation_results(sol, model):
#     t0, t1 = sol.ts[0], sol.ts[-1]
#     dt = 1e-4

#     def generate_spikes(t, state : SystemState , args):
#         base_key = jr.fold_in(spike_key, jnp.rint(t / dt))
#         spikes = jnp.zeros((N_neurons, N_inputs))
#         all_group_rates = generate_rates(t, state, args).reshape(N_groups, -1)

#         for group_idx in range(N_groups):
#             teacher_idx = int(TEACHER_INDICES[group_idx])
#             student_idx = int(STUDENT_INDICES[group_idx])
#             reference_idx = int(REFERENCE_INDICES[group_idx])
#             signal_start, signal_end = _group_signal_slice(group_idx)
#             background_start, background_end = _group_background_slice(group_idx)

#             group_rates = all_group_rates[group_idx]
#             group_key = jr.fold_in(base_key, group_idx)
#             group_spikes = jr.poisson(group_key, lam=group_rates * dt)

#             spikes = spikes.at[teacher_idx, signal_start:signal_end].set(group_spikes[:N_signal_input_per_group])
#             spikes = spikes.at[student_idx, signal_start:signal_end].set(group_spikes[:N_signal_input_per_group])
#             spikes = spikes.at[reference_idx, signal_start:signal_end].set(group_spikes[:N_signal_input_per_group])

#             spikes = spikes.at[teacher_idx, background_start:background_end].set(group_spikes[N_signal_input_per_group:])
#             spikes = spikes.at[student_idx, background_start:background_end].set(group_spikes[N_signal_input_per_group:])
#             spikes = spikes.at[reference_idx, background_start:background_end].set(group_spikes[N_signal_input_per_group:])

#         return spikes

#     state : SystemState = sol.ys
#     env_state = state.environment_state
#     reward_signal = state.reward_signal

#     plot_environment_state = True
#     plot_reward = True
#     plot_raster = False
#     plot_rates = True
#     plot_weights = True
#     plot_V = False
#     plot_G = False
#     plot_incoming_spikes=False
#     plot_noise_state = False
#     plot_eligibility = False
#     fig, axs = plt.subplots(plot_environment_state + plot_reward + plot_incoming_spikes + plot_weights + plot_rates + plot_raster + plot_V + plot_G + plot_noise_state + plot_eligibility, figsize=(3.5, 2), sharex=True)

#     curr_ax = 0
#     if plot_environment_state:
#         for group_idx in range(N_groups):
#             teacher_idx = int(TEACHER_INDICES[group_idx])
#             student_idx = int(STUDENT_INDICES[group_idx])
#             reference_idx = int(REFERENCE_INDICES[group_idx])
#             axs[curr_ax].plot(sol.ts, env_state[:, teacher_idx], label=f"Teacher {group_idx+1}")
#             axs[curr_ax].plot(sol.ts, env_state[:, student_idx], label=f"Student {group_idx+1}", alpha=0.7)
#             axs[curr_ax].plot(sol.ts, env_state[:, reference_idx], linestyle='--', alpha=0.7, label=f"Reference {group_idx+1}")
#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Signal")
#         axs[curr_ax].legend()
#         curr_ax += 1


#     if plot_reward:
#         RPE = state.agent_state.RPE
#         axs[curr_ax].plot(sol.ts, RPE, label="Reward Prediction Error")
#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Reward")
#         axs[curr_ax].legend()
#         curr_ax += 1


#     if plot_weights:
#         weights = state.agent_state.noisy_network.network_state.W
#         colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#         linestyles = ['-', '--', '-.', ':']
#         for group_idx in range(N_groups):
#             teacher_idx = int(TEACHER_INDICES[group_idx])
#             student_idx = int(STUDENT_INDICES[group_idx])
#             signal_start, signal_end = _group_signal_slice(group_idx)
#             for local_signal_idx, signal_idx in enumerate(range(signal_start, signal_end)):
#                 c = colors[local_signal_idx % len(colors)]
#                 axs[curr_ax].plot(
#                     sol.ts,
#                     weights[:, student_idx, N_neurons + signal_idx],
#                     color=c,
#                     alpha=0.8,
#                     linestyle=linestyles[group_idx % len(linestyles)],
#                     label=f"Student {group_idx+1} Weight {local_signal_idx+1}" if group_idx == 0 else None,
#                 )
#                 axs[curr_ax].plot(
#                     sol.ts,
#                     weights[:, teacher_idx, N_neurons + signal_idx],
#                     linestyle=':',
#                     color=c,
#                     alpha=0.5,
#                 )
#         axs[curr_ax].set_ylabel("Weights")
#         curr_ax += 1

#     if plot_rates:
#         t_rates = jnp.linspace(t0, t1, 1000)
#         true_rates = jnp.array([generate_rates(t, state, {}) for t in t_rates])
#         rates_by_group = true_rates.reshape(t_rates.shape[0], N_groups, -1)

#         group_base_colors = [
#             jnp.array([0.1216, 0.4667, 0.7059]),  # blue
#             jnp.array([0.1725, 0.6275, 0.1725]),  # green
#             jnp.array([0.8392, 0.1529, 0.1569]),  # red
#         ]

#         def _mix_with_white(color, amount):
#             return color * (1.0 - amount) + jnp.ones_like(color) * amount

#         for group_idx in range(N_groups):
#             group_signal_rates = rates_by_group[:, group_idx, :N_signal_input_per_group]

#             base_color = group_base_colors[group_idx % len(group_base_colors)]
#             shade_amounts = jnp.linspace(0.05, 0.55, N_signal_input_per_group)

#             for local_idx in range(N_signal_input_per_group):
#                 line_color = _mix_with_white(base_color, shade_amounts[local_idx])
#                 line_color = tuple(float(c) for c in line_color)
#                 axs[curr_ax].plot(
#                     t_rates,
#                     group_signal_rates[:, local_idx],
#                     color=line_color,
#                     label=f"Group {group_idx + 1} Signal {local_idx + 1}",
#                 )

#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Rate (Hz)")
#         axs[curr_ax].legend(ncol=2, fontsize=7)
#         curr_ax += 1

#     if plot_V:
#         V = state.agent_state.noisy_network.network_state.V
#         axs[curr_ax].plot(sol.ts, V, alpha=0.5)
#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Voltage")

#         # Add gating function on second y-axis
#         gating = CoupledGatedLIFNetwork(dt=1e-4, N_neurons=1).gating_function(state.agent_state.noisy_network.network_state.V[:, int(STUDENT_INDICES[0])])
#         axs[curr_ax].twinx().plot(sol.ts, gating, color='gray', alpha=1)
#         curr_ax += 1

#     if plot_G:
#         _plot_conductances(axs[curr_ax], sol, model, split_noise=True, neurons_to_plot=[int(STUDENT_INDICES[0])])
#         curr_ax += 1

#     if plot_incoming_spikes:
#         signal_start, signal_end = _group_signal_slice(0)
#         input_G = state.agent_state.noisy_network.network_state.G[:, int(STUDENT_INDICES[0]), N_neurons + signal_start:N_neurons + signal_end] + 1e-9 * jnp.arange(N_signal_input_per_group)  # Conductance from input synapses to the student neuron
#         axs[curr_ax].plot(sol.ts, input_G, alpha=0.5)
#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Input Conductance")
#         curr_ax += 1

#     if plot_noise_state:
#         noise_state = state.agent_state.noisy_network.noise_state[:, int(STUDENT_INDICES[0])]
#         axs[curr_ax].plot(sol.ts, noise_state, label="Noise State", alpha=0.5)
#         axs[curr_ax].axhline(0, color='gray', linestyle='--', alpha=0.5)
#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Noise State")
#         axs[curr_ax].legend()
#         curr_ax += 1

#     if plot_eligibility:
#         eligibility = state.agent_state.noisy_network.network_state.features.eligibility
#         signal_start, signal_end = _group_signal_slice(0)
#         for signal_idx in range(signal_start, signal_end):
#             axs[curr_ax].plot(sol.ts, eligibility[:, int(STUDENT_INDICES[0]), N_neurons + signal_idx])
#         axs[curr_ax].set_xlabel("Time")
#         axs[curr_ax].set_ylabel("Eligibility")
#         curr_ax += 1

#     if plot_raster:
#         # Build spike times per neuron (reverse order to show I neurons at bottom)
#         spike_ts = jnp.arange(t0, t1, dt)
#         all_spikes = jnp.stack([generate_spikes(t, state, {}) for t in spike_ts])
#         spikes = jnp.zeros((spike_ts.shape[0], N_signal_input))
#         for group_idx in range(N_groups):
#             teacher_idx = int(TEACHER_INDICES[group_idx])
#             signal_start, signal_end = _group_signal_slice(group_idx)
#             spikes = spikes.at[:, signal_start:signal_end].set(
#                 all_spikes[:, teacher_idx, signal_start:signal_end]
#             )
#         spike_times_per_neuron = [
#             spike_ts[jnp.nonzero(spikes[:, i])[0]] for i in range(spikes.shape[1])
#         ][::-1]
#         if len(spike_times_per_neuron) < 10:
#             axs[curr_ax].set_yticks(range(len(spike_times_per_neuron)))
#         else:
#             axs[curr_ax].set_yticks([])
#         axs[curr_ax].eventplot(
#             spike_times_per_neuron,
#             colors="black",
#             linelengths=0.8,
#             linewidths=0.4,
#         )
#         axs[curr_ax].set_ylabel("Input (all groups)")
#         axs[curr_ax].set_xlabel("Time (s)")

#     for ax in axs:
#         ax.label_outer()
#     plt.show()
