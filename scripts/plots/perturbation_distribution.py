import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from diffrax import SaveAt

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.networks import NoisyNetwork, NoisyNetworkState
from adaptive_SNN.simulation_configs.single_neuron_simulation import (
    create_single_neuron_config_extra_synapse,
)
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state


class ExternalNoiseStd(NoisyNetwork):
    def compute_desired_noise_std(self, t, state: NoisyNetworkState, args):
        return args.get("external_noise_std")(t, state, args)

    def update(self, t, state, args):
        state = super().update(t, state, args)
        noise_state = state.noise_state
        external_noise_std = args.get("external_noise_std")(t, state, args)
        new_noise_state = jnp.where(
            external_noise_std > 0, noise_state, jnp.zeros_like(noise_state)
        )
        return NoisyNetworkState(state.network_state, new_noise_state)


def plot_perturbation_distribution_over_time():
    config = create_single_neuron_config_extra_synapse(N_neurons=2)

    config.N_neurons = 2
    t0 = 0.0
    t1 = 2.04
    t_start_saving = 2.0
    t_onset = 2.01
    t_offset = 2.02
    external_noise_std = 10e-9

    config.noisy_network_cls = ExternalNoiseStd
    config.t0 = t0
    config.t1 = t1

    config.save_at = SaveAt(
        ts=jnp.linspace(t_start_saving, t1, int((t1 - t_start_saving) / config.dt)),
        fn=lambda t, x, args: save_part_of_state(
            x,
            V=True,
            S=True,
        ),
    )

    config.args["external_noise_std"] = lambda t, x, args: jnp.where(
        t > t_onset,
        jnp.where(
            t < t_offset,
            jnp.zeros((config.N_neurons,)).at[0].set(external_noise_std),
            jnp.zeros((config.N_neurons,)),
        ),
        jnp.zeros((config.N_neurons,)),
    )
    config.args["tau_RPE"] = 0.1
    config.args["use_noise"] = jnp.array([True, False])

    n_iterations = 100
    key = jr.PRNGKey(2001)
    V_diff = None
    for i in range(n_iterations):
        print(f"Running simulation {i + 1}/{n_iterations}...", end="\r")
        key = jr.fold_in(key, i)
        config.key = key

        sol, model = run_simulation(config, save_results=False)
        state: SystemState = sol.ys

        if jnp.sum(state.agent_state.noisy_network.network_state.S[:, 0]) > 0:
            print(
                f"Run {i}: Spikes detected during perturbation window, skipping this run."
            )
            continue  # Skip this run if there are any spikes, as we want to analyze the voltage distribution without the influence of spiking activity

        V_diff = (
            state.agent_state.noisy_network.network_state.V[:, 0]
            - state.agent_state.noisy_network.network_state.V[:, 1]
            if V_diff is None
            else jnp.vstack(
                (
                    V_diff,
                    state.agent_state.noisy_network.network_state.V[:, 0]
                    - state.agent_state.noisy_network.network_state.V[:, 1],
                )
            )
        )

    plt.figure(figsize=(3.5, 2))
    plt.plot(
        sol.ts,
        V_diff.T * 1e3,
        color="darkgreen",
        alpha=0.3,
        label="Voltage Difference Samples",
    )

    y0, y1 = plt.ylim()
    y_extreme = max(abs(y0), abs(y1))
    y0, y1 = -y_extreme, y_extreme
    plt.vlines(
        [t_onset, t_offset],
        ymin=y0,
        ymax=y1,
        color="lightgray",
        linestyle="--",
        label="Perturbation Window",
    )
    plt.fill_betweenx([y0, y1], x1=t_onset, x2=t_offset, color="lightgray", alpha=0.5)
    plt.ylim(y0, y1)
    plt.xticks(
        jnp.linspace(t_start_saving, t1, 5),
        labels=[
            f"{jnp.round((t - t_start_saving) * 1000).astype(int)}"
            for t in jnp.linspace(t_start_saving, t1, 5)
        ],
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage Difference (mV)")
    plt.show()


if __name__ == "__main__":
    plot_perturbation_distribution_over_time()
