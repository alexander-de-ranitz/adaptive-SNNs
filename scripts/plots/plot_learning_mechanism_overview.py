"""
Visualize the learning mechanism in GatedLIFNetwork.

This script creates a detailed visualization showing how the learning mechanism works
in the GatedLIFNetwork over a short time window. It illustrates:
1. Membrane voltage rising to spike
2. Noise modulation of synaptic conductance
3. Eligibility trace formation
4. Reward and RPE signals
5. Resulting weight changes
"""

import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from adaptive_SNN.models import (
    AgentEnvSystem,
    SystemState,
)
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.networks import Agent, GatedLIFNetwork
from adaptive_SNN.models.reward_prediction import MovingAverageRewardPredictor
from adaptive_SNN.solver import solve_ODE
from adaptive_SNN.utils.save_helper import save_part_of_state

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def main():
    # Simulation parameters
    t0 = 0.0
    t1 = 0.5
    dt = 1e-4  # 0.1 ms

    for i in range(15, 16):
        key = jr.PRNGKey(i)
        spike_times = jnp.array([15, 30]) * 1e-2

        # Network parameters - single neuron, single input
        N_neurons = 1
        N_inputs = 1

        # Set up input parameters
        input_weight = 26.0  # Strong enough to drive spiking

        # Learning parameters
        learning_rate = 2000  # Increased learning rate for more visible weight changes

        # Set up the GatedLIFNetwork
        neuron_model = GatedLIFNetwork(
            N_neurons=N_neurons,
            N_inputs=N_inputs,
            dt=dt,
            fully_connected_input=True,
            initial_input_weight=input_weight,
            initial_rec_weight=0.0,  # No recurrent connections
            fraction_excitatory_input=1.0,
            fraction_excitatory_recurrent=0.8,
            min_noise_std=1e-9,
            key=key,
        )

        agent = Agent(
            neuron_model=neuron_model,
            reward_prediction_model=MovingAverageRewardPredictor(rate=0.0),
        )

        model = AgentEnvSystem(
            agent=agent,
            environment=SpikeRateEnvironment(rate=50),
            agent_output_shape=(1,),
        )

        solver = dfx.EulerHeun()
        init_state = model.initial

        def input_spike_fn(t, x, args):
            # Return 1 if current time is within spike times, 0 otherwise
            return jnp.where(
                jnp.any(jnp.isclose(t, spike_times, atol=dt / 2)),
                jnp.array([[1.0]]),
                jnp.array([[0.0]]),
            )

        def reward_fn(t, x: SystemState, args):
            return jnp.exp(
                -x.agent_state.network_state.time_since_last_spike[0] / 0.05
            ).reshape((1,))

        args = {
            "input_spike_fn": input_spike_fn,
            "get_learning_rate": lambda t, x, args: learning_rate,
            "reward_fn": reward_fn,
            "noise_scale_hyperparam": 1e-6,
            "network_output_fn": lambda t,
            agent_state,
            args,
            env_state: agent_state.network_state.S[0].reshape((1,)),
            "use_noise": jnp.array([True]),
            "feature_fn": lambda t, x, args: jnp.zeros(1),
        }

        # Define what to save
        def save_fn(t, y, args):
            """Save relevant state variables for plotting."""
            return save_part_of_state(
                y,
                environment_state=True,
                W=True,
                G=True,
                S=True,
                RPE=True,
                reward_signal=True,
                time_since_last_spike=True,
                eligibility=True,
                V=True,
                perturbations=True,
            )

        print("Running simulation...")
        sol = solve_ODE(
            model,
            solver,
            t0,
            t1,
            dt,
            init_state,
            save_at=dfx.SaveAt(steps=True, fn=save_fn),
            args=args,
            key=key,
        )

        state: SystemState = sol.ys
        spikes = state.agent_state.network_state.S[:, 0]
        noise = state.agent_state.network_state.perturbations[:, 0]
        G_total = state.agent_state.network_state.G[:, 0].sum(axis=-1) * 1e9
        V = state.agent_state.network_state.V[:, 0] * 1000
        eligibility = state.agent_state.network_state.features.eligibility[:, 0].sum(
            axis=-1
        )
        RPE = state.agent_state.RPE[:, 0]
        W = state.agent_state.network_state.W[:, 0]
        gating = neuron_model.gating_function(
            state.agent_state.network_state.V[:, 0], neuron_model.delta_V
        )
        t = sol.ts.squeeze() * 1000

        # Detect spike times for vertical lines
        spike_indices = jnp.where(spikes > 0)[0]

        if jnp.sum(spikes) != 1:
            print(f"Got {jnp.sum(spikes)} spikes for key {i}. Skipping")
            continue
        if t[spike_indices[0]] < 300:
            print(
                f"Spike occurred at {t[spike_indices[0]]} ms for key {i}, which is too early. Skipping."
            )
            continue
        else:
            print(f"Got 1 spike for key {i}. Proceeding with plotting.")

        # Create figure
        fig = plt.figure(figsize=(3.05, 2.5), dpi=300)
        gs = GridSpec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.35)

        # Color scheme
        color_voltage = "#2E86AB"
        color_conductance = "#A23B72"
        color_noise = "#F18F01"
        color_eligibility = "#C73E1D"
        color_rpe = "#6A994E"
        color_weight = "#BC4B51"

        axes = []

        fontsize = 4
        linewidth = 1

        # Panel 1: Synaptic Conductance
        ax1 = fig.add_subplot(gs[0])
        axes.append(ax1)
        ax1.plot(
            t,
            G_total,
            color=color_conductance,
            linewidth=linewidth,
            label="Total conductance",
        )
        # Annotate to indicate where pre-synaptic spikes occur
        spike_times_ms = spike_times * 1000
        # Calculate midpoint for text placement
        text_x = (spike_times_ms[0] + spike_times_ms[1]) / 2
        text_y = G_total.max() * 1.3
        # Add single text annotation
        ax1.text(text_x, text_y, "Pre-synaptic spikes", fontsize=fontsize, ha="center")
        ax1.axhline(y=0, color="gray", linestyle="dashed", linewidth=0.3, alpha=0.3)
        # Add arrows to both spike locations
        for spike_time_ms in spike_times_ms:
            spike_idx = jnp.where(t >= spike_time_ms)[0][0]
            ax1.annotate(
                "",
                xy=(spike_time_ms - (spike_time_ms - text_x) * 0.1, G_total[spike_idx]),
                xytext=(text_x + (spike_time_ms - text_x) * 0.5, text_y * 0.93),
                arrowprops=dict(
                    facecolor="black",
                    shrink=0.0,
                    headwidth=2,
                    headlength=2,
                    width=0.1,
                    linewidth=0.5,
                ),
            )

        # Panel 2: Noise
        ax2 = fig.add_subplot(gs[1])
        axes.append(ax2)
        ax2.plot(t, noise, color=color_noise, linewidth=linewidth, label="Noise (exc.)")
        ax2.axhline(y=0, color="gray", linestyle="dashed", linewidth=0.3, alpha=0.3)

        # Panel 3: Membrane Voltage
        ax3 = fig.add_subplot(gs[2])
        axes.append(ax3)
        ax3.plot(
            t, V, color=color_voltage, linewidth=linewidth, label="Membrane voltage"
        )
        for ind in spike_indices:
            spike_time = t[ind]
            ax3.vlines(
                spike_time,
                ymin=-60,
                ymax=V[ind - 1] + 10,
                linewidth=linewidth,
                color=color_voltage,
            )

        # Add text and arrow to indicate spiking
        ax3.text(
            spike_time + 20,
            V[ind - 1] + 11,
            "Spike",
            fontsize=fontsize,
            ha="left",
            va="bottom",
        )
        ax3.annotate(
            "",
            xy=(spike_time + 3, V[ind - 1] + 8),
            xytext=(spike_time + 18, V[ind - 1] + 11),
            arrowprops=dict(
                facecolor="black",
                shrink=0,
                headwidth=2,
                headlength=2,
                width=0.01,
                linewidth=0.5,
            ),
        )

        # Panel 4: Gating function (for context)
        ax4 = fig.add_subplot(gs[3])
        axes.append(ax4)
        ax4.plot(
            t, gating, color="#7209B7", linewidth=linewidth, label="Gating function"
        )
        ax4.axhline(y=0, color="gray", linestyle="dashed", linewidth=0.3, alpha=0.3)

        # Panel 5: Eligibility Trace
        ax5 = fig.add_subplot(gs[4])
        axes.append(ax5)
        ax5.plot(
            t,
            eligibility,
            color=color_eligibility,
            linewidth=linewidth,
            label="Eligibility trace",
        )
        ax5.axhline(y=0, color="gray", linestyle="dashed", linewidth=0.3, alpha=0.3)

        # Panel 6: RPE
        ax6 = fig.add_subplot(gs[5])
        axes.append(ax6)
        ax6.plot(t, RPE, color=color_rpe, linewidth=linewidth, label="RPE")
        ax6.axhline(y=0, color="gray", linestyle="dashed", linewidth=0.3, alpha=0.3)

        # Panel 7: weight change
        ax7 = fig.add_subplot(gs[6])
        axes.append(ax7)
        ax7.plot(t, W, color=color_weight, linewidth=linewidth, label="Weight change")
        ax7.set_xlabel(r"Time (ms)")
        ax7.set_ylim(bottom=input_weight * 0.985)  # Zoom in around initial weight
        # ax7.grid(alpha=0.3, linewidth=0.5)

        y_labels = [
            "Syn. Conductance",
            "Noise",
            "Voltage",
            "Gating Function",
            "Eligibility",
            "RPE",
            "Weight",
        ]
        y_offsets = [0.1, 0.01, 0.16, 0.1, 0.1, 0.1, 0.43]
        for i, ax in enumerate(axes):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(t[0], t[-1])
            ax.label_outer()
            ax.set_yticks([])
            ax.text(
                0.01,
                y_offsets[i],
                y_labels[i],
                va="bottom",
                ha="left",
                transform=ax.transAxes,
                fontsize=4,
            )

        fig.align_ylabels(axes)

        output_path = "../figures/learning_mechanism_illustration.pdf"
        fig.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")
        break
        # plt.show()


if __name__ == "__main__":
    main()
