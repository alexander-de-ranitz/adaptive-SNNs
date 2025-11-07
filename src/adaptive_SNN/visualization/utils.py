import jax.numpy as jnp

from adaptive_SNN.models import (
    Agent,
    AgentEnvSystem,
    AgentState,
    LIFNetwork,
    LIFState,
    NoisyNetwork,
    NoisyNetworkState,
    SystemState,
)

# =====================================================
# Helper functions to extract models and states
# =====================================================


def get_LIF_state(state):
    if isinstance(state, LIFState):
        return state
    elif isinstance(state, NoisyNetworkState):
        return state.network_state
    elif isinstance(state, AgentState):
        return state.noisy_network.network_state
    elif isinstance(state, SystemState):
        return state.agent_state.noisy_network.network_state
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")


def get_LIF_model(model):
    if isinstance(model, LIFNetwork):
        return model
    elif isinstance(model, NoisyNetwork):
        return model.base_network
    elif isinstance(model, Agent):
        return model.noisy_network.base_network
    elif isinstance(model, AgentEnvSystem):
        return model.agent.noisy_network.base_network
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def get_noisy_network_state(state):
    if isinstance(state, NoisyNetworkState):
        return state
    elif isinstance(state, AgentState):
        return state.noisy_network
    elif isinstance(state, SystemState):
        return state.agent_state.noisy_network
    else:
        raise ValueError(f"Unsupported state type: {type(state)}")


def get_noisy_network_model(model):
    if isinstance(model, NoisyNetwork):
        return model
    elif isinstance(model, Agent):
        return model.noisy_network
    elif isinstance(model, AgentEnvSystem):
        return model.agent.noisy_network
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


# ======================================================================
# Plotting functions- reusable functions to be used to create plots
# ======================================================================


def _plot_membrane_potential(ax, t, state, model, neurons_to_plot=None):
    lif_state = get_LIF_state(state)
    V = lif_state.V
    S = lif_state.S

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    # Plot membrane potentials
    for i in neurons_to_plot:
        spike_times = t[S[:, i] > 0]
        ax.vlines(
            spike_times,
            V[:, i][S[:, i] > 0] * 1e3,
            -40,
        )
        ax.plot(t, V[:, i] * 1e3, label=f"Neuron {i + 1} V")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title("Neuron Membrane Potential")


def _plot_spikes_raster(ax, t, state, model, neurons_to_plot=None):
    # TODO: use neurons_to_plot
    lif_network = get_LIF_model(model)
    lif_state = get_LIF_state(state)

    N_neurons = lif_network.N_neurons
    N_inputs = lif_network.N_inputs
    exc_mask = lif_network.excitatory_mask
    spikes = lif_state.S

    dt = t[1] - t[0]
    spike_times_per_neuron = [
        jnp.nonzero(spikes[:, i])[0] * dt for i in range(spikes.shape[1])
    ][::-1]
    ax.set_yticks(range(len(spike_times_per_neuron)))
    ax.eventplot(spike_times_per_neuron, colors="black", linelengths=0.8)
    ax.set_ylabel("Neuron")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spike Raster Plot")

    if N_inputs > 0:
        # Shade background to distinguish input vs. main neurons
        # This assumes that all exc/inh inputs are grouped toghether at the end of the neuron list
        N_exc_input = jnp.sum(exc_mask[N_neurons:])
        N_inh_input = N_inputs - N_exc_input
        ax.axhspan(-0.5, N_inh_input - 0.5, facecolor="#E8BFB5", alpha=0.3)
        ax.axhspan(
            N_inh_input - 0.5,
            N_inh_input + N_exc_input - 0.5,
            facecolor="#B5D6E8",
            alpha=0.3,
        )


def _plot_conductances(ax, t, state, model, neurons_to_plot=None, split_noise=False):
    base_network = get_LIF_model(model)
    network_state = get_LIF_state(state)

    N_neurons = base_network.N_neurons
    W = jnp.where(~jnp.isfinite(network_state.W), 0.0, network_state.W)
    G = network_state.G
    exc_mask = base_network.excitatory_mask

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(N_neurons)

    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)

    if split_noise:
        if isinstance(model, LIFNetwork):
            raise ValueError(
                "Cannot split noise conductances for LIFNetwork model; no noise present."
            )

        noisy_network_state = get_noisy_network_state(state)
        noise = noisy_network_state.noise_state

        ax.plot(
            t,
            weighed_G_excitatory[:, neurons_to_plot],
            label="Synaptic E Conductance",
            color="g",
        )
        ax.plot(
            t,
            weighed_G_inhibitory[:, neurons_to_plot],
            label="Synaptic I Conductance",
            color="r",
        )
        ax.plot(
            t,
            noise[:, neurons_to_plot],
            label="Noise E Conductance",
            color="g",
            linestyle=":",
        )
    else:
        if isinstance(model, LIFNetwork):
            total_G_excitatory = weighed_G_excitatory
            total_G_inhibitory = weighed_G_inhibitory
        else:
            noisy_network_state = get_noisy_network_state(state)
            noise = noisy_network_state.noise_state
            total_G_excitatory = weighed_G_excitatory + noise
            total_G_inhibitory = weighed_G_inhibitory

        ax.plot(
            t,
            total_G_excitatory[:, neurons_to_plot],
            label="Total E Conductance",
            color="g",
        )
        ax.plot(
            t,
            total_G_inhibitory[:, neurons_to_plot],
            label="Total I Conductance",
            color="r",
        )
    ax.set_ylabel("Total Conductance (S)")
    ax.set_title("Total Conductances")


def _plot_voltage_distribution(ax, t, state, model, neurons_to_plot=None):
    V = get_LIF_state(state).V

    if neurons_to_plot is None:
        neurons_to_plot = jnp.arange(V.shape[1])

    for i in neurons_to_plot:
        ax.hist(
            V[:, i] * 1e3, bins=50, alpha=0.5, label=f"Neuron {i + 1}", density=True
        )
    ax.set_xlabel("Membrane Potential (mV)")
    ax.set_ylabel("Density")
    ax.set_title("Membrane Potential Distribution")
    ax.set_xlim(jnp.min(V) * 1e3 - 5, jnp.max(V) * 1e3 + 5)
