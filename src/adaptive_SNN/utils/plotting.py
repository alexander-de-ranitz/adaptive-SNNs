import jax.numpy as jnp
from matplotlib import pyplot as plt


def plot_results(sol, spikes, model, t0, t1, dt0):
    # Get results
    t = sol.ts
    (V, G), noise_E, noise_I = sol.ys
    print("No conductance" if jnp.all(G == 0.0) else "Conductance present")
    G_inhibitory = (
        jnp.sum(G * jnp.invert(model.network.excitatory_mask[None, None, :]), axis=-1)
        + noise_I
    )
    G_excitatory = (
        jnp.sum(G * model.network.excitatory_mask[None, None, :], axis=-1) + noise_E
    )
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot membrane potentials
    for i in range(model.N_neurons):
        spike_times = t[spikes[:, i] > 0]
        ax1.vlines(
            spike_times,
            V[:, i][spikes[:, i] > 0] * 1e3,
            V[:, i][spikes[:, i] > 0] * 1e3 + 10,
        )
        ax1.plot(t, V[:, i] * 1e3, label=f"Neuron {i + 1} V")
    ax1.set_ylabel("Membrane Potential (mV)")

    print(G_excitatory.shape, G_inhibitory.shape)
    # Plot total conductance of neuron 0
    ax2.plot(t, G_excitatory, label="Total E Conductance", color="g")
    ax2.plot(t, G_inhibitory, label="Total I Conductance", color="r")
    ax2.set_ylabel("Total Conductance (S)")

    # Plot spikes as raster plot
    spike_times_per_neuron = [
        jnp.nonzero(spikes[:, i])[0] * dt0 for i in range(spikes.shape[1])
    ][::-1]
    ax3.eventplot(spike_times_per_neuron, colors="black", linelengths=0.8)
    ax3.set_yticks(range(len(spike_times_per_neuron)))
    ax3.set_ylabel("Neuron")

    # Set x-axis limits and ticks for all subplots
    xticks = jnp.linspace(t0, t1, 6)  # 6 evenly spaced ticks
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(t0, t1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
    plt.show()
