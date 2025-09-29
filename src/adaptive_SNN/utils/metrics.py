from jax import numpy as jnp
from jaxtyping import Array


def compute_CV_ISI(spikes) -> Array:
    """Compute the coefficient of variation (CV) of inter-spike intervals (ISI) for each neuron.

    Args:
        spikes (jnp.ndarray): A binary array of shape (num_time_steps, num_neurons) indicating spike occurrences.

    Returns:
        jnp.ndarray: An array of shape (num_neurons,) containing the CV of ISI for each neuron.
    """
    num_time_steps, num_neurons = spikes.shape
    CV_ISI = jnp.zeros(num_neurons)

    for neuron_idx in range(num_neurons):
        spike_times = jnp.nonzero(spikes[:, neuron_idx] == 1)[0]
        if len(spike_times) < 2:
            CV_ISI = CV_ISI.at[neuron_idx].set(
                jnp.nan
            )  # Not enough spikes to compute ISI
            continue

        ISIs = jnp.diff(spike_times)
        mean_ISI = jnp.mean(ISIs)
        std_ISI = jnp.std(ISIs)

        if mean_ISI > 0:
            CV_ISI = CV_ISI.at[neuron_idx].set(std_ISI / mean_ISI)
        else:
            CV_ISI = CV_ISI.at[neuron_idx].set(jnp.nan)  # Avoid division by zero

    return CV_ISI
