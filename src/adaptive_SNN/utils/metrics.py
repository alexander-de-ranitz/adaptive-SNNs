from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.networks.lif import LIFNetwork
from adaptive_SNN.models.networks.noisy_network import NoisyNetwork


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


def compute_conductance_ratio(t, state, model) -> Array:
    """Compute the ratio of weighted inhibitory to excitatory synaptic conductances for each neuron.

    Essentially computes the average over time of the ratio:
        (sum_j W_ij * G_ij * (1 - exc_mask_j) + noise_I) /
        (sum_j W_ij * G_ij * exc_mask_j + noise_E)

    Args:
        t (jnp.ndarray): Time array of shape (num_time_steps,).
        state: The state of the network containing synaptic conductances.
        model: The network model containing parameters and structure.
    Returns:
        jnp.ndarray: An array of shape (num_neurons,) containing the average ratio of inhibitory to excitatory conductances.
    """
    if isinstance(model, LIFNetwork):
        base_network = model
        network_state = state

        noise = jnp.zeros(base_network.N_neurons)
    elif isinstance(model, NoisyNetwork):
        base_network = model.base_network
        network_state = state.network_state

        noise = state.noise_state

    W = network_state.W
    G = network_state.G
    exc_mask = base_network.excitatory_mask

    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1) + noise

    ratio = weighed_G_inhibitory / (
        weighed_G_excitatory + 1e-9
    )  # Avoid division by zero
    avg_ratio = jnp.mean(ratio, axis=0)  # Average over time

    return avg_ratio


def compute_charge_ratio(t, state, model) -> Array:
    """Compute the ratio of total inhibitory to excitatory charge for each neuron.

    Essentially computes the ratio:
        total_inhibitory_charge / total_excitatory_charge
    where total_inhibitory_charge = (sum_j W_ij * G_ij * (1 - exc_mask_j) + noise_I) * (E_I - V_i)
          total_excitatory_charge = (sum_j W_ij * G_ij * exc_mask_j + noise_E) * (E_E - V_i)

    Args:
        t (jnp.ndarray): Time array of shape (num_time_steps,).
        state: The state of the network containing synaptic conductances.
        model: The network model containing parameters and structure.
    Returns:
        jnp.ndarray: An array of shape (num_neurons,) containing the ratio of total inhibitory to excitatory charge.
    """
    if isinstance(model, LIFNetwork):
        base_network: LIFNetwork = model
        network_state = state

        noise = jnp.zeros(base_network.N_neurons)
    elif isinstance(model, NoisyNetwork):
        base_network: LIFNetwork = model.base_network
        network_state = state.network_state

        noise = state.noise_state

    W = network_state.W
    G = network_state.G
    V = network_state.V
    exc_mask = base_network.excitatory_mask

    dt = t[1] - t[0]
    W = jnp.where(jnp.isfinite(W), W, 0.0)
    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1) + noise

    total_inhibitory_charge = (
        jnp.sum(weighed_G_inhibitory * (base_network.reversal_potential_I - V), axis=0)
        * dt
    )
    total_excitatory_charge = (
        jnp.sum(weighed_G_excitatory * (base_network.reversal_potential_E - V), axis=0)
        * dt
    )

    # TODO: this does not account for leak conductance current
    # Might be easier to use the voltage trace directly to compute total charge?

    ratio = jnp.abs(total_inhibitory_charge / (total_excitatory_charge))
    return ratio


def compute_synchrony(spikes: Array, bin_width_ms: float = 2.0, dt: float = 0.1, n_pairs: int = 250) -> float:
    """Compute the average synchrony of neural population activity using correlation coefficients.
    
    Synchrony for a pair of neurons is measured by the correlation coefficient of their spike counts
    in time bins. This function computes the average correlation coefficient for a set of disjoint 
    neuron pairs.
    
    Args:
        spikes (Array): A binary array of shape (num_time_steps, num_neurons) indicating spike occurrences.
        bin_width_ms (float): Width of time bins in milliseconds (default: 2.0 ms).
        dt (float): Time step in milliseconds (default: 0.1 ms).
        n_pairs (int): Number of disjoint neuron pairs to sample (default: 250).
    
    Returns:
        float: The average correlation coefficient across the sampled neuron pairs.
    """
    num_time_steps, num_neurons = spikes.shape
    
    # Calculate the number of time steps per bin
    steps_per_bin = int(bin_width_ms / dt)
    
    # Calculate the number of complete bins
    n_bins = num_time_steps // steps_per_bin
    
    # Truncate spikes to fit complete bins
    truncated_steps = n_bins * steps_per_bin
    spikes_truncated = spikes[:truncated_steps, :]
    
    # Reshape to bins and sum spikes within each bin for all neurons
    # Shape: (n_bins, num_neurons)
    spike_counts = spikes_truncated.reshape(n_bins, steps_per_bin, num_neurons).sum(axis=1)
    
    # Determine the number of pairs to sample (limited by available neurons)
    max_pairs = num_neurons // 2
    n_pairs_actual = min(n_pairs, max_pairs)
    
    # Create disjoint pairs by pairing neurons sequentially
    # Shuffle indices to randomize pairing
    indices = jnp.arange(num_neurons)
    # For deterministic behavior, we'll use the first 2*n_pairs_actual neurons
    neuron_pairs = indices[:2 * n_pairs_actual].reshape(n_pairs_actual, 2)
    
    # Compute correlation coefficients for each pair
    correlations = []
    for i in range(n_pairs_actual):
        neuron_i, neuron_j = neuron_pairs[i]
        
        # Get spike counts for this pair
        counts_i = spike_counts[:, neuron_i]
        counts_j = spike_counts[:, neuron_j]
        
        # Compute correlation coefficient: Corr = Cov / sqrt(Var_i * Var_j)
        cov = jnp.cov(counts_i, counts_j)[0, 1]
        var_i = jnp.var(counts_i)
        var_j = jnp.var(counts_j)
        
        # Avoid division by zero
        if var_i > 0 and var_j > 0:
            corr = cov / jnp.sqrt(var_i * var_j)
            correlations.append(corr)
        else:
            correlations.append(jnp.nan)
    
    # Compute average correlation, ignoring NaN values
    correlations = jnp.array(correlations)
    avg_correlation = jnp.nanmean(correlations)
    
    return float(avg_correlation)
