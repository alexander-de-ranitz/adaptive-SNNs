from jax import numpy as jnp
from jax import random as jr


def poisson_rate_code(
    rate: float, dt: float, key: jr.PRNGKey, encoding_shape
) -> jnp.ndarray:
    """
    Generates Poissonian spikes based on the given firing rate.

    The function simulates a Poisson process where spikes occur with a probability
    proportional to the firing rate and the time step size. Each neuron in the encoding_size
    fires independently with equal rates.

    Args:
        rate (float): The firing rate in Hz.
        encoding_size (int): The number of neurons used to encode the signal
        dt (float): The time step size in seconds.

    Returns:
        jnp.ndarray: A binary array of shape (encoding_size,) representing the spike train.
    """
    # Calculate the probability of a spike occurring in each time step
    p_spike = rate * dt
    spikes = jr.bernoulli(key, p=p_spike, shape=encoding_shape)
    return spikes
