from jax import numpy as jnp
from jaxtyping import Array


def poisson_grid_encoding(
    state: Array,
    grid_shape: tuple[int, ...],
    grid_range: list[tuple[float, float]],
    tuning_curve_width=None,
    total_rate=1500,
):
    """Encodes a continuous state into a grid of Poisson firing rates using Gaussian tuning curves.

    Args:
        state: A 1D array of shape (n_dim,) representing the continuous state to encode.
        grid_shape: A tuple of n_dim integers representing the number neurons along each dimension of the grid.
        grid_range: A list of n_dim tuples specifying the (min, max) range for each dimension of the state space.
        tuning_curve_width: Optional; a 1D array of shape (n_dim,) specifying the width of the Gaussian tuning curves for each dimension as a fraction of the grid range. If none, defaults to 2 * 1/grid_shape for each dimension.
        total_rate: The total firing rate to be distributed across the grid of neurons. The sum over the returned rates exactly equals total_rate.

    Returns:
        A array of shape grid_shape representing the firing rates of the neurons in the grid.
    """
    n_dim = len(grid_shape)
    if len(grid_range) != n_dim:
        raise ValueError(
            f"grid_range must have length {n_dim}, but got {len(grid_range)}"
        )
    if state.shape[0] != n_dim:
        raise ValueError(f"state must have length {n_dim}, but got {state.shape[0]}")
    if tuning_curve_width is None:
        tuning_curve_width = jnp.array(
            [2 * (r[1] - r[0]) / (s) for r, s in zip(grid_range, grid_shape)]
        )

    # Grid of tuning curve centers
    centres = [jnp.linspace(r[0], r[1], s) for r, s in zip(grid_range, grid_shape)]
    preferred_inputs = jnp.stack(
        jnp.meshgrid(*centres, indexing="ij"), axis=-1
    ).reshape(-1, n_dim)

    rates = jnp.exp(
        -jnp.sum((state - preferred_inputs) ** 2 / (2 * tuning_curve_width**2), axis=-1)
    ).reshape(grid_shape)
    rates = rates / jnp.sum(rates) * total_rate
    return rates


if __name__ == "__main__":
    # Example usage
    state = jnp.array([0.5, -0.5])
    grid_shape = (4, 3)
    grid_range = [(-1.0, 1.0), (-1.0, 1.0)]
    rates = poisson_grid_encoding(state, grid_shape, grid_range)
    print(rates)
