from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import RESULTS, Euler, SaveAt, Solution, SubSaveAt
from jaxtyping import Array, PyTree

# Default args for simulate_noisy_SNN, can be overridden by user-provided args
# provided here for convenience, such that you only need to specify the args relevant to the experiment
# note that is important that these are saved outside of the function definition to avoid recompilation on each call
DEFAULT_ARGS = {
    "get_learning_rate": lambda t, x, args: 0.0,
    "get_input_spikes": lambda t, x, args: jnp.zeros(
        shape=(
            x.W.shape[0],
            x.W.shape[1] - x.W.shape[0],
        )  # shape = (n_neurons, n_inputs)
    ),
    "get_desired_balance": lambda t, x, args: 0.0,
    "RPE": jnp.array(0.0),
}


def simulate_noisy_SNN(
    model,
    solver: Euler,
    t0: float,
    t1: float,
    dt0: float,
    y0: PyTree,
    save_at: SaveAt,
    save_fn: Callable = None,
    args: PyTree = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    """
    Fixed-step Euler simulation of a (noisy) SNN model with selective state saving.

    Given a model, a solver, an initial state, and a time interval, simulates the model
    from `t0` to `t1` using fixed step size `dt0`. The state is saved as specified in SaveAt (see https://docs.kidger.site/diffrax/api/saveat/).
    Note that this solver might not interpret the SaveAt object indentically to diffrax. In particular, only a single SubSaveAt is currently supported in save_at.subs.

    Notes:
      - Final interval may be shorter than `dt0` if (t1 - t0) is not an integer multiple of dt0.

    Args:
        model: Provides `terms(key)` and `update(t, y, args)`.
        solver: A diffrax solver implementing `.step(...)`.
        t0, t1: Simulation interval.
        dt0: Nominal step size.
        y0: Initial PyTree state.
        save_at: SaveAt object specifying when to save states.

        save_fn: Optional function `fn(y) -> state_to_save' used to determine how to save the state (e.g. only save part of the current state, or some summary statistics). If None, saves the full state.
                    if the function returns None, no states are saved and only the final state is returned.
        args: Extra args passed to solver/model (PyTree or None).
        key: Optional PRNG key. If None, a default key is used.

    Returns:
        diffrax.Solution with (ts, ys) containing saved times and states.
    """
    # Args are the default args, overridden by any user-provided args
    args = {**DEFAULT_ARGS, **(args or {})}

    # Default save function is identity
    if save_fn is None:
        save_fn = lambda y: y

    # Compute time grid
    n_steps = jnp.floor((t1 - t0) / dt0).astype(int)
    times = t0 + dt0 * jnp.arange(n_steps + 1)  # Add 1 for t1
    times = times.at[-1].set(t1)  # Make sure last time is exactly t1
    save_mask = compute_save_mask(save_at, times)
    n_saves = jnp.sum(save_mask)

    # Preallocate storage for results
    y0_to_save = save_fn(y0)
    ys = jax.tree_util.tree_map(
        lambda x: jnp.empty((n_saves, *x.shape), x.dtype), y0_to_save
    )

    terms = model.terms(key)

    y_final, ys = run_simulation(
        times, y0, ys, save_mask, save_fn, terms, args, model, solver
    )

    # If no states were saved, return the final state
    if ys is None:
        ys = y_final

    return Solution(
        t0=t0,
        t1=t1,
        ts=times[save_mask],
        ys=ys,
        interpolation=None,
        stats=None,
        result=RESULTS.successful,
        solver_state=None,
        controller_state=None,
        made_jump=None,
        event_mask=None,
    )


@eqx.filter_jit
def run_simulation(times, y0, ys, save_mask, save_fn, terms, args, model, solver):
    """Runs the simulation loop."""

    # Utility function to save the current state
    def save_state(carry, save_fn):
        y, ys, save_index = carry
        ys = jax.tree_util.tree_map(
            lambda arr, v: arr.at[save_index].set(v), ys, save_fn(y)
        )
        return (y, ys, save_index + 1)

    def step(i, carry):
        """Inner loop of the simulation. Takes one step and saves if needed."""
        y, ys, save_index = carry

        # Take a step
        t_start = times[i]
        t_end = times[i + 1]
        y, _, _, _, _ = solver.step(terms, t_start, t_end, y, args, None, False)
        y = model.update(t_end, y, args)

        # Is save_mask true at this index? If so, save, else do nothing
        y, ys, save_index = jax.lax.cond(
            save_mask[i + 1],  # + 1 because we already stepped
            lambda c: save_state(c, save_fn),
            lambda c: c,
            (y, ys, save_index),
        )

        return (y, ys, save_index)

    # Save the initial state if needed
    y0, ys, save_index = jax.lax.cond(
        save_mask[0], lambda c: save_state(c, save_fn), lambda c: c, (y0, ys, 0)
    )

    # Loop over all intervals
    y_final, ys, save_index = jax.lax.fori_loop(
        0, times.size - 1, step, (y0, ys, save_index)
    )
    return y_final, ys


def compute_save_mask(save_at: SaveAt, times: Array) -> Array:
    """Computes the indices at which to save the state, based on the SaveAt object.

    Setting t0=True or t1=True will always include the initial and final times. Setting steps=True
    will save every step. Setting ts to an array of times will cause the state to be saved at the closest available timepoint to the desired save time.

    Note that the way SaveAt is interpreted here may differ from how diffrax interprets it.

    Args:
        save_at: SaveAt object specifying when to save states.
        times: Array of all time points in the simulation.

    Returns:
        Binary array of same length as times, with True at indices where the state should be saved.
    """
    if not isinstance(save_at.subs, SubSaveAt):
        raise NotImplementedError(
            "simulate_noisy_SNN currently only supports a single SubSaveAt in save_at.subs."
        )
    if save_at.subs.steps:
        if save_at.subs.ts is not None:
            raise ValueError(
                "Both steps=True and ts specified in save_at.subs; only one of these can be used."
            )
        # Save every step
        save_every_n_steps = 1
        save_mask = (jnp.arange(times.size) % save_every_n_steps) == 0
    else:
        # Save at specified times using a vectorized search, choosing the closer of (i-1) or i
        save_mask = jnp.zeros(times.shape, dtype=bool)

        if save_at.subs.t0:
            save_mask = save_mask.at[0].set(True)
        if save_at.subs.t1:
            save_mask = save_mask.at[-1].set(True)

        if save_at.subs.ts is not None:
            save_ts = jnp.asarray(save_at.subs.ts)

            # Only consider requested times that fall within [times[0], times[-1]]
            valid = (save_ts >= times[0]) & (save_ts <= times[-1])
            save_ts = save_ts[valid]

            if save_ts.size > 0:
                # First index i s.t. times[i] >= st
                idx_right = jnp.searchsorted(times, save_ts, side="left")

                # Candidate neighbors: previous (i-1) and next (i)
                prev_idx = jnp.clip(idx_right - 1, 0, times.size - 1)
                next_idx = jnp.clip(idx_right, 0, times.size - 1)

                prev_diff = jnp.abs(times[prev_idx] - save_ts)
                next_diff = jnp.abs(times[next_idx] - save_ts)

                # We save at the closest of the two timepoints
                closest_idx = jnp.where(
                    (idx_right > 0) & (prev_diff < next_diff),
                    prev_idx,
                    next_idx,
                )
                save_mask = save_mask.at[closest_idx].set(True)
    return save_mask
