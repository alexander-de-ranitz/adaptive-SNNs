from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import RESULTS, Euler, Solution
from jaxtyping import PyTree

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
    save_every_n_steps: int = 1,  # TODO: use Diffrax' SaveAt class instead for better compatibility
    save_fn: Callable = None,
    args: PyTree = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    """
    Fixed-step Euler simulation of a (noisy) SNN model with selective state saving.

    Given a model, a solver, an initial state, and a time interval, simulates the model
    from `t0` to `t1` using fixed step size `dt0`. The state is saved every `save_every_n_steps` steps.

    Notes:
      - Final interval may be shorter than `dt0` if (t1 - t0) is not an integer multiple of dt0.

    Args:
        model: Provides `terms(key)` and `update(t, y, args)`.
        solver: A diffrax solver implementing `.step(...)`.
        t0, t1: Simulation interval.
        dt0: Nominal step size.
        y0: Initial PyTree state.
        save_every_n_steps: After how many steps to save state (>=1).
        save_fn: Optional function `fn(y) -> state_to_save' used to determine how to save the state (e.g. only save part of the current state, or some summary statistics). If None, saves the full state.
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

    save_mask = (jnp.arange(times.size) % save_every_n_steps) == 0
    save_times = times[save_mask]
    n_saves = save_times.size

    # Preallocate storage for results
    y0_to_save = save_fn(y0)
    ys = jax.tree_util.tree_map(
        lambda x: jnp.empty((n_saves, *x.shape), x.dtype), y0_to_save
    )

    terms = model.terms(key)

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
        return ys

    ys = run_simulation(times, y0, ys, save_mask, save_fn, terms, args, model, solver)

    return Solution(
        t0=t0,
        t1=t1,
        ts=save_times,
        ys=ys,
        interpolation=None,
        stats=None,
        result=RESULTS.successful,
        solver_state=None,
        controller_state=None,
        made_jump=None,
        event_mask=None,
    )
