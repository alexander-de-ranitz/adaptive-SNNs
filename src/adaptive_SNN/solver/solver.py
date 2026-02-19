from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import RESULTS, Euler, SaveAt, Solution, SubSaveAt
from jaxtyping import Array, PyTree

jax.config.update(
    "jax_enable_x64", True
)  # Use float64 for better numerical accuracy in the solver
default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

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
    args: PyTree = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    """
    Fixed-step Euler simulation of a (noisy) SNN model with selective state saving.

    Given a model, a solver, an initial state, and a time interval, simulates the model
    from `t0` to `t1` using fixed step size `dt0`. The state is saved as specified in SaveAt (see https://docs.kidger.site/diffrax/api/saveat/).
    Note that this solver might not interpret the SaveAt object indentically to diffrax. In particular, only a single SubSaveAt is currently supported in save_at.subs.
    Furthermore, the function `fn` in SaveAt is assumed to produce an output shape that is constant over time, i.e., the output shape does not depend on `t` or `y`.

    Notes:
      - Final interval may be shorter than `dt0` if (t1 - t0) is not an integer multiple of dt0.

    Args:
        model: Provides `terms(key)` and `update(t, y, args)`.
        solver: A diffrax solver implementing `.step(...)`.
        t0, t1: Simulation interval.
        dt0: Nominal step size.
        y0: Initial PyTree state.
        save_at: SaveAt object specifying when to save states.
        args: Extra args passed to solver/model (PyTree or None).
        key: Optional PRNG key. If None, a default key is used.

    Returns:
        diffrax.Solution with (ts, ys) containing saved times and states.
    """
    # Args are the default args, overridden by any user-provided args
    args = {**DEFAULT_ARGS, **(args or {})}

    # Compute number of steps and save indices/times based on save_at
    n_steps = jnp.ceil((t1 - t0) / dt0).astype(int)
    save_indices, save_times = compute_save_indices(save_at, t0, t1, dt0, n_steps)
    n_saves = save_indices.shape[0]

    # Preallocate storage for results
    save_fn = save_at.subs.fn
    y0_to_save = save_fn(0.0, y0, args)
    ys = jax.tree_util.tree_map(
        lambda x: jnp.empty((n_saves, *x.shape), x.dtype), y0_to_save
    )

    terms = model.terms(key)

    y_final, ys = run_simulation(
        t0, t1, n_steps, dt0, y0, ys, save_indices, save_fn, terms, args, model, solver
    )

    # If no states were saved, return the final state
    if ys is None:
        ys = y_final

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


@eqx.filter_jit
def save_state(args, save_fn, carry, t):
    """Helper to save the state at the specified index."""
    y, ys, save_index = carry
    ys = jax.tree_util.tree_map(
        lambda arr, v: arr.at[save_index].set(v), ys, save_fn(t, y, args)
    )
    return (y, ys, save_index + 1)


@eqx.filter_jit
def step(i, carry, t0, t1, dt, solver, terms, args, save_indices, model, save_fn):
    """Inner loop of the simulation. Takes one step and saves if needed."""
    y, ys, save_index = carry

    t0_64 = jnp.asarray(t0, dtype=jnp.float64)
    dt_64 = jnp.asarray(dt, dtype=jnp.float64)
    t1_64 = jnp.asarray(t1, dtype=jnp.float64)

    t_start = t0_64 + i * dt_64
    t_end = t0_64 + (i + 1) * dt_64
    t_end = jnp.minimum(t_end, t1_64)  # Ensure we don't go past t1

    y, _, _, _, _ = solver.step(terms, t_start, t_end, y, args, None, False)
    y = model.update(t_end, y, args)

    # Is save_index in save_indices? If so, save, else do nothing
    y, ys, save_index = jax.lax.cond(
        save_indices.size > 0
        and save_indices[save_index]
        == i
        + 1,  # Check if we need to save at this step (+1 because this is after the step)
        lambda c: save_state(args=args, save_fn=save_fn, carry=c, t=t_end),
        lambda c: c,
        (y, ys, save_index),
    )

    return (y, ys, save_index)


def run_simulation(
    t0, t1, n_steps, dt0, y0, ys, save_indices, save_fn, terms, args, model, solver
):
    """Runs the simulation loop."""

    # Partially apply all fixed arguments to the step function
    step_partial = partial(
        step,
        t0=t0,
        t1=t1,
        dt=dt0,
        solver=solver,
        terms=terms,
        args=args,
        save_indices=save_indices,
        model=model,
        save_fn=save_fn,
    )

    # Save the initial state if needed
    y0, ys, save_index = jax.lax.cond(
        save_indices.size > 0
        and save_indices[0] == 0,  # Check if we need to save at the initial time
        lambda c: save_state(args=args, save_fn=save_fn, carry=c, t=t0),
        lambda c: c,
        (y0, ys, 0),
    )

    # Loop over all intervals
    y_final, ys, save_index = jax.lax.fori_loop(
        0, n_steps, step_partial, (y0, ys, save_index)
    )

    # Ensure computation is complete before returning
    jax.tree.map(lambda x: x.block_until_ready(), (ys, y_final))

    return y_final, ys


def compute_save_indices(save_at: SaveAt, t0, t1, dt, nsteps) -> tuple[Array, Array]:
    """Computes the indices at which to save the state, based on the SaveAt object.

    Setting t0=True or t1=True will always include the initial and final times. Setting steps=True
    will save every step. Setting ts to an array of times will cause the state to be saved at the closest available timepoint to the desired save time.

    Note that the way SaveAt is interpreted here may differ from how diffrax interprets it.

    Args:
        save_at: SaveAt object specifying when to save states.
        t0: Start time of the simulation.
        t1: End time of the simulation.
        dt: Time step size.
        nsteps: Number of steps in the simulation.

    Returns:
        tuple of (save_indices, save_times), where save_indices is an array of indices at which to save, and save_times is an array of the corresponding times.
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
        save_indices = jnp.arange(
            0, nsteps + 1
        )  # We add 1, since we might save the initial state, and then after each step
        save_times = jnp.unique(jnp.clip(t0 + save_indices * dt, t0, t1))
        return save_indices, save_times
    else:
        if save_at.subs.ts is not None:
            save_times = jnp.unique(jnp.clip(save_at.subs.ts, t0, t1))
            save_indices = jnp.rint((save_times - t0) / dt)
        else:
            save_indices = jnp.array([], dtype=int)
            save_times = jnp.array([], dtype=default_float)
        if save_at.subs.t0:
            if save_indices.size == 0 or save_indices[0] != 0:
                save_indices = jnp.insert(save_indices, 0, 0)
                save_times = jnp.insert(save_times, 0, t0)
        if save_at.subs.t1:
            if save_indices.size == 0 or save_indices[-1] != nsteps:
                save_indices = jnp.append(save_indices, nsteps)
                save_times = jnp.append(save_times, t1)
        return save_indices, save_times
