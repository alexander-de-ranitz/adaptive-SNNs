import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import RESULTS, Euler, Solution
from jaxtyping import PyTree

from adaptive_SNN.models.models import AgentSystem, NoisyNetwork


def simulate_noisy_SNN(
    model: NoisyNetwork | AgentSystem,
    solver: Euler,
    t0: float,
    t1: float,
    dt0: float,
    y0: PyTree,
    save_every_n_steps: int = 1,  # TODO: use Diffrax' SaveAt class instead for better compatibility
    args: PyTree = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
):
    # Compute time grid
    n_steps = jnp.floor((t1 - t0) / dt0).astype(int)
    times = t0 + dt0 * jnp.arange(n_steps + 1)  # Add 1 for t1
    times = times.at[-1].set(t1)  # Make sure last time is exactly t1

    save_mask = (jnp.arange(times.size) % save_every_n_steps) == 0
    save_times = times[save_mask]
    n_saves = save_times.size

    # Preallocate storage for results
    ys = jax.tree_util.tree_map(lambda x: jnp.empty((n_saves, *x.shape), x.dtype), y0)

    terms = model.terms(key)

    @eqx.filter_jit
    def run_simulation(times, y0, ys, save_mask, terms, args, model, solver):
        """Runs the simulation loop."""

        # Utility function to save the current state
        def save_state(carry):
            y, ys, save_index = carry
            ys = jax.tree_util.tree_map(lambda arr, v: arr.at[save_index].set(v), ys, y)
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
                save_state,
                lambda c: c,
                (y, ys, save_index),
            )

            return (y, ys, save_index)

        # Save the initial state if needed
        y0, ys, save_index = jax.lax.cond(
            save_mask[0], save_state, lambda c: c, (y0, ys, 0)
        )

        # Loop over all intervals
        y_final, ys, save_index = jax.lax.fori_loop(
            0, times.size - 1, step, (y0, ys, save_index)
        )
        return ys

    ys = run_simulation(times, y0, ys, save_mask, terms, args, model, solver)

    return Solution(
        t0=float(save_times[0]),
        t1=float(save_times[-1]),
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
