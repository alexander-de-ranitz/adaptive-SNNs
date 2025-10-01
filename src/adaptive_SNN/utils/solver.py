import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import RESULTS, Euler, Solution
from jaxtyping import PyTree
from joblib.memory import Memory

from adaptive_SNN.models.models import AgentSystem, NoisyNetwork

memory = Memory(location="./.cache", verbose=0)


def simulate_noisy_SNN(
    model: NoisyNetwork | AgentSystem,
    solver: Euler,
    t0: float,
    t1: float,
    dt0: float,
    y0: PyTree,
    save_every_n_steps: int = 1,
    args: PyTree = None,
):
    """
    Run a simulation using the specified solver and terms.

    Steps through the differential equation defined by `terms` from time `t0` to `t1` with increments of `dt0`.

    Args:
        model (NoisyNeuronModel): The neuron model containing the terms.
        solver (Euler): The solver to use for integration.
        t0 (float): Initial time.
        t1 (float): Final time.
        dt0 (float): Initial time step.
        y0 (PyTree): Initial state.
        save_every_n_steps (int, optional): Interval of steps to save the state. Defaults to 1 (save every step).
        args (PyTree, optional): Additional arguments for the terms. Defaults to None.

    Returns:
        Solution object containing times and states.
    """

    # Helper function to add current state to ys
    def add_to_ys(ys, y, index):
        return jax.tree.map(lambda arr, v: arr.at[index].set(v), ys, y)

    # Set up solution parameters
    times = jnp.arange(t0, t1, dt0)
    if times[-1] < t1:
        times = jnp.append(times, t1)  # Ensure t1 is included
    n_saves = len(times) // save_every_n_steps

    # Set up storage for results
    ys = jax.tree.map(lambda x: jnp.empty(shape=(n_saves, *x.shape)), y0)
    ys = add_to_ys(ys, y0, index=0)
    save_index = 1

    step = 0
    y = y0
    terms = model.terms(jr.PRNGKey(0))
    for t in times:
        y, _, _, _, result = solver.step(terms, t, t + dt0, y, args, None, False)
        if result != RESULTS.successful:
            raise RuntimeError(f"Solver step failed with result: {result}")
        step += 1

        y = model.compute_spikes_and_update(t, y, args)

        # Save results if at the correct interval
        if step % save_every_n_steps == 0:
            ys = add_to_ys(ys, y, save_index)
            save_index += 1

    return Solution(
        t0=t0,
        t1=t1,
        ts=times,
        ys=ys,
        interpolation=None,
        stats=None,
        result=RESULTS.successful,
        solver_state=None,
        controller_state=None,
        made_jump=None,
        event_mask=None,
    )


def simulate_learning_SNN(
    model: LearningModel,
    solver: Euler,
    t0: float,
    t1: float,
    dt0: float,
    y0: PyTree,
    save_every_n_steps: int = 1,
    args: PyTree = None,
):
    """
    Run a simulation of the LearningModel using the specified solver.

    Steps through the differential equation defined by `terms` of the model from time `t0` to `t1` with increments of `dt0`.
    y0 is the initial state of the model, which is (network_state, reward_state, environment_state).

    Args:
        model (LearningModel): The learning neuron model containing the terms.
        solver (Euler): The solver to use for integration.
        t0 (float): Initial time.
        t1 (float): Final time.
        dt0 (float): Initial time step.
        y0 (PyTree): Initial state.
        save_every_n_steps (int, optional): Interval of steps to save the state. Defaults to 1 (save every step).
        args (PyTree, optional): Additional arguments for the terms. Defaults to None.

    Returns:
        Solution object containing times and states.
    """
