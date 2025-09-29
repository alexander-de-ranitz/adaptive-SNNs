import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import RESULTS, Euler, Solution
from jaxtyping import PyTree
from joblib.memory import Memory
from typing_extensions import TypeAlias

from adaptive_SNN.models.models import LearningModel, NoisyNeuronModel

memory = Memory(location="./.cache", verbose=0)

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


@memory.cache
def run_SNN_simulation_cached(
    model: LearningModel | NoisyNeuronModel,
    solver: Euler,
    t0: float,
    t1: float,
    dt0: float,
    y0: PyTree,
    save_every_n_steps: int = 1,
    args: PyTree = None,
):
    """
    Cached version of run_SNN_simulation to avoid recomputation for the same parameters.
    """
    return simulate_noisy_SNN(model, solver, t0, t1, dt0, y0, save_every_n_steps, args)


def simulate_noisy_SNN(
    model: NoisyNeuronModel,
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
    y0 is ((V, W, G), noise_E, noise_I).

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
    # TODO: think about how to elegantly store and pass around the spikes. Should they be part of y?

    N_neurons = model.network.N_neurons
    N_inputs = model.network.N_inputs

    # Helper function to add current state to ys
    def add_to_ys(ys, y, index):
        return jax.tree.map(lambda arr, v: arr.at[index].set(v), ys, y)

    # Set up solution parameters
    times = jnp.arange(t0, t1, dt0)
    if times[-1] < t1:
        times = jnp.append(times, t1)  # Ensure t1 is included

    n_saves = len(times) // save_every_n_steps
    terms = model.terms(jr.PRNGKey(0))

    # Set up storage for results
    ys = jax.tree.map(lambda x: jnp.empty(shape=(n_saves, *x.shape)), y0)
    spikes_hist = jnp.empty(shape=(n_saves, N_neurons + N_inputs))

    # Set states for t=t0
    ys = add_to_ys(ys, y0, index=0)
    spikes_hist = spikes_hist.at[0].set(jnp.zeros((N_neurons + N_inputs,)))

    # If args contains 'p', set up input spikes as Poisson process
    # TODO: move this to a more sensible place and make input spikes more flexible
    if args and "p" in args:
        args["input_spikes"] = lambda t, x, args: jr.bernoulli(
            jr.PRNGKey(int(t / dt0)), p=args["p"], shape=(N_inputs,)
        )

    save_index = 1  # Start saving from the first index after initial
    step = 0
    y = y0
    for t in times:
        y, _, _, _, result = solver.step(terms, t, t + dt0, y, args, None, False)
        if result != RESULTS.successful:
            raise RuntimeError(f"Solver step failed with result: {result}")
        step += 1

        new_network_state, spikes = model.compute_spikes_and_update(t, y, args)
        y = (new_network_state, y[1], y[2])  # Update state with new network state

        # Save results if at the correct interval
        if step % save_every_n_steps == 0:
            spikes_hist = spikes_hist.at[save_index].set(spikes)
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
    ), spikes_hist


def run_learning_simulation():
    pass  # TODO: implement learning simulation function
