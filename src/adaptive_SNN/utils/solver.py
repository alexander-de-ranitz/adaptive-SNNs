import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import jax
from typing import Optional
from jaxtyping import Array, Bool, PyTree, Scalar
from diffrax._custom_types import DenseInfo
from diffrax import Euler
from typing import Tuple
from diffrax import RESULTS, AbstractTerm, Solution
from typing_extensions import TypeAlias
from adaptive_SNN.models.models import NoisyNeuronModel


_ErrorEstimate: TypeAlias  = None
_SolverState: TypeAlias = None

def run_SNN_simulation(model: NoisyNeuronModel, solver: Euler, t0: float, t1: float, dt0: float, y0: PyTree, save_every_n_steps: int = 1, args: PyTree = None):
    """
    Run a simulation using the specified solver and terms.

    Steps through the differential equation defined by `terms` from time `t0` to `t1` with increments of `dt0`.
    y_0 is ((V, conductances, spikes), noise_E, noise_I).

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
    # Set up solution parameters
    times = jnp.arange(t0, t1, dt0)
    n_saves = len(times) // save_every_n_steps + 1
    ys = jax.tree.map(lambda x: jnp.empty(shape=(n_saves, *x.shape)), y0)
    ts = [t0]
    terms = model.terms(jrandom.PRNGKey(0))

    # Spiking parameters from neuron model
    V_threshold = model.network.firing_threshold
    V_reset = model.network.V_reset

    # Helper function to add current state to ys
    def add_to_ys(ys, y, index):
        return index+1, jax.tree.map(lambda arr, v: arr.at[index].set(v), ys, y)
    
    save_index, ys = add_to_ys(ys, y0, 0)
    step = 0
    y = y0
    for t in times:
        y, _, _, _, result = solver.step(terms, t, t+dt0, y, args, None, False)
        if result != RESULTS.successful:
            raise RuntimeError(f"Solver step failed with result: {result}")
        step += 1
        
        V = y[0][0] # y = ((V, conductances, spikes), noise_E, noise_I)
        spikes = jnp.float32(V > V_threshold)
        V_new = (1.0 - spikes) * V + spikes * V_reset
        y = ((V_new, y[0][1], spikes), y[1], y[2]) # reset V and set spikes
        
        # Save results if at the correct interval
        if step % save_every_n_steps == 0:
            save_index, ys = add_to_ys(ys, y, save_index)
            ts.append(t + dt0)

    return Solution(t0=t0, t1=t1, ts=ts, ys=ys, interpolation=None, stats=None, result=RESULTS.successful, solver_state=None, controller_state=None, made_jump=None, event_mask=None)
    
class SpikingEuler(Euler):
    """
    Custom solver to solve spiking neuron models with firing thresholds.

    After each integration step, checks if the membrane potential exceeds the firing threshold (Vthresh).
    If so, resets the membrane potential to the resting potential (EL). y is a tuple (V, spikes).
    """

    V_threshold: float
    V_reset: float

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
       
        y1, _, dense_info, _, RESULTS.successful = super().step(terms, t0, t1, y0, args, solver_state, made_jump)

        V = y1[0]

        # membrane potential reset
        spikes = jnp.float32(V > self.V_threshold)
        Vnew = (1.0 - spikes) * V + spikes * self.V_reset
    
        return (Vnew, spikes), None, dense_info, None, RESULTS.successful