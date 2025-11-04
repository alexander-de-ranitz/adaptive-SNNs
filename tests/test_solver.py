import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
from helpers import (
    DeterministicNoisyNeuronModel,
    DeterministicOUP,
    allclose_pytree,
    get_non_inf_ts_ys,
    make_default_args,
)

from adaptive_SNN.models import LIFNetwork, NoisyNetworkState
from adaptive_SNN.solver import simulate_noisy_SNN


def test_solver_timesteps():
    N_neurons = 4
    N_inputs = 0
    t0, t1, dt0 = 0.0, 1.0, 0.1

    key = jr.PRNGKey(0)

    network = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, dt=dt0, key=key)

    noise_model = DeterministicOUP(
        tau=network.tau_E, noise_scale=0.0, dim=N_neurons, t0=t0, t1=t1 + dt0
    )

    model = DeterministicNoisyNeuronModel(
        neuron_model=network,
        noise_model=noise_model,
        t0=t0,
        t1=t1 + dt0,
    )

    # Prepare initial state from model
    y0 = model.initial
    solver = dfx.Euler()
    args = make_default_args(N_neurons, N_inputs)

    # Our method
    save_every = 1
    sol_1 = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, y0, save_every_n_steps=save_every, args=args
    )
    sol_1_ts = sol_1.ts

    # Direct diffrax call for comparison
    terms = model.terms(jr.PRNGKey(0))
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sol_2 = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
    )

    # Remove any -inf timepoints from sol_2 (pre-allocated but not used)
    sol_2_ts, _ = get_non_inf_ts_ys(sol_2)

    assert sol_1_ts.shape == sol_2_ts.shape
    assert jnp.allclose(sol_1_ts, sol_2_ts)


def test_solver_output():
    """Tests that our custom solver function produces the same output as a direct diffrax call.
    Note that this only works when our model does not spike, this is not implemented in the diffrax solver."""

    N_neurons = 3
    N_inputs = 0
    t0, t1, dt0 = 0.0, 0.01, 1e-4
    key = jr.PRNGKey(0)

    network = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, dt=dt0, key=key)

    noise_model = DeterministicOUP(
        tau=network.tau_E, noise_scale=2e-16, dim=N_neurons, t0=t0, t1=t1 + dt0
    )

    model = DeterministicNoisyNeuronModel(
        neuron_model=network,
        noise_model=noise_model,
        t0=t0,
        t1=t1 + dt0,
    )

    # Prepare initial state from model
    y0 = model.initial
    solver = dfx.Euler()
    args = make_default_args(N_neurons, N_inputs)

    # Define a save function that extracts only the relevant parts of the state. Necessary because
    # diffrax does not update the spike buffer in the same way as our custom solver.
    def save_fn_custom(y: NoisyNetworkState):
        return (
            (y.network_state.V, y.network_state.G, y.network_state.S),
            y.noise_state,
        )

    def save_fn_dfx(t, y: NoisyNetworkState, args):
        return (
            (y.network_state.V, y.network_state.G, y.network_state.S),
            y.noise_state,
        )

    # Our method
    save_every = 1
    sol_1 = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt0,
        y0,
        save_every_n_steps=save_every,
        save_fn=save_fn_custom,
        args=args,
    )

    # Direct diffrax call for comparison
    terms = model.terms(jr.PRNGKey(0))
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True, fn=save_fn_dfx)
    sol_2 = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        args=args,
        adjoint=dfx.ForwardMode(),
    )

    sol_2_ts, sol_2_ys = get_non_inf_ts_ys(sol_2)

    assert jnp.allclose(sol_1.ts, sol_2_ts)
    assert allclose_pytree(sol_1.ys, sol_2_ys)
