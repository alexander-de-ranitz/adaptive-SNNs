import jax

jax.config.update("jax_enable_x64", True)
import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from helpers import (
    DeterministicLIFNetwork,
    DeterministicOUP,
    DummyEnvironment,
    allclose_pytree,
    get_non_inf_ts_ys,
    make_default_args,
)

from adaptive_SNN.models import AgentEnvSystem, SystemState
from adaptive_SNN.models.networks import LIFNetwork
from adaptive_SNN.models.networks.agent import Agent
from adaptive_SNN.models.reward_prediction import MovingAverageRewardPredictor
from adaptive_SNN.solver import solve_ODE


def test_solver_timesteps():
    N_neurons = 4
    N_inputs = 0
    t0, t1, dt0 = 0.0, 1.00, 0.1

    model = DeterministicOUP(tau=1.0, noise_std=1.0, dim=N_neurons, t0=t0, t1=t1 + dt0)

    # Prepare initial state from model
    y0 = model.initial
    solver = dfx.Euler()
    args = make_default_args(N_neurons, N_inputs)

    # Our method
    saveat = dfx.SaveAt(t0=True, t1=True, steps=True)
    sol_1 = solve_ODE(model, solver, t0, t1, dt0, y0, save_at=saveat, args=args)
    sol_1_ts = sol_1.ts

    # Direct diffrax call for comparison
    terms = model.terms(jr.PRNGKey(0))
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
        stepsize_controller=dfx.ConstantStepSize(),
    )

    # Remove any -inf timepoints from sol_2 (pre-allocated but not used)
    sol_2_ts, _ = get_non_inf_ts_ys(sol_2)

    assert sol_1_ts.shape == sol_2_ts.shape
    assert jnp.allclose(sol_1_ts, sol_2_ts)


def test_solver_timesteps_precision():
    N_neurons = 1
    N_inputs = 0
    t0, t1, dt0 = 10**4, 10**4 + 1e-4, 1e-4

    model = DeterministicOUP(tau=1.0, noise_std=1.0, dim=N_neurons, t0=t0, t1=t1 + dt0)

    # Prepare initial state from model
    y0 = model.initial

    solver = dfx.Euler()
    args = make_default_args(N_neurons, N_inputs)

    # Our method
    save_at = dfx.SaveAt(subs=dfx.SubSaveAt(steps=True, t0=True, t1=True))
    sol_1 = solve_ODE(model, solver, t0, t1, dt0, y0, save_at=save_at, args=args)
    sol_1_ts = sol_1.ts
    # Check that time steps are consistent with dt0
    actual_dts = jnp.ediff1d(sol_1_ts)
    assert jnp.allclose(actual_dts, dt0, rtol=1e-6, atol=1e-6)


def test_solver_output():
    """Tests that our custom solver function produces the same output as a direct diffrax call.
    Note that this only works when our model does not spike, this is not implemented in the diffrax solver."""

    N_neurons = 3
    N_inputs = 0
    t0, t1, dt0 = 0.0, 0.01, 1e-4
    key = jr.PRNGKey(0)

    network = LIFNetwork(N_neurons=N_neurons, N_inputs=N_inputs, dt=dt0, key=key)

    noise_model = DeterministicOUP(
        tau=network.tau_E, noise_std=1e-9, dim=N_neurons, t0=t0, t1=t1 + dt0
    )

    network = DeterministicLIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt0,
        key=key,
        noise_model=noise_model,
        t0=t0,
        t1=t1 + dt0,
    )

    agent = Agent(
        neuron_model=network, reward_prediction_model=MovingAverageRewardPredictor()
    )

    model = AgentEnvSystem(
        agent=agent,
        environment=DummyEnvironment(),
        agent_output_shape=(N_neurons,),
    )

    # Prepare initial state from model
    y0 = model.initial

    # Set some arbitrary voltages to ensure non-zero drift
    y0 = eqx.tree_at(
        lambda s: s.agent_state.network_state.V, y0, jnp.array([-65e-3, -70e-3, -73e-3])
    )

    solver = dfx.Euler()
    args = make_default_args(N_neurons, N_inputs)

    args["reward_fn"] = lambda t, x, args: jnp.sum(x.environment_state).reshape(
        (1,)
    )  # Dummy reward function that depends on the environment state
    args["network_output_fn"] = (
        lambda t, x, args: x.network_state.V
    )  # Some random function of the network state as output

    # Save some random part of the state for comparison
    # imporantly, we should not compare any fields that are modified
    # using the update() or pre_step_update() functions, as these are not implemented diffrax
    def save_fn(t, y: SystemState, args):
        return (
            y.agent_state.network_state.V,
            y.environment_state,
            y.agent_state.network_state.perturbations,
        )

    # Our method
    save_at = dfx.SaveAt(subs=dfx.SubSaveAt(fn=save_fn, steps=True, t0=True, t1=True))
    sol_1 = solve_ODE(
        model, solver, t0, t1, dt0, y0, save_at=save_at, args=args, key=key
    )

    # Direct diffrax call for comparison
    terms = model.terms(key)
    sol_2 = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=save_at,
        args=args,
        adjoint=dfx.ForwardMode(),
    )

    sol_2_ts, sol_2_ys = get_non_inf_ts_ys(sol_2)

    print(sol_2_ys[-3:])
    assert jnp.allclose(sol_1.ts, sol_2_ts)
    assert allclose_pytree(sol_1.ys, sol_2_ys)
