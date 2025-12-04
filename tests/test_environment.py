import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
from helpers import DummySpikingNetwork

from adaptive_SNN.models import AgentEnvSystem, SystemState
from adaptive_SNN.models.environments import (
    DoubleIntegrator,
    InputTrackingEnvironment,
    SpikeRateEnvironment,
)
from adaptive_SNN.solver import simulate_noisy_SNN


def test_input_tracking_env():
    env = InputTrackingEnvironment(dim=1)

    args = {"get_env_input": lambda t, x, args: jnp.array([1.0])}
    solver = dfx.Euler()
    t0 = 0.0
    t1 = 10.0
    dt0 = 0.01
    y0 = env.initial
    key = jr.PRNGKey(0)
    terms = env.terms(key)
    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(t1=True),
        adjoint=dfx.ForwardMode(),
    )

    ts = sol.ts
    ys = sol.ys
    assert ys.shape == (ts.shape[0], env.dim)
    assert jnp.allclose(ys[-1], jnp.array([1.0]), atol=1e-2)


def test_spike_rate_env():
    solver = dfx.Euler()
    t0 = 0.0
    t1 = 5.0
    dt0 = 0.001
    dim = 5

    rates = jnp.linspace(1.0, 5.0, dim) * 10

    env = SpikeRateEnvironment(dim=dim)
    network = DummySpikingNetwork(output_dim=dim, output_rates=rates, dt=dt0)

    model = AgentEnvSystem(
        agent=network,
        environment=env,
    )

    args = {
        "network_output_fn": lambda t, agent_state, args: agent_state.S,
        "reward_fn": lambda t, environment_state, args: 0.0,
    }

    y0 = model.initial
    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt0,
        y0,
        save_at=dfx.SaveAt(ts=jnp.arange(t0, t1, dt0)),
        args=args,
    )

    ts = sol.ts
    ys: SystemState = sol.ys

    # Check shapes
    assert ys.environment_state.shape == (ts.shape[0], env.dim)

    # Check that the final environment state is close to the target rates
    # there tolerance is quite large since the tracking is not exact
    assert jnp.allclose(ys.environment_state[-1], rates, atol=1.0)


def test_double_integrator_optimal_control():
    env = DoubleIntegrator()

    # Define args with optimal LQR controller for double integrator
    args = {
        "get_env_input": lambda t, x, args: jnp.array(
            -1.0 * x.at[0].get() - jnp.sqrt(1 / 3.0) * x.at[1].get()
        )
    }

    solver = dfx.Euler()
    t0 = 0.0
    t1 = 30.0
    dt0 = 0.01

    # Initial state: position=1.0, velocity=0.0
    y0 = jnp.array([[1.0, 0.0]]).reshape((2,))

    key = jr.PRNGKey(0)
    sol = simulate_noisy_SNN(
        env,
        solver,
        t0,
        t1,
        dt0,
        y0,
        save_at=dfx.SaveAt(t1=True),
        args=args,
        key=key,
    )
    ys = sol.ys

    # The system should converge to zero state under optimal control
    assert jnp.allclose(ys[-1], jnp.array([0.0, 0.0]), atol=1e-2)


def test_double_integrator_simple_control():
    env = DoubleIntegrator()

    # Acceleration of 1 for 1 second, then 0
    args = {
        "get_env_input": lambda t, x, args: jnp.where(
            t < 1.0, jnp.array(1.0), jnp.array(0.0)
        )
    }

    solver = dfx.Euler()
    t0 = 0.0
    t1 = 5.0
    dt0 = 0.01
    y0 = env.initial
    key = jr.PRNGKey(0)
    sol = simulate_noisy_SNN(
        env,
        solver,
        t0,
        t1,
        dt0,
        y0,
        save_at=dfx.SaveAt(t1=True),
        args=args,
        key=key,
    )
    ys = sol.ys

    # The system should be at position 4.5 with velocity 1.0 at t=5
    assert jnp.allclose(ys[-1], jnp.array([4.5, 1.0]), atol=1e-2)
