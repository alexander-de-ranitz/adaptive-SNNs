import diffrax as dfx
import jax.numpy as jnp
from helpers import DummySpikingNetwork

from adaptive_SNN.models import AgentEnvSystem, SystemState
from adaptive_SNN.models.environments import (
    SpikeRateEnvironment,
)
from adaptive_SNN.models.networks import Agent
from adaptive_SNN.models.reward_prediction import MovingAverageRewardPredictor
from adaptive_SNN.solver import solve_ODE


def test_spike_rate_env():
    solver = dfx.Euler()
    t0 = 0.0
    t1 = 5.0
    dt0 = 0.001
    dim = 5

    rates = jnp.linspace(1.0, 5.0, dim) * 10

    env = SpikeRateEnvironment(dim=dim)
    network = DummySpikingNetwork(output_dim=dim, output_rates=rates, dt=dt0)
    agent = Agent(
        neuron_model=network,
        reward_prediction_model=MovingAverageRewardPredictor(rate=0.0),
    )
    model = AgentEnvSystem(
        agent=agent,
        environment=env,
        agent_output_shape=dim,
    )

    args = {
        "network_output_fn": lambda t, agent_state, args: agent_state.network_state.S,
        "reward_fn": lambda t, environment_state, args: jnp.array([0.0]),
    }

    y0 = model.initial
    sol = solve_ODE(
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
    assert jnp.allclose(ys.environment_state[-1], rates, rtol=0.1)
