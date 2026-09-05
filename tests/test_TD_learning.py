import diffrax as dfx
import equinox as eqx
from helpers import DummyModel, RotatingDummyEnv
from jax import numpy as jnp
from jax import random as jr

from adaptive_snn.models.agent_env_system import AgentEnvSystem, SystemState
from adaptive_snn.models.networks.agent import Agent
from adaptive_snn.models.reward_prediction.critic import (
    CriticPrediction,
    LinearReadoutCritic,
)
from adaptive_snn.simulation_configs.pendulum_AC_config import create_pendulum_AC_config
from adaptive_snn.solver import solve_ODE
from adaptive_snn.utils.runner import setup_simulation


def test_TD_error():
    # Use the config to generate the model and args for the simulation
    cfg = create_pendulum_AC_config(key=jr.PRNGKey(0), N_neurons=10)
    model, args, key = setup_simulation(cfg)
    model: AgentEnvSystem = model  # Type hint for clarity
    initial_state: SystemState = model.initial

    # Turn off warmup for these tests
    args["env_warmup_fn"] = lambda t, x, args: False

    # Manually set relevant state variables and functions for testing
    #  agent_output is set to 0.15, reward is -0.2 + agent_output
    #  predicted value is 2.0 (which will turn into previous_value during the pre_step_update)
    args["reward_fn"] = lambda t, x, args: jnp.array([-0.2]) - x.agent_output
    input_spikes = jr.normal(key, (model.agent.network.N_inputs,)) * 0.1
    args["input_spike_fn"] = lambda t, x, args: input_spikes
    args["network_output_fn"] = lambda t, agent_state, args, env_state: jnp.array(
        [0.15]
    )

    weights = (
        jnp.mod(jnp.arange(model.agent.network.N_inputs + 1), jnp.array([3.0])) - 1.5
    )
    features = jnp.mod(jnp.arange(model.agent.network.N_inputs), jnp.array([4.0])) - 2.0
    initial_state = eqx.tree_at(
        lambda s: s.agent_state.reward_predictor_state.weights, initial_state, weights
    )
    initial_state = eqx.tree_at(
        lambda s: s.agent_state.reward_predictor_state.features,
        initial_state,
        features,
    )

    new_state = model.pre_step_update(0.0, initial_state, args)
    new_state = model.update(0.0, new_state, args)

    expected_previous_value = weights @ jnp.concatenate([features, jnp.array([1.0])])
    expected_new_value = weights @ jnp.concatenate(
        [features + input_spikes, jnp.array([1.0])]
    )

    # Check that the TD error is computed correctly
    # this is the Euler discretization of the TD error: RPE = reward - predicted_reward + gamma * new_value - previous_value
    expected_RPE = (
        (-0.2 - 0.15) + args["gamma"] * expected_new_value - expected_previous_value
    )
    assert jnp.allclose(
        new_state.agent_state.RPE, jnp.array([expected_RPE]), atol=1e-10, rtol=1e-10
    )


def test_critic_drift():
    critic = LinearReadoutCritic(input_dim=5)
    state = CriticPrediction(
        value=jnp.array([1.0]),
        previous_value=jnp.array([0.5]),
        weights=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        features=jnp.zeros(4),
        features_prev=jnp.zeros(4),
        features_prev_prev=jnp.array([3.0, 4.0, 5.0, 6.0]),
    )
    RPE = jnp.array([0.25])
    learning_rate = 0.01

    # Compute the expected weight update
    expected_weight_update = (
        learning_rate
        * RPE
        * jnp.concatenate([state.features_prev_prev, jnp.array([1.0])])
    )
    drift = critic.drift(
        0.0,
        state,
        {"get_critic_lr": lambda t, x, args: learning_rate},
        reward=0.0,
        RPE=RPE,
    )

    assert jnp.allclose(drift.weights, expected_weight_update)


def test_critic_convergence_static_target():
    critic = LinearReadoutCritic(input_dim=5)
    state = CriticPrediction(
        value=jnp.array([0.0]),
        previous_value=jnp.array([0.0]),
        weights=jnp.zeros(5),
        features=jnp.array([1.0, 2.0, 3.0, 4.0]),
        features_prev=jnp.array([0.0, 0.0, 0.0, 0.0]),
        features_prev_prev=jnp.array([0.0, 0.0, 0.0, 0.0]),
    )
    learning_rate = 0.01
    args = {
        "get_critic_lr": lambda t, x, args: learning_rate,
        "critic_input_fn": lambda t, x, args, input_spikes, env_state: input_spikes,
    }
    true_weights = jnp.array([0.5, -0.3, 0.2, 0.1, 0.4])

    # Simulate multiple updates to see if the weights converge
    for t in range(100):
        # Update the critic
        # no drift, so weights remain constant
        state = critic.pre_step_update(
            0.0,
            state,
            args,
            reward=0.0,
            network_state=None,
            input_spikes=jnp.zeros((4,)),
            env_state=None,
        )
        state = critic.update(0.0, state, args)

        # Compute error
        # Note: this is not doing TD learning, just gradient descent on the error
        true_value = true_weights @ jnp.concatenate([state.features, jnp.array([1.0])])
        RPE = true_value - state.value
        drift = critic.drift(0.0, state, args, reward=0.0, RPE=RPE)
        state = eqx.tree_at(lambda s: s.weights, state, state.weights + drift.weights)
        print(
            f"Step {t}: Weights = {state.weights}, RPE = {RPE}, True Value = {true_value}, Predicted Value = {state.value}"
        )

    # After enough updates, the predicted value should match the true value
    # since the system is underdetermined, the weights will not necessarily converge to the true_weights
    assert jnp.allclose(true_value, state.value, atol=0.01)


def test_critic_convergence_full():
    cfg = create_pendulum_AC_config(key=jr.PRNGKey(0), N_neurons=10)
    _, _, _ = setup_simulation(cfg)

    critic = LinearReadoutCritic(input_dim=2)
    agent = Agent(neuron_model=DummyModel(), reward_prediction_model=critic)
    model = AgentEnvSystem(
        agent=agent, environment=RotatingDummyEnv(), agent_output_shape=(1,)
    )
    learning_rate = 1.0
    true_weights = jnp.array([0.5, -0.3, 0.1])
    discount_rate = 0.99

    def reward_fn(t, x, args):
        s_t_prev = x.agent_state.reward_predictor_state.features_prev
        s_t = x.agent_state.reward_predictor_state.features
        V_star_t_prev = true_weights @ jnp.concatenate([s_t_prev, jnp.array([1.0])])
        V_star_t = true_weights @ jnp.concatenate([s_t, jnp.array([1.0])])
        true_reward = V_star_t_prev - discount_rate * V_star_t
        return jnp.atleast_1d(true_reward)

    args = {
        "get_critic_lr": lambda t, x, args: learning_rate,
        "critic_input_fn": lambda t, x, args, input_spikes, env_state: input_spikes,
        "env_warmup_fn": lambda t, x, args: False,
        "episode_end_fn": lambda t, x, args: False,
        "network_output_fn": lambda t, agent_state, args, env_state: jnp.zeros((1,)),
        "input_spike_fn": lambda t,
        x,
        args: -x.agent_state.reward_predictor_state.features + x.environment_state,
        "reward_fn": lambda t, x, args: reward_fn(t, x, args),
        "RPE_fn": lambda t, x, args, reward: reward
        + discount_rate * x.reward_predictor_state.value
        - x.reward_predictor_state.previous_value,
    }

    key = jr.PRNGKey(5555)
    initial = model.initial

    sol = solve_ODE(
        model,
        solver=dfx.Euler(),
        t0=0.0,
        t1=1000,
        dt0=1e-3,
        y0=initial,
        args=args,
        save_at=dfx.SaveAt(t1=True),
        key=key,
    )
    final_state: SystemState = sol.ys
    final_weights = final_state.agent_state.reward_predictor_state.weights
    assert jnp.allclose(final_weights, true_weights, atol=0.1)
