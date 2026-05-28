import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models import AgentEnvSystem, SystemState
from adaptive_SNN.models.environments.base import (
    AbstractEnvironment,
    AbstractEnvironmentState,
)
from adaptive_SNN.models.networks import Agent, AgentState
from adaptive_SNN.models.networks.base import AbstractNeuronModel
from adaptive_SNN.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_SNN.solver import solve_ODE
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul


class DummyNetworkState(eqx.Module):
    value: Array


class DummyEnvironmentState(AbstractEnvironmentState):
    value: Array


class DummyNetwork(AbstractNeuronModel):
    initial_value: Array
    N_neurons: int = 1
    N_inputs: int = 0

    def __init__(self, initial_value: float):
        self.initial_value = jnp.array([initial_value])

    @property
    def initial(self):
        return DummyNetworkState(value=self.initial_value)

    def pre_step_update(self, t, x, args):
        return DummyNetworkState(value=x.value + args["net_pre_step_add"])

    def drift(self, t, x, args, RPE=None):
        rpe = jnp.zeros_like(x.value) if RPE is None else RPE
        return DummyNetworkState(value=x.value + args["net_drift_rpe_scale"] * rpe)

    def diffusion(self, t, x, args):
        zeros = jnp.zeros_like(x.value)
        return DummyNetworkState(
            value=DefaultIfNone(
                default=zeros,
                else_do=ElementWiseMul(zeros),
            )
        )

    @property
    def noise_shape(self):
        return DummyNetworkState(value=None)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args, input_spikes):
        return DummyNetworkState(value=x.value + args["net_update_add"])


class DummyRewardPredictor(AbstractRewardPredictor):
    initial_value: Array

    def __init__(self, initial_value: float = 0.0):
        self.initial_value = jnp.array([initial_value])

    @property
    def initial(self):
        return RewardPrediction(value=self.initial_value)

    @property
    def noise_shape(self):
        return RewardPrediction(value=None)

    def pre_step_update(self, t, x, args, reward, network_state):
        return RewardPrediction(value=args["predicted_reward_value"])

    def drift(self, t, x, args, reward, network_state):
        return RewardPrediction(value=reward + network_state.value)

    def diffusion(self, t, x, args):
        return RewardPrediction(
            value=DefaultIfNone(
                default=jnp.zeros_like(x.value),
                else_do=ElementWiseMul(jnp.zeros_like(x.value)),
            )
        )

    def terms(self, key):
        raise NotImplementedError("DummyRewardPredictor.terms is not used in tests.")

    def update(self, t, x, args):
        return x


class DummyEnvironment(AbstractEnvironment):
    initial_value: Array

    def __init__(self, initial_value: float):
        self.initial_value = jnp.array([initial_value])

    @property
    def initial(self):
        return DummyEnvironmentState(value=self.initial_value)

    @property
    def noise_shape(self):
        return DummyEnvironmentState(value=None)

    def pre_step_update(self, t, x, args):
        return DummyEnvironmentState(value=x.value + args["env_pre_step_add"])

    def drift(self, t, x, args, env_input):
        return DummyEnvironmentState(
            value=x.value + env_input * args["env_drift_scale"]
        )

    def diffusion(self, t, x, args):
        return DummyEnvironmentState(
            value=DefaultIfNone(
                default=jnp.zeros_like(x.value),
                else_do=ElementWiseMul(jnp.zeros_like(x.value)),
            )
        )

    def terms(self, key):
        raise NotImplementedError("DummyEnvironment.terms is not used in tests.")

    def update(self, t, x, args, env_input):
        return DummyEnvironmentState(
            value=x.value
            + env_input * args["env_update_scale"]
            + args["env_update_bias"]
        )


def build_dummy_system():
    network = DummyNetwork(initial_value=1.0)
    reward_predictor = DummyRewardPredictor()
    agent = Agent(neuron_model=network, reward_prediction_model=reward_predictor)
    environment = DummyEnvironment(initial_value=2.0)
    return AgentEnvSystem(
        agent=agent,
        environment=environment,
        agent_output_shape=(1,),
    )


def test_agent_env_system_pre_step_update_order():
    model = build_dummy_system()
    args = {
        "network_output_fn": lambda t, agent_state, a: agent_state.network_state.value
        * 2.0,
        "reward_fn": lambda t, system_state, a: system_state.environment_state.value
        + system_state.agent_output,
        "net_pre_step_add": jnp.array([0.5]),
        "env_pre_step_add": jnp.array([5.0]),
        "predicted_reward_value": jnp.array([0.25]),
        "net_drift_rpe_scale": jnp.array([0.0]),
        "net_update_add": jnp.array([0.0]),
        "env_drift_scale": jnp.array([0.0]),
        "env_update_scale": jnp.array([0.0]),
        "env_update_bias": jnp.array([0.0]),
        "input_spike_fn": lambda t, x, args: None,
    }

    x0 = model.initial
    x1 = model.pre_step_update(0.0, x0, args)

    assert jnp.allclose(x1.agent_output, jnp.array([2.0]))
    assert jnp.allclose(x1.reward_signal, jnp.array([4.0]))
    assert jnp.allclose(x1.agent_state.RPE, jnp.array([3.75]))
    assert jnp.allclose(x1.agent_state.network_state.value, jnp.array([1.5]))
    assert jnp.allclose(x1.environment_state.value, jnp.array([7.0]))
    assert jnp.allclose(x1.agent_state.reward_predictor_state.value, jnp.array([0.25]))


def test_agent_env_system_drift_uses_connections():
    model = build_dummy_system()
    args = {
        "net_drift_rpe_scale": jnp.array([5.0]),
        "env_drift_scale": jnp.array([3.0]),
    }

    agent_state = AgentState(
        network_state=DummyNetworkState(value=jnp.array([1.0])),
        reward_predictor_state=RewardPrediction(value=jnp.array([0.0])),
        RPE=jnp.array([2.0]),
    )
    env_state = DummyEnvironmentState(value=jnp.array([3.0]))
    x = SystemState(
        agent_state=agent_state,
        environment_state=env_state,
        agent_output=jnp.array([4.0]),
        reward_signal=jnp.array([5.0]),
    )

    drift = model.drift(0.0, x, args)

    assert jnp.allclose(drift.environment_state.value, jnp.array([15.0]))
    assert jnp.allclose(drift.agent_state.network_state.value, jnp.array([11.0]))
    assert jnp.allclose(
        drift.agent_state.reward_predictor_state.value, jnp.array([6.0])
    )
    assert jnp.allclose(drift.agent_state.RPE, jnp.array([0.0]))
    assert jnp.allclose(drift.agent_output, jnp.array([0.0]))
    assert jnp.allclose(drift.reward_signal, jnp.array([0.0]))


def test_agent_env_system_update_uses_agent_output():
    model = build_dummy_system()
    args = {
        "net_update_add": jnp.array([0.5]),
        "env_update_scale": jnp.array([2.0]),
        "env_update_bias": jnp.array([1.0]),
        "input_spike_fn": lambda t, x, args: jnp.zeros(
            (model.agent.network.N_neurons, model.agent.network.N_inputs)
        ),
    }

    agent_state = AgentState(
        network_state=DummyNetworkState(value=jnp.array([1.0])),
        reward_predictor_state=RewardPrediction(value=jnp.array([0.0])),
        RPE=jnp.array([0.0]),
    )
    env_state = DummyEnvironmentState(value=jnp.array([5.0]))
    x = SystemState(
        agent_state=agent_state,
        environment_state=env_state,
        agent_output=jnp.array([3.0]),
        reward_signal=jnp.array([4.0]),
    )

    updated = model.update(0.0, x, args)

    assert jnp.allclose(updated.environment_state.value, jnp.array([12.0]))
    assert jnp.allclose(updated.agent_state.network_state.value, jnp.array([1.5]))
    assert jnp.allclose(updated.agent_output, jnp.array([3.0]))
    assert jnp.allclose(updated.reward_signal, jnp.array([4.0]))


def test_agent_env_system_noise_shape_structure():
    model = build_dummy_system()

    noise_shape = model.noise_shape

    assert noise_shape.agent_output is None
    assert noise_shape.reward_signal is None
    assert noise_shape.environment_state.value is None
    assert noise_shape.agent_state.network_state.value is None
    assert noise_shape.agent_state.reward_predictor_state.value is None
    assert noise_shape.agent_state.RPE is None


def test_solve_ode_runs_pre_step_and_update():
    model = build_dummy_system()

    args = {
        "network_output_fn": lambda t, agent_state, a: agent_state.network_state.value
        * 2.0,
        "reward_fn": lambda t, system_state, a: system_state.environment_state.value
        + system_state.agent_output,
        "net_pre_step_add": jnp.array([0.5]),
        "env_pre_step_add": jnp.array([1.0]),
        "predicted_reward_value": jnp.array([0.0]),
        "net_drift_rpe_scale": jnp.array([0.0]),
        "net_update_add": jnp.array([0.2]),
        "env_drift_scale": jnp.array([0.0]),
        "env_update_scale": jnp.array([3.0]),
        "env_update_bias": jnp.array([0.1]),
        "input_spike_fn": lambda t, x, args: None,
    }

    def save_fn(t, y, a):
        return (
            y.agent_state.network_state.value,
            y.environment_state.value,
            y.agent_output,
            y.reward_signal,
        )

    save_at = dfx.SaveAt(subs=dfx.SubSaveAt(steps=True, t0=True, t1=True, fn=save_fn))

    y0 = model.initial
    response = solve_ODE(
        model,
        dfx.Euler(),
        0.0,
        0.2,
        0.1,
        y0,
        save_at=save_at,
        args=args,
    )

    values, env_values, outputs, rewards = response.ys

    # Each step: pre_step adds 0.5 to network and 1.0 to env, then Euler adds dt * state
    # (drift is identity), then update adds 0.2 to network and 3 * agent_output + 0.1 to env.
    assert jnp.allclose(values, jnp.array([[1.0], [1.85], [2.785]]))
    assert jnp.allclose(env_values, jnp.array([[2.0], [9.4], [22.64]]))
    assert jnp.allclose(outputs, jnp.array([[0.0], [2.0], [3.7]]))
    assert jnp.allclose(rewards, jnp.array([[0.0], [4.0], [13.1]]))
