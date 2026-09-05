import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_snn.models import AgentEnvSystem, SystemState
from adaptive_snn.models.environments.base import (
    AbstractEnvironment,
    AbstractEnvironmentState,
)
from adaptive_snn.models.networks import Agent, AgentState
from adaptive_snn.models.networks.base import AbstractNeuronModel
from adaptive_snn.models.reward_prediction.base import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_snn.solver import solve_ODE
from adaptive_snn.utils.operators import DefaultIfNone, ElementWiseMul


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

    def pre_step_update(self, t, x, args, input_spikes=None):
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

    def update(self, t, x, args):
        return DummyNetworkState(value=x.value + args["net_update_add"])

    def reset(self, t, x, args):
        return x

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

    def pre_step_update(
        self, t, x, args, reward, network_state, input_spikes, env_state
    ):
        return RewardPrediction(value=args["predicted_reward_value"])

    def drift(self, t, x, args, reward, RPE):
        return RewardPrediction(value=jnp.zeros_like(x.value))

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


def test_agent_env_system_drift_uses_connections():
    model = build_dummy_system()
    args = {
        "net_drift_rpe_scale": jnp.array([5.0]),
        "env_drift_scale": jnp.array([3.0]),
        "reward_fn": lambda t, system_state, a: jnp.zeros((1,)),
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
        drift.agent_state.reward_predictor_state.value, jnp.array([0.0])
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
        "reward_fn": lambda t, system_state, a: jnp.zeros((1,)),
        "RPE_fn": lambda t, x, args, reward: jnp.zeros((1,)),
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
        reward_signal=jnp.array([0.0]),
    )

    updated = model.update(0.0, x, args)

    assert jnp.allclose(updated.environment_state.value, jnp.array([12.0]))
    assert jnp.allclose(updated.agent_state.network_state.value, jnp.array([1.5]))
    assert jnp.allclose(updated.agent_output, jnp.array([3.0]))
    assert jnp.allclose(updated.reward_signal, jnp.array([0.0]))


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
        "network_output_fn": lambda t,
        agent_state,
        a,
        env_state: agent_state.network_state.value * 2.0,
        "reward_fn": lambda t, system_state, a: system_state.environment_state.value
        + system_state.agent_output,
        "RPE_fn": lambda t, x, args, reward: reward - x.reward_predictor_state.value,
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

    # Each step: reward is computed from previous agent_output, then output is computed from
    # current network. pre_step adds 0.5 to network and 1.0 to env. Euler adds dt * state
    # (net/env drift = current value). update adds 0.2 to network and 3 * agent_output + 0.1 to env.
    #
    # Step 1: output=1*2=2, net_pre=1.5, env_pre=3, net_euler=1.65, env_euler=3.3, net_upd=1.85, env_upd=9.4, reward=9.4+2=11.4
    # Step 2: output=1.85*2=3.7, net_pre=2.35, env_pre=10.4, net_euler=2.585, env_euler=11.44, net_upd=2.785, env_upd=22.64, reward=22.64+3.7=26.34
    assert jnp.allclose(values, jnp.array([[1.0], [1.85], [2.785]]))
    assert jnp.allclose(env_values, jnp.array([[2.0], [9.4], [22.64]]))
    assert jnp.allclose(outputs, jnp.array([[0.0], [2.0], [3.7]]))
    assert jnp.allclose(rewards, jnp.array([[0.0], [11.4], [26.34]]))


def test_env_warmup_disables_rpe():
    model = build_dummy_system()
    args = {
        "network_output_fn": lambda t,
        agent_state,
        a,
        env_state: agent_state.network_state.value * 2.0,
        "reward_fn": lambda t, system_state, a: system_state.environment_state.value
        + system_state.agent_output,
        "RPE_fn": lambda t, x, args, reward: reward - x.reward_predictor_state.value,
        "env_warmup_fn": lambda t, x, args: True,  # always in warmup
        "net_pre_step_add": jnp.array([0.0]),
        "env_pre_step_add": jnp.array([0.0]),
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

    # RPE should be zero even though reward - predicted_reward would be non-zero
    assert jnp.allclose(x1.agent_state.RPE, jnp.array([0.0]))


def _minimal_sim_config():
    """Config used for testing that just runs 10 steps of a small network."""
    from adaptive_snn.models.environments import SpikeRateEnvironment
    from adaptive_snn.models.reward_prediction import MovingAverageRewardPredictor
    from adaptive_snn.utils.config import SimulationConfig

    N_neurons, N_inputs = 2, 1
    return SimulationConfig(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        t0=0.0,
        t1=1e-3,
        dt=1e-4,
        actor_warmup_time=0.0,
        min_noise_std=1e-9,
        network_output_fn=(
            lambda t, agent_state, args, env_state: agent_state.network_state.S
        ),
        network_output_shape=(N_neurons,),
        input_spike_fn=lambda t, x, args: jnp.zeros((N_neurons, N_inputs)),
        reward_fn=lambda t, x, args: jnp.array([0.0]),
        RPE_fn=lambda t, x, args, reward: reward,
        environment_model=SpikeRateEnvironment,
        environment_kwargs={"rate": 1, "dim": N_neurons},
        reward_prediction_model=MovingAverageRewardPredictor,
        reward_predictor_kwargs={"rate": 0.0, "dim": 1},
        use_noise=jnp.array([True] * N_neurons),
    )


def test_named_result_save(tmp_path):
    import numpy as np
    from diffrax import SaveAt

    from adaptive_snn.utils.runner import run_simulation
    from adaptive_snn.utils.save_helper import load_named_result

    class _Summary(eqx.Module):
        mean_V: jnp.ndarray

    def save_fn(t, x, args):
        return _Summary(mean_V=jnp.mean(x.agent_state.network_state.V))

    cfg = _minimal_sim_config()
    cfg.t1 = cfg.t0 + 5 * cfg.dt
    cfg.save_at = SaveAt(ts=jnp.linspace(cfg.t0, cfg.t1, 3), fn=save_fn)
    cfg.save_file = str(tmp_path / "out")

    run_simulation(cfg, save_results=True, overwrite=True, named_result=True)

    out = load_named_result(str(tmp_path / "out.npz"))
    assert set(out) == {"mean_V", "ts"}
    assert out["mean_V"].shape == (3,)
    assert np.asarray(out["ts"]).shape == (3,)


def test_named_result_save_with_final_state(tmp_path):
    import jax
    from diffrax import SaveAt

    from adaptive_snn.utils.runner import load_final_state, run_simulation
    from adaptive_snn.utils.save_helper import load_named_result

    class _Summary(eqx.Module):
        mean_V: jnp.ndarray

    def save_fn(t, x, args):
        return _Summary(mean_V=jnp.mean(x.agent_state.network_state.V))

    def make_cfg(name):
        cfg = _minimal_sim_config()
        cfg.t1 = cfg.t0 + 5 * cfg.dt
        cfg.save_at = SaveAt(ts=jnp.linspace(cfg.t0, cfg.t1, 3), fn=save_fn)
        cfg.save_file = str(tmp_path / name)
        return cfg

    # named_result + return_final_state: summary and final state share one file.
    run_simulation(
        make_cfg("with_fs"),
        save_results=True,
        overwrite=True,
        named_result=True,
        return_final_state=True,
    )
    fs_path = str(tmp_path / "with_fs.npz")
    out = load_named_result(fs_path)
    # Only the summary half is surfaced; final_state* stay out of load_named_result.
    assert set(out) == {"mean_V", "ts"}
    assert out["mean_V"].shape == (3,)
    assert out["ts"].shape == (3,)
    # The final-state pytree is reachable only via load_final_state.
    final_state = load_final_state(fs_path)
    assert final_state is not None
    assert jax.tree_util.tree_leaves(final_state)

    # Plain named_result file (no return_final_state) -> load_final_state is None.
    run_simulation(
        make_cfg("plain"),
        save_results=True,
        overwrite=True,
        named_result=True,
    )
    assert load_final_state(str(tmp_path / "plain.npz")) is None


def test_named_result_save_batched(tmp_path):
    from diffrax import SaveAt

    from adaptive_snn.utils.runner import load_final_state, run_batched_simulation
    from adaptive_snn.utils.save_helper import load_named_result

    class _Summary(eqx.Module):
        mean_V: jnp.ndarray

    def save_fn(t, x, args):
        return _Summary(mean_V=jnp.mean(x.agent_state.network_state.V))

    def make_cfg(name):
        cfg = _minimal_sim_config()
        cfg.t1 = cfg.t0 + 5 * cfg.dt
        cfg.save_at = SaveAt(ts=jnp.linspace(cfg.t0, cfg.t1, 3), fn=save_fn)
        cfg.save_file = str(tmp_path / name)
        return cfg

    cfg0 = make_cfg("batched0")
    cfg1 = make_cfg("batched1")

    run_batched_simulation(
        [cfg0, cfg1],
        save_results=True,
        overwrite=True,
        named_result=True,
        return_final_state=True,
    )

    for name in ("batched0", "batched1"):
        path = str(tmp_path / f"{name}.npz")
        out = load_named_result(path)
        assert set(out) == {"mean_V", "ts"}
        assert out["mean_V"].shape == (3,)
        assert load_final_state(path) is not None
