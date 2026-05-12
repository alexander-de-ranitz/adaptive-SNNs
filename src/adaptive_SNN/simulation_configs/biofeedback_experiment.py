import jax

from adaptive_SNN.models.RPE import BiphasicRPEModel

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.noise.poisson_jump import PoissonJumpProcess
from adaptive_SNN.models.reward import MovingAverageRewardModel
from adaptive_SNN.utils.config import SimulationConfig


def create_config(model_cls, N_neurons=100, key=jr.PRNGKey(0)) -> SimulationConfig:
    t0 = 0
    t1 = 0.5
    dt = 1e-4
    lr = 0.0
    noise_level = 0.0
    RPE_noise_rate = 0.0
    N_inputs = 1
    min_noise_std = 1e-9
    balance = 0.5

    input_rate = jnp.array([1500.0])  # High frequency background input

    key, spike_key, reward_noise_key = jr.split(key, 3)

    # Output is based on a single neuron
    network_output_fn = lambda t, agent_state, args: jnp.squeeze(
        agent_state.noisy_network.network_state.S[0]
    )

    def input_spike_fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint((t - t0) / dt), dtype=jnp.int64)
        return jr.poisson(
            jr.fold_in(spike_key, step_idx),
            input_rate * dt,
            shape=(N_neurons, N_inputs),
        )

    def save(t, x: SystemState, args):
        """Save S, RPE, weights to target neuron, and mean weight at each time step"""
        return (
            x.agent_state.noisy_network.network_state.S.astype(jnp.bool),
            x.agent_state.RPE.RPE.astype(jnp.float32),
            x.agent_state.noisy_network.network_state.W[0].astype(jnp.float32),
            jnp.nanmean(
                jnp.where(
                    jnp.isfinite(x.agent_state.noisy_network.network_state.W),
                    x.agent_state.noisy_network.network_state.W,
                    jnp.nan,
                )
            ),
        )

    save_at = SaveAt(steps=True, fn=save)

    # RPE = 1 when the target neuron spikes, 0 otherwise.
    def RPE_fn(t, agent_state, args):
        return agent_state.noisy_network.network_state.S[0]

    cfg = SimulationConfig(
        base_network_cls=model_cls,
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        balance=balance,
        connection_prob_E=0.1,
        connection_prob_I=0.2,
        initial_input_weight=1.0,
        rec_weight_std=0.2,
        initial_rec_weight=1.0,
        input_types=jnp.array([1]),
        fully_connected_input=True,
        t0=t0,
        t1=t1,
        dt=dt,
        lr=lr * jnp.zeros((N_neurons, N_neurons + N_inputs)),
        mean_synaptic_delay=1.5e-3,
        noise_level=jnp.array([noise_level] * N_neurons),
        min_noise_std=min_noise_std,
        warmup_time=0,
        key=key,
        save_at=save_at,
        save_file="results/biofeedback/biofeedback_experiment.npz",
        network_output_fn=network_output_fn,
        input_spike_fn=input_spike_fn,
        reward_fn=lambda t, x, args: jnp.array([0.0]),
        environment_model=SpikeRateEnvironment,
        environment_kwargs={"rate": 10, "dim": N_neurons},
        reward_model=MovingAverageRewardModel,
        reward_kwargs={"reward_rate": 0.0, "dim": 1},
        reward_noise_model=PoissonJumpProcess,
        reward_noise_kwargs={
            "jump_rate": RPE_noise_rate,
            "jump_mean": 0.0,
            "jump_std": 1.0,
            "dim": 1,
            "dt": dt,
            "tau": 0.05,
            "key": reward_noise_key,
        },
        RPE_model=BiphasicRPEModel,
        RPE_model_kwargs={"time_constants": jnp.array([0.1, 1.0]), "scale": 1.0},
        args={
            "use_noise": jnp.array([True] * N_neurons),
            "RPE_fn": RPE_fn,
        },
    )
    return cfg
