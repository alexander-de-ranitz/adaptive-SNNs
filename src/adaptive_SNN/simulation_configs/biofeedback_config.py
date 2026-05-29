import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.agent_env_system import SystemState
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.reward_prediction import MovingAverageRewardPredictor
from adaptive_SNN.utils.config import SimulationConfig

# TODO: fix this config if needed
# It relied on the old RPEModel implementation, which has been removed
# The appropriate reward should be computed inside the environment
# However, we might not use this script for the thesis, so it is not a priority to fix it right now.


def create_config(model_cls, N_neurons=100, key=jr.PRNGKey(0)) -> SimulationConfig:
    t0 = 0
    t1 = 0.5
    dt = 1e-4
    lr = 0.0
    noise_level = 0.0
    N_inputs = 1
    min_noise_std = 1e-9
    balance = 0.5

    input_rate = jnp.array([1500.0])  # High frequency background input

    key, spike_key, reward_noise_key = jr.split(key, 3)

    # Output is the network's spikes
    network_output_fn = lambda t, agent_state, args: agent_state.network_state.S

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
            x.agent_state.network_state.S.astype(jnp.bool),
            x.agent_state.RPE.astype(jnp.float32),
            x.agent_state.network_state.W[0].astype(jnp.float32),
            jnp.nanmean(x.agent_state.network_state.W),
            x.environment_state.astype(jnp.float32),
        )

    save_at = SaveAt(steps=True, fn=save)

    cfg = SimulationConfig(
        network_cls=model_cls,
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
        network_output_shape=(N_neurons,),
        input_spike_fn=input_spike_fn,
        reward_fn=lambda t, x, args: jnp.array([0.0]),
        environment_model=SpikeRateEnvironment,
        environment_kwargs={"rate": 1, "dim": N_neurons},
        reward_prediction_model=MovingAverageRewardPredictor,
        reward_predictor_kwargs={"rate": 0.0, "dim": 1},
        args={
            "use_noise": jnp.array([True] * N_neurons),
        },
    )
    return cfg
