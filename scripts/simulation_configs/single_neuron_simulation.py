import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.networks import LIFNetwork
from adaptive_SNN.models.noise.poisson_jump import PoissonJumpProcess
from adaptive_SNN.models.reward import MovingAverageRewardModel
from adaptive_SNN.utils.config import SimulationConfig
from adaptive_SNN.utils.save_helper import save_part_of_state


def create_single_neuron_config_extra_synapse(
    N_neurons=1, key=jr.PRNGKey(0)
) -> SimulationConfig:
    t0 = 0
    t1 = 10
    dt = 1e-4
    lr = 0.0
    noise_level = 0.0
    RPE_noise_rate = 0.0
    N_inputs = 3
    min_noise_std = 1e-9

    rates = jnp.array(
        [5000, 1250, 5]
    )  # High frequency background input and one moderate frequency input
    initial_weights = jnp.tile(
        jnp.array([-jnp.inf] * N_neurons + [1.1, 11, 5.0]), (N_neurons, 1)
    )

    key, spike_key, reward_noise_key = jr.split(key, 3)

    network_output_fn = lambda t, agent_state, args: jnp.squeeze(
        agent_state.noisy_network.network_state.S
    )

    def input_spike_fn(t, x, args):
        step_idx = jnp.asarray(jnp.rint((t - t0) / dt), dtype=jnp.int64)
        spikes_1d = jr.poisson(
            jr.fold_in(spike_key, step_idx),
            rates * dt,
            shape=(1, N_inputs),
        )
        return jnp.tile(spikes_1d, (N_neurons, 1))

    save_at = SaveAt(
        ts=jnp.linspace(t0, t1, 50000),
        fn=lambda t, x, args: save_part_of_state(
            x,
            environment_state=True,
            W=True,
            G=True,
            RPE=True,
            reward_noise=True,
            eligibility=True,
            V=True,
            noise_state=True,
        ),
    )

    def RPE_fn(t, x, args):
        return jnp.zeros((1,))

    model_cls = LIFNetwork
    cfg = SimulationConfig(
        base_network_cls=model_cls,
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        balance=0.0,
        input_types=jnp.array([1, 0, 1]),
        t0=t0,
        t1=t1,
        dt=dt,
        initial_weight_matrix=initial_weights,
        lr=jnp.zeros_like(initial_weights),
        noise_level=jnp.array([noise_level] * N_neurons),
        min_noise_std=min_noise_std,
        warmup_time=0,
        key=key,
        save_at=save_at,
        save_file=f"results/MWE/simulation_results_{model_cls.__name__}_lr{lr:.2f}_nl{noise_level:.2f}_rnl{RPE_noise_rate:.2f}.npz",
        network_output_fn=network_output_fn,
        input_spike_fn=input_spike_fn,
        reward_fn=lambda t, x, args: jnp.array([0.0]),
        environment_model=SpikeRateEnvironment,
        environment_kwargs={"rate": 1, "dim": N_neurons},
        reward_model=MovingAverageRewardModel,
        reward_kwargs={"reward_rate": 0.0, "dim": 1},
        reward_noise_model=PoissonJumpProcess,
        reward_noise_kwargs={
            "jump_rate": RPE_noise_rate,
            "jump_mean": 0.0,
            "jump_std": 0.0,
            "dim": 1,
            "dt": dt,
            "tau": 0.05,
            "key": reward_noise_key,
        },
        args={
            "use_noise": jnp.array([True] * N_neurons),
            "RPE_fn": RPE_fn,
        },
    )
    return cfg
