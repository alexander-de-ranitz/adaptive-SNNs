import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.networks.coupled import (
    CoupledNoiseEligibilityLIFNetwork,
    CoupledNoiseGatedLIFNetwork,
)
from adaptive_SNN.models.noise.poisson_jump import PoissonJumpProcess
from adaptive_SNN.models.reward import MovingAverageRewardModel
from adaptive_SNN.utils.config import SimulationConfig
from adaptive_SNN.utils.save_helper import save_part_of_state


def create_default_config_single_synapse_task(
    lr,
    noise_level,
    RPE_noise_rate,
    key=jr.PRNGKey(0),
    model_cls: CoupledNoiseGatedLIFNetwork
    | CoupledNoiseEligibilityLIFNetwork = CoupledNoiseGatedLIFNetwork,
) -> SimulationConfig:
    t0 = 0
    t1 = 100
    dt = 1e-4
    N_neurons = 3
    N_inputs = 3
    min_noise_std = 1e-9

    rates = jnp.array(
        [5000, 1250, 50]
    )  # High frequency background input and one moderate frequency input
    initial_weights = jnp.tile(
        jnp.array([-jnp.inf] * N_neurons + [1.1, 11, 0.0]), (N_neurons, 1)
    )
    initial_weights = initial_weights.at[0, -1].set(10.0)

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
        teacher_state = args["env_state"][0]
        student_state = args["env_state"][1]
        direction = jnp.where(teacher_state > student_state, 1.0, -1.0)
        direction = jnp.where(
            jnp.allclose(teacher_state, student_state), 0.0, direction
        )
        spike_diff = (
            x.noisy_network.network_state.S[1] - x.noisy_network.network_state.S[2]
        )
        RPE_update = direction * spike_diff
        return RPE_update.reshape((1,))

    cfg = SimulationConfig(
        base_network_cls=model_cls,
        base_network_kwargs={
            "weight_coupling_indices": (jnp.array([2]), jnp.array([1])),
            "noise_coupling_indices": (jnp.array([0]), jnp.array([1])),
        },
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        balance=0.0,
        input_types=jnp.array([1, 0, 1]),
        t0=t0,
        t1=t1,
        dt=dt,
        initial_weight_matrix=initial_weights,
        lr=jnp.zeros_like(initial_weights).at[1, -1].set(lr),
        noise_level=jnp.array([0.0, noise_level, 0.0]),
        min_noise_std=min_noise_std,
        warmup_time=0,
        key=key,
        save_at=save_at,
        save_file=f"results/MWE/simulation_results_{model_cls.__name__}_lr{lr:.2f}_nl{noise_level:.2f}_rnl{RPE_noise_rate:.2f}.npz",
        network_output_fn=network_output_fn,
        input_spike_fn=input_spike_fn,
        reward_fn=lambda t, x, args: jnp.array([0.0]),
        environment_model=SpikeRateEnvironment,
        environment_kwargs={"rate": 0.05, "dim": N_neurons},
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
        args={
            "use_noise": jnp.array([False, True, False]),
            "RPE_fn": RPE_fn,
        },
    )
    return cfg
