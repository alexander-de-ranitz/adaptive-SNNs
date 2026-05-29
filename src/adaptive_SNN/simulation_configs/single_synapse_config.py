import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability

import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.environments import SingleSynapseLearningEnv
from adaptive_SNN.models.networks import EligibilityLIFNetwork, GatedLIFNetwork
from adaptive_SNN.models.noise import PoissonJumpProcess
from adaptive_SNN.models.reward_prediction import MovingAverageRewardPredictor
from adaptive_SNN.utils.config import SimulationConfig
from adaptive_SNN.utils.save_helper import save_part_of_state


def create_single_synapse_learning_config(
    lr=0.0,
    noise_level=0.0,
    reward_noise_jump_rate=0.0,
    initial_synapse_weight=0.0,
    key=jr.PRNGKey(0),
    network_cls: GatedLIFNetwork | EligibilityLIFNetwork = GatedLIFNetwork,
) -> SimulationConfig:
    t0 = 0
    t1 = 100
    dt = 1e-4
    N_neurons = 2
    N_inputs = 3
    min_noise_std = 1e-9

    rates = jnp.array(
        [5000, 1250, 10]
    )  # High frequency background input and one moderate frequency input
    initial_weights = jnp.tile(
        jnp.array([jnp.nan] * N_neurons + [1.1, 11, initial_synapse_weight]),
        (N_neurons, 1),
    )

    key, spike_key = jr.split(key, 2)

    network_output_fn = lambda t, agent_state, args: (
        agent_state.network_state.S[0] - agent_state.network_state.S[1]
    ).reshape((1,))

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
            perturbations=True,
        ),
    )

    cfg = SimulationConfig(
        network_cls=network_cls,
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        balance=0.0,
        input_types=jnp.array([1, 0, 1]),
        t0=t0,
        t1=t1,
        dt=dt,
        initial_weight_matrix=initial_weights,
        mean_synaptic_delay=0.0,
        lr=jnp.zeros_like(initial_weights),
        noise_level=jnp.array([noise_level] * N_neurons),
        min_noise_std=min_noise_std,
        warmup_time=0,
        key=key,
        save_at=save_at,
        save_file=f"results/ssl_{network_cls.__name__}_lr{lr:.2f}_nl{noise_level:.2f}_rnl{reward_noise_jump_rate:.2f}.npz",
        network_output_fn=network_output_fn,
        input_spike_fn=input_spike_fn,
        reward_fn=lambda t, x, args: x.environment_state.reward
        + x.environment_state.reward_noise,  # The reward is determined by the environment, so we just return it here.
        environment_model=SingleSynapseLearningEnv,
        environment_kwargs={
            "tau_reward": 0.1,
            "reward_dim": 1,
            "reward_noise_process": PoissonJumpProcess(
                jump_rate=reward_noise_jump_rate, jump_std=1.0, jump_mean=0.0
            ),
        },
        reward_prediction_model=MovingAverageRewardPredictor,  # Not used here, but required by the runner. Rate = 0 -> always predicts 0 reward.
        reward_predictor_kwargs={"rate": 0.0, "dim": 1},
        args={
            "use_noise": jnp.array([True, False]),
        },
    )
    return cfg
