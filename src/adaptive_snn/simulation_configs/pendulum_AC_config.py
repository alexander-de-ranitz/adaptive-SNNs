import jax

jax.config.update("jax_enable_x64", True)
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_snn.models import SystemState
from adaptive_snn.models.environments import PendulumEnvironment
from adaptive_snn.models.networks import GatedLIFNetwork
from adaptive_snn.models.reward_prediction import LinearReadoutCritic
from adaptive_snn.utils.config import SimulationConfig
from adaptive_snn.utils.encoding import poisson_grid_encoding

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
TAU_DISCOUNT = 0.1  # Time constant of the TD discount: gamma = 1 - dt / TAU_DISCOUNT


def create_pendulum_AC_config(
    N_neurons=1000,
    model_cls=GatedLIFNetwork,
    encoding_grid_shape=(16, 16),
    key=jr.PRNGKey(0),
    **env_kwargs,
) -> SimulationConfig:
    t0 = 0
    t1 = 5
    dt = 1e-4
    lr = 0.0
    noise_scaling_factor = 0.0
    gamma = 1 - dt / TAU_DISCOUNT
    noise_floor = 5e-9
    balance = 0.01
    N_inputs = encoding_grid_shape[0] * encoding_grid_shape[1]
    total_input_rate = N_inputs * 60
    agent_output_scaling = (
        0.25  # Scale the output so that it doesn't produce excessively large torques
    )

    key, spike_key, env_key = jr.split(key, 3)
    env = PendulumEnvironment(key=env_key, **env_kwargs)

    def input_spike_fn(t, x: SystemState, args):
        step_idx = jnp.asarray(jnp.rint((t) / dt), dtype=jnp.int64)
        current_key = jr.fold_in(spike_key, step_idx)

        env_state = x.environment_state

        rates = poisson_grid_encoding(
            env_state[:2],  # Only encode time and ang. vel, not time
            grid_shape=encoding_grid_shape,
            total_rate=total_input_rate,
            grid_range=[
                (-env.max_allowed_angle, env.max_allowed_angle),
                (-env.max_allowed_angular_velocity, env.max_allowed_angular_velocity),
            ],
        ).flatten()

        # Generate the spikes of the encoding population — shape (N_inputs,).
        # This is broadcast to all neurons in the recurrent population
        encoding_spikes = jr.poisson(current_key, rates * dt)
        return encoding_spikes

    def critic_input_fn(t, x, args, input_spikes, env_state):
        return input_spikes

    # Define network output function
    output_pop = jnp.arange(
        jnp.max(jnp.array([jnp.round(N_neurons * 0.2).astype(int), 2]))
    )  # the first 20% (min 2) of neurons are the output population
    left = output_pop[
        : output_pop.size // 2
    ]  # first half codes for left torque, second half for right torque
    right = output_pop[output_pop.size // 2 :]
    network_output_fn = (
        lambda t, agent_state, args, env_state: (
            agent_state.network_state.filtered_spike_trains[left].mean()
            - agent_state.network_state.filtered_spike_trains[right].mean()
        ).reshape((1,))
        * agent_output_scaling
    )

    def reward_fn(t, x: SystemState, args):
        return env.reward_fn(t, x.environment_state, args, x.agent_output)

    def save(t, x: SystemState, args):
        return x.agent_state.network_state.S.astype(jnp.bool_)

    save_at = SaveAt(steps=True, fn=save)

    cfg = SimulationConfig(
        network_cls=model_cls,
        base_network_kwargs={"tau_spike_filter": 50e-3},
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        balance=balance,
        connection_prob_E=0.1,
        connection_prob_I=0.2,
        initial_input_weight=2.5,
        rec_weight_std=0.2,
        initial_rec_weight=2.5,
        input_types=jnp.ones((N_inputs,), dtype=bool),
        fully_connected_input=False,
        t0=t0,
        t1=t1,
        dt=dt,
        lr=lr,
        critic_lr=1e-1,
        mean_synaptic_delay=1.5e-3,
        noise_scaling_factor=noise_scaling_factor,
        noise_floor=noise_floor,
        actor_warmup_time=300,
        critic_warmup_time=10,
        key=key,
        save_at=save_at,
        save_file="results/pendulum.npz",
        network_output_fn=network_output_fn,
        network_output_shape=(1,),
        input_spike_fn=input_spike_fn,
        reward_fn=reward_fn,
        environment_model=PendulumEnvironment,
        environment_kwargs={"key": env_key, **env_kwargs},
        reward_prediction_model=LinearReadoutCritic,
        reward_predictor_kwargs={
            "input_dim": N_inputs,
            "pre_trained_weights": None,
        },
        delta_V=jnp.power(jnp.float64(2), jnp.float64(-13)),
        critic_input_fn=critic_input_fn,
        get_balance_rate=lambda t, state, args: jnp.max(
            jnp.array([10 - t * 9.0 / 100, 1.0])
        ),
        use_noise=jnp.array([True]),
        env_warmup_fn=lambda t, env_state, args: jnp.asarray(env_state[2] < 0.25),
        RPE_fn=lambda t, x, args, reward: reward
        - x.reward_predictor_state.previous_value
        + gamma * x.reward_predictor_state.value,
        episode_end_fn=lambda t, x, args: jnp.any(
            jnp.abs(x.environment_state)
            > jnp.array(
                [
                    env.max_allowed_angle,
                    env.max_allowed_angular_velocity,
                    env.max_episode_time,
                ]
            )
        ),
    )
    return cfg
