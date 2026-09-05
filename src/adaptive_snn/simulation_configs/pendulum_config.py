import jax

jax.config.update("jax_enable_x64", True)
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_snn.models import SystemState
from adaptive_snn.models.environments import PendulumEnvironment
from adaptive_snn.models.networks import GatedLIFNetwork
from adaptive_snn.models.reward_prediction import RLSRewardPredictor
from adaptive_snn.utils.config import SimulationConfig


def compute_rates(
    env_state,
    grid_shape=(16, 16),
    copies_per_feature=1,
    width_factor=0.2,
    pad_factor=0.2,
    target_in_rate=1500,
    downstream_connectivity=0.1,
    normalize=True,
    max_state_values=jnp.array(
        [
            PendulumEnvironment.max_allowed_angle,
            PendulumEnvironment.max_allowed_angular_velocity,
        ]
    ),
):
    """Compute the firing rates for the encoding population.

    The angle and angular velocity are jointly encoded by a population of 2D Gaussian tuning
    curves on a grid_shape grid, so each neuron is tuned to an (angle, velocity) pair and a
    linear critic can represent their interaction. Each tuning curve is replicated into
    copies_per_feature independent Poisson units, and the rates are normalized so the total
    drive stays constant across states. The amplitude is scaled such that a neuron connected to
    a fraction downstream_connectivity of the population receives target_in_rate spikes per second.

    Args:
        env_state: The state of the pendulum environment.
        grid_shape: The (angle, velocity) grid of tuning curves.
        copies_per_feature: Independent Poisson replicas per tuning curve.
        width_factor: The Gaussian width as a fraction of the padded encoding span.
        pad_factor: How far to extend the tuning-curve centers beyond the visited region.
        target_in_rate: The target input rate for a downstream neuron.
        downstream_connectivity: The connection probability assumed for the rate calibration.
        normalize: Whether to keep the total drive constant across states.
        max_state_values: The maximum (visited) state values, i.e. the reset threshold.
    Returns:
        rates: A vector of firing rates of length grid_shape[0] * grid_shape[1] * copies_per_feature.
    """
    env_state = env_state[
        :2
    ]  # Only encode the angle and angular velocity, not the time
    center_range = max_state_values * (1 + pad_factor)
    tuning_curve_widths = width_factor * (2 * center_range)

    # Grid of preferred (angle, velocity) pairs
    axes = [
        jnp.linspace(-center_range[i], center_range[i], grid_shape[i]) for i in range(2)
    ]
    preferred_inputs = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    # 2D Gaussian tuning curves, normalized to a constant total drive
    env_state = jnp.clip(env_state, -center_range, center_range)
    rates = jnp.exp(
        -jnp.sum(
            (env_state - preferred_inputs) ** 2 / (2 * tuning_curve_widths**2), axis=-1
        )
    )
    if normalize:
        rates = rates / jnp.sum(rates)

    # Scale so a downstream neuron sampling downstream_connectivity of the population gets target_in_rate
    expected_total = (
        1.0
        if normalize
        else grid_shape[0] * grid_shape[1] * (jnp.sqrt(2 * jnp.pi) * width_factor) ** 2
    )
    max_encoding_rate = target_in_rate / (
        downstream_connectivity * copies_per_feature * expected_total
    )

    return jnp.tile(max_encoding_rate * rates, copies_per_feature)


def create_pendulum_config(
    N_neurons=1000, model_cls=GatedLIFNetwork, key=jr.PRNGKey(0)
) -> SimulationConfig:
    t0 = 0
    t1 = 5
    dt = 1e-4
    lr = 0.0
    noise_level = 0.0
    min_noise_std = 5e-9
    balance = 1.05
    encoding_grid_shape = (16, 16)
    N_inputs = encoding_grid_shape[0] * encoding_grid_shape[1]
    agent_output_scaling = (
        0.25  # Scale the output so that it doesn't produce excessively large torques
    )

    env = PendulumEnvironment()

    key, spike_key = jr.split(key)

    def input_spike_fn(t, x: SystemState, args):
        step_idx = jnp.asarray(jnp.rint((t) / dt), dtype=jnp.int64)
        current_key = jr.fold_in(spike_key, step_idx)

        env_state = x.environment_state
        rates = compute_rates(env_state, grid_shape=encoding_grid_shape)

        # Generate the spikes of the encoding population — shape (N_inputs,).
        # This is broadcast to all neurons in the recurrent population
        encoding_spikes = jr.poisson(current_key, rates * dt)
        return encoding_spikes

    # Define network output function
    output_pop = jnp.arange(
        jnp.round(N_neurons * 0.1).astype(int)
    )  # the first 10% of neurons are the output population
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
        # return (x.environment_state, x.reward_signal, x.agent_state.reward_predictor_state.value, x.agent_state.network_state.filtered_spike_trains)
        return x.agent_state.network_state.S.astype(jnp.bool_)

    save_at = SaveAt(steps=True, fn=save)

    model_cls = model_cls
    cfg = SimulationConfig(
        network_cls=model_cls,
        base_network_kwargs={"tau_spike_filter": 0.05},
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        balance=balance,
        connection_prob_E=0.1,
        connection_prob_I=0.2,
        initial_input_weight=1.0,
        rec_weight_std=0.2,
        initial_rec_weight=1.0,
        input_types=jnp.ones((N_inputs,), dtype=bool),
        fully_connected_input=False,
        t0=t0,
        t1=t1,
        dt=dt,
        lr=lr,
        mean_synaptic_delay=1.5e-3,
        noise_level=noise_level,
        min_noise_std=min_noise_std,
        actor_warmup_time=100,
        key=key,
        save_at=save_at,
        save_file="results/pendulum.npz",
        network_output_fn=network_output_fn,
        network_output_shape=(1,),
        input_spike_fn=input_spike_fn,
        reward_fn=reward_fn,
        environment_model=PendulumEnvironment,
        environment_kwargs={},
        reward_prediction_model=RLSRewardPredictor,
        reward_predictor_kwargs={"input_dim": N_inputs},
        args={
            "delta_V": jnp.power(jnp.float64(2), jnp.float64(-12)),
            "use_noise": jnp.array([True]),
            "get_critic_lr": lambda t, x, args: 1e-5,
            "feature_fn": lambda t,
            network_state,
            args: network_state.filtered_spike_trains,
            "env_warmup_fn": lambda t, env_state, args: jnp.asarray(env_state[2] < 0.5),
            "episode_end_fn": lambda t, state, args: jnp.any(
                jnp.abs(state.environment_state)
                > jnp.array(
                    [
                        env.max_allowed_angle,
                        env.max_allowed_angular_velocity,
                        env.max_episode_time,
                    ]
                )
            ),
        },
    )
    return cfg


if __name__ == "__main__":
    cfg = create_pendulum_config()
