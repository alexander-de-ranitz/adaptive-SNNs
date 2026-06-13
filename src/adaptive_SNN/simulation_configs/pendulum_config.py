import jax

jax.config.update("jax_enable_x64", True)
import jax.random as jr
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import SystemState
from adaptive_SNN.models.environments import PendulumEnvironment
from adaptive_SNN.models.networks import GatedLIFNetwork
from adaptive_SNN.models.reward_prediction import RLSRewardPredictor
from adaptive_SNN.utils.config import SimulationConfig


def compute_rates(
    env_state,
    N_encoding_inputs=200,
    width_factor=0.2,
    target_in_rate=1500,
    max_state_values=jnp.array(
        [
            PendulumEnvironment.max_allowed_angle,
            PendulumEnvironment.max_allowed_angular_velocity,
        ]
    ),
):
    """Compute the firing rates for the encoding population

    The population consists of N_encoding_inputs neurons, divided into 2 subpopulations that encode the angle and angular velocity respectively.
    On average, width_factor * 100 percent of the neurons will be active for any given state. The max firing rate of the neurons is scaled to achieve an average firing rate of target_rate across the entire population.


    Args:
        env_state: The state of the pendulum environment.
        N_encoding_inputs: The number of encoding neurons.
        width_factor: The width of the Gaussian tuning curves as a fraction of the state space.
        target_rate: The target input rate for the recurrent population
        max_state_value: The maximum value of the state space.
    Returns:
        rates: A vector of firing rates for the encoding population.
    """
    env_state = env_state[
        :2
    ]  # Only encode the angle and angular velocity, not the time
    N_encoding_populations = 2
    encoding_population_size = N_encoding_inputs // N_encoding_populations
    max_state_values = max_state_values * (1 + 3 * width_factor)
    tuning_curve_widths = width_factor * (2 * max_state_values)

    mean_expected_rate = jnp.sqrt(2 * jnp.pi) * width_factor
    max_encoding_rate = target_in_rate / (mean_expected_rate * N_encoding_inputs * 0.1)

    # Create preferred inputs for each encoding population
    # array of (encoding_population_size, N_encoding_populations) where each column corresponds to the preferred inputs of one population
    preferred_inputs = jnp.linspace(
        -max_state_values, max_state_values, encoding_population_size
    )

    # Encode the environment state into spikes using a population of encoding neurons
    env_state = jnp.clip(
        env_state, -max_state_values, max_state_values
    )  # clip to ensure within tuning curve range
    rates = max_encoding_rate * jnp.exp(
        -((env_state - preferred_inputs) ** 2) / (2 * (tuning_curve_widths) ** 2)
    )  # Gaussian tuning curves

    # Flatten in column-major order to turn into a single vector of length N_encoding_inputs
    # The first encoding_pop_size values of the vector corresponds to the first encoding population, etc.
    rates = rates.flatten(order="F")
    return rates


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
    N_inputs = 200
    agent_output_scaling = (
        0.25  # Scale the output so that it doesn't produce excessively large torques
    )

    env = PendulumEnvironment()

    key, spike_key = jr.split(key)

    def input_spike_fn(t, x: SystemState, args):
        step_idx = jnp.asarray(jnp.rint((t) / dt), dtype=jnp.int64)
        current_key = jr.fold_in(spike_key, step_idx)

        env_state = x.environment_state
        rates = compute_rates(env_state, N_encoding_inputs=N_inputs)

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
        warmup_time=100,
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
        reward_predictor_kwargs={"input_dim": N_neurons},
        args={
            "delta_V": jnp.power(jnp.float64(2), jnp.float64(-12)),
            "use_noise": jnp.array([True]),
            "feature_fn": lambda t,
            network_state,
            args: network_state.filtered_spike_trains,
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
