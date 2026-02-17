import diffrax as dfx
import equinox as eqx
import jax.random as jr
import numpy as np
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models import (
    OUP,
    Agent,
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
    NoisyNetworkState,
)
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN


def run_rate_learning_simulation(
    t0: float,
    t1: float,
    dt: float,
    save_at: SaveAt,
    output_file: str,
    noise_level: float,
    lr: float,
    iterations: int,
    key_seed: int,
    initial_weight: float,
    balance: float,
):
    key = jr.fold_in(jr.PRNGKey(0), key_seed)

    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    network_key = jr.PRNGKey(0)  # Doesn't actually matter since weights fixed

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=initial_weight,
        input_types=jnp.array([1, 0]),
        weight_std=0.0,
        key=network_key,
    )

    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=5e-9,
    )

    agent = Agent(
        neuron_model=network,
        reward_model=RewardModel(reward_rate=10),
    )

    model = AgentEnvSystem(
        agent=agent,
        environment=SpikeRateEnvironment(
            dim=1,
        ),
    )
    solver = dfx.EulerHeun()
    init_state = model.initial

    for i in range(iterations):
        key, spike_key, simulation_key = jr.split(key, 3)

        args = {
            "get_learning_rate": lambda t, x, args: jnp.where(t < 5, 0.0, lr),
            "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
                agent_state.noisy_network.network_state.S[0]
            ),
            "reward_fn": lambda t, environment_state, args: -jnp.abs(
                environment_state[0] - target_state
            ),
            "get_input_spikes": lambda t, x, args: jr.poisson(
                jr.fold_in(spike_key, jnp.round(t / dt).astype(int)),
                rates * dt,
                shape=(N_neurons, N_inputs),
            ),
            "get_desired_balance": lambda t, x, args: jnp.array([balance]),
            "noise_scale_hyperparam": noise_level,
        }

        sol = simulate_noisy_SNN(
            model,
            solver,
            t0,
            t1,
            dt,
            init_state,
            save_at=save_at,
            args=args,
            key=simulation_key,
        )

        np.savez(
            output_file,
            times=sol.ts,
            sol=sol.ys,
        )


def run_rate_learning_simulation_fixed_I_weight(
    t0: float,
    t1: float,
    dt: float,
    save_at: SaveAt,
    output_file: str,
    noise_level: float,
    lr: float,
    iterations: int,
    key_seed: int,
    initial_weight_E: float,
    initial_weight_I: float,
):
    key = jr.fold_in(jr.PRNGKey(0), key_seed)

    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    network_key = jr.PRNGKey(0)  # Doesn't actually matter since weights fixed

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=initial_weight_E,
        input_types=jnp.array([1, 0]),
        weight_std=0.0,
        key=network_key,
    )

    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=5e-9,
    )

    agent = Agent(
        neuron_model=network,
        reward_model=RewardModel(reward_rate=10),
    )

    model = AgentEnvSystem(
        agent=agent,
        environment=SpikeRateEnvironment(
            dim=1,
        ),
    )
    solver = dfx.EulerHeun()
    init_state = model.initial

    init_state = eqx.tree_at(
        lambda s: s.agent_state.noisy_network.network_state.W,
        init_state,
        jnp.array([[-jnp.inf, initial_weight_E, initial_weight_I]]),
    )

    for i in range(iterations):
        key, spike_key, simulation_key = jr.split(key, 3)

        args = {
            "get_learning_rate": lambda t, x, args: jnp.where(t < 5, 0.0, lr),
            "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
                agent_state.noisy_network.network_state.S[0]
            ),
            "reward_fn": lambda t, environment_state, args: -jnp.abs(
                environment_state[0] - target_state
            ),
            "get_input_spikes": lambda t, x, args: jr.poisson(
                jr.fold_in(spike_key, jnp.round(t / dt).astype(int)),
                rates * dt,
                shape=(N_neurons, N_inputs),
            ),
            "get_desired_balance": lambda t, x, args: jnp.array(
                [0.0]
            ),  # Balance not used since I weight fixed
            "noise_scale_hyperparam": noise_level,
        }

        sol = simulate_noisy_SNN(
            model,
            solver,
            t0,
            t1,
            dt,
            init_state,
            save_at=save_at,
            args=args,
            key=simulation_key,
        )

        np.savez(
            output_file,
            times=sol.ts,
            sol=sol.ys,
        )


def run_noisy_network_simulation(
    t0: float,
    t1: float,
    dt: float,
    save_at: dfx.SaveAt,
    output_file: str,
    balance: float,
    noise_level: float,
    key_seed: int,
    initial_weight: float,
):
    key = jr.fold_in(jr.PRNGKey(0), key_seed)

    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    network_key = jr.PRNGKey(0)  # Doesn't actually matter since weights fixed

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt,
        fully_connected_input=True,
        input_weight=initial_weight,
        input_types=jnp.array([1, 0]),
        weight_std=0.0,
        key=network_key,
    )

    noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_model=noise_model,
        min_noise_std=5e-9,
    )

    solver = dfx.EulerHeun()
    init_state = model.initial

    key, spike_key, simulation_key = jr.split(key, 3)

    args = {
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.fold_in(spike_key, jnp.round(t / dt).astype(int)),
            rates * dt,
            shape=(N_neurons, N_inputs),
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([balance]),
        "noise_scale_hyperparam": noise_level,
    }

    sol = simulate_noisy_SNN(
        model,
        solver,
        t0,
        t1,
        dt,
        init_state,
        save_at=save_at,
        args=args,
        key=simulation_key,
    )

    state: NoisyNetworkState = sol.ys

    np.savez(output_file, times=sol.ts, sol=state)


def compute_I_weight_for_balance(desired_balance, E_weight):
    curr_balance = (
        E_weight
        * jnp.abs(LIFNetwork.reversal_potential_I - LIFNetwork.resting_potential)
        * LIFNetwork.tau_I
    ) / (
        E_weight
        * jnp.abs(LIFNetwork.reversal_potential_E - LIFNetwork.resting_potential)
        * LIFNetwork.tau_E
        + 1e-12
    )
    adjust_ratio = desired_balance / curr_balance
    I_weight = adjust_ratio * E_weight
    return I_weight
