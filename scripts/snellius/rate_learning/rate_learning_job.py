import argparse

import diffrax as dfx
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
    SystemState,
)
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.utils.save_helper import save_part_of_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_weight", type=float, default=5.0, help="Initial weight factor"
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.0, help="Noise level hyperparameter"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6.5,
        help="Learning rate for synaptic weights",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to average over",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="rate_learning_results",
        help="Output file name",
    )
    parser.add_argument("--key_seed", type=int, default=0, help="Random key seed")

    args = parser.parse_args()
    noise_level = args.noise_level
    lr = args.learning_rate
    iterations = args.iterations
    key = jr.fold_in(jr.PRNGKey(0), args.key_seed)
    output_file = args.output_file
    initial_weight = args.initial_weight

    t0 = 0
    t1 = 50
    dt = 1e-4

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
            "get_desired_balance": lambda t, x, args: jnp.array([0.86]),
            "noise_scale_hyperparam": noise_level,
        }

        def save_fn(t, y: SystemState, args):
            return save_part_of_state(y, W=True, environment_state=True)

        sol = simulate_noisy_SNN(
            model,
            solver,
            t0,
            t1,
            dt,
            init_state,
            save_at=SaveAt(fn=save_fn, ts=jnp.linspace(t0, t1, 500)),
            args=args,
            key=simulation_key,
        )

        state: SystemState = sol.ys
        weights = state.agent_state.noisy_network.network_state.W
        env_state = state.environment_state
        filename = f"{output_file}_nl{noise_level}_lr{lr}_iter{i}.npz"
        np.savez(
            filename,
            times=sol.ts,
            weights=weights,
            environment_state=env_state,
        )


if __name__ == "__main__":
    main()
