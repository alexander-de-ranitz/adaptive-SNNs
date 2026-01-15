import os
import time

import diffrax as dfx
import equinox as eqx
import jax.random as jr
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
from adaptive_SNN.visualization import plot_learning_results


def main():
    t0 = 0
    t1 = 100
    dt = 1e-4
    key = jr.PRNGKey(0)
    key, _ = jr.split(key)
    N_neurons = 1
    N_inputs = 2

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rates = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    lrs = [0.05, 0.2, 0.5]
    noise_levels = [0.05, 0.2, 0.5]
    min_noise_std = 5e-9
    for lr in lrs:
        for noise_level in noise_levels:
            if os.path.exists(
                f"../figures/learning_output_rate_noise_{noise_level}_lr_{lr}.png"
            ):
                print(
                    f"Figure for noise level {noise_level} and lr {lr} already exists. Skipping simulation."
                )
                continue

            print(f"Running simulations with noise level: {noise_level}")

            # Set up models
            neuron_model = LIFNetwork(
                N_neurons=N_neurons,
                N_inputs=N_inputs,
                dt=dt,
                fully_connected_input=True,
                input_weight=5.0,
                input_types=jnp.array([1, 0]),
                key=key,
            )
            key, _ = jr.split(key)

            noise_model = OUP(tau=neuron_model.tau_E, dim=N_neurons)

            network = NoisyNetwork(
                neuron_model=neuron_model,
                noise_model=noise_model,
                min_noise_std=min_noise_std,
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

            # Define args
            # Create a base seed that's unique for each loop iteration
            args = {
                "get_learning_rate": lambda t, x, args: jnp.where(t < 5, 0.0, lr),
                "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
                    agent_state.noisy_network.network_state.S[0]
                ),
                "reward_fn": lambda t, environment_state, args: -jnp.abs(
                    environment_state[0] - target_state
                ),
                "get_input_spikes": lambda t, x, args: jr.poisson(
                    jr.fold_in(jr.PRNGKey(0), jnp.round(t / dt).astype(int)),
                    rates * dt,
                    shape=(N_neurons, N_inputs),
                ),
                "get_desired_balance": lambda t, x, args: jnp.array([2.0]),
                "noise_scale_hyperparam": noise_level,
            }

            initial_weight_factors = jnp.array([0.5, 1.0, 2.0])

            def get_weights(state: SystemState):
                return state.agent_state.noisy_network.network_state.W

            sols = []
            for i in range(len(initial_weight_factors)):
                init = eqx.tree_at(
                    get_weights,
                    init_state,
                    get_weights(init_state) * initial_weight_factors[i],
                )
                key = jr.fold_in(jr.PRNGKey(42), i)
                print(f"Running simulation {i + 1}/{len(initial_weight_factors)} ...")
                start = time.time()

                def save_fn(t, y: SystemState, args):
                    return save_part_of_state(
                        y,
                        reward=True,
                        W=True,
                        environment_state=True,
                        mean_E_conductance=True,
                        var_E_conductance=True,
                        noise_state=True,
                        V=True,
                        S=True,
                        G=True,
                    )

                sol = simulate_noisy_SNN(
                    model,
                    solver,
                    t0,
                    t1,
                    dt,
                    init,
                    save_at=SaveAt(fn=save_fn, ts=jnp.linspace(t0, t1, 1000)),
                    args=args,
                    key=key,
                )
                end = time.time()
                print(f"Simulation completed in {end - start:.2f} seconds.")

                sols.append(sol)

            print("Plotting results...")

            plot_learning_results(
                sols,
                model,
                args,
                target_state,
                save_path=f"../figures/learning_output_rate_noise_{noise_level}_lr_{lr}.png",
            )


if __name__ == "__main__":
    main()
