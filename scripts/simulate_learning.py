import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import (
    OUP,
    Agent,
    AgentEnvSystem,
    LIFNetwork,
    NoisyNetwork,
)
from adaptive_SNN.models.environment import SpikeRateEnvironment
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.visualization.plotting import plot_learning_results


def main():
    t0 = 0
    t1 = 75.0
    dt0 = 5e-5
    key = jr.PRNGKey(1)
    N_neurons = 1
    N_inputs = 2

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        dt=dt0,
        input_neuron_types=jnp.array([1.0, 0.0]),
        fully_connected_input=True,
        input_weight=1.0,
        key=key,
    )
    key, _ = jr.split(key)

    noise_E_model = OUP(theta=250.0, noise_scale=100e-9, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=250.0, noise_scale=100e-9, mean=0.0, dim=N_neurons)

    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )
    agent = Agent(
        neuron_model=network,
        reward_model=RewardModel(reward_rate=10),
    )

    model = AgentEnvSystem(
        agent=agent,
        environment=SpikeRateEnvironment(
            dim=1,
            buffer_size=1.0 / dt0,  # 1 second buffer
            dt=dt0,
        ),
    )
    solver = dfx.EulerHeun()
    init_state = model.initial

    # Define input parameters
    exc_rate = 5000
    exc_to_inh_ratio = 4.0
    inh_rate = exc_rate / exc_to_inh_ratio
    rate = jnp.array([exc_rate, inh_rate])  # firing rate in Hz

    target_state = 10.0  # Target output state

    # Define args
    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(t < 2.5, 0.0, 0.0),
        "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
            agent_state.noisy_network.network_state.S[0]
        ),
        "reward_fn": lambda t, environment_state, args: -jnp.abs(
            environment_state[0] - target_state
        ),
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey((t / dt0).astype(int)), rate * dt0, shape=(N_inputs,)
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([4.0]),
    }

    print("Running simulation...")
    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1000, args=args
    )

    print("Plotting results...")
    plot_learning_results(sol, model, t0, t1, dt0, args)


if __name__ == "__main__":
    main()
