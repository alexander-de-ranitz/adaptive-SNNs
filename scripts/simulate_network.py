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
from adaptive_SNN.visualization import plot_voltage_spectrum


def main():
    t0 = 0
    t1 = 10
    dt0 = 1e-4
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
        input_weight=2.0,
        key=key,
    )

    # noise_E_model = OUP(theta=250.0, noise_scale=1e-7, mean=0.0, dim=N_neurons)
    # noise_I_model = OUP(theta=250.0, noise_scale=1e-7, mean=0.0, dim=N_neurons)

    noise_E_model = OUP(tau=250.0, noise_scale=0, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(tau=250.0, noise_scale=0, mean=0.0, dim=N_neurons)

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
        "network_output_fn": lambda t, agent_state, args: jnp.squeeze(
            agent_state.noisy_network.network_state.S[0]
        ),
        "reward_fn": lambda t, environment_state, args: -jnp.abs(
            environment_state[0] - target_state
        ),
        "get_desired_balance": lambda t, x, args: jnp.array([4.0]),
        "get_input_spikes": lambda t, x, args: jr.poisson(
            jr.PRNGKey((t / dt0).astype(int)), rate * dt0, shape=(N_inputs,)
        ),
    }

    print("Running simulation...")
    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    # print("Plotting results...")
    # plot_simulate_SNN_results(
    #     sol, model, t0, t1, dt0, args, neurons_to_plot=jnp.array([0])
    # )

    plot_voltage_spectrum(sol, model, t0, t1, dt0, neurons_to_plot=jnp.array([0]))

    W = sol.ys.agent_state.noisy_network.network_state.W
    G = sol.ys.agent_state.noisy_network.network_state.G
    exc_mask = model.agent.noisy_network.base_network.excitatory_mask
    weighed_G_inhibitory = jnp.sum(W * G * jnp.invert(exc_mask[None, :]), axis=-1)
    weighed_G_excitatory = jnp.sum(W * G * exc_mask[None, :], axis=-1)
    print(
        "Mean conductances: = "
        + str(jnp.nanmean(weighed_G_excitatory[:, 0]))
        + ", "
        + str(jnp.nanmean(weighed_G_inhibitory[:, 0]))
    )
    print(
        "Conductance stds: = "
        + str(jnp.nanstd(weighed_G_excitatory[:, 0]))
        + ", "
        + str(jnp.nanstd(weighed_G_inhibitory[:, 0]))
    )
    expected_E_conductance = (
        model.agent.noisy_network.base_network.tau_E
        * model.agent.noisy_network.base_network.synaptic_increment
        * exc_rate
        * jnp.sum(W[-1, 0, :] * exc_mask)
    )
    print("Expected excitatory conductance: " + str(expected_E_conductance))
    expected_I_conductance = (
        model.agent.noisy_network.base_network.tau_I
        * model.agent.noisy_network.base_network.synaptic_increment
        * inh_rate
        * jnp.sum(W[-1, 0, :] * jnp.invert(exc_mask))
    )
    print("Expected inhibitory conductance: " + str(expected_I_conductance))


if __name__ == "__main__":
    main()
