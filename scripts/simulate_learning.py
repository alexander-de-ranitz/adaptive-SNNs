import diffrax as dfx
import jax.random as jr
from jax import numpy as jnp

from adaptive_SNN.models import OUP, AgentSystem, LIFNetwork, NoisyNetwork
from adaptive_SNN.models.environment import EnvironmentModel
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.solver import simulate_noisy_SNN
from adaptive_SNN.visualization.plotting import plot_learning_results


def main():
    t0 = 0
    t1 = 50
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
        input_weight=10.0,
        key=key,
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=50.0, noise_scale=5e-8, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=50.0, noise_scale=5e-8, mean=0.0, dim=N_neurons)
    network = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )
    model = AgentSystem(
        neuron_model=network,
        reward_model=RewardModel(),
        environment=EnvironmentModel(),
    )
    solver = dfx.EulerHeun()
    init_state = model.initial

    rate = 500  # firing rate in Hz
    p = 1.0 - jnp.exp(-rate * dt0)  # per-step spike probability, Poisson process
    target_state = 5.0  # Target output state

    # Define args
    args = {
        "get_learning_rate": lambda t, x, args: jnp.where(t < 0.3, 0.0, 1),
        "network_output_fn": lambda t, x, args: 1 / dt0 * jnp.sum(x.network_state.S[0]),
        "reward_fn": lambda t, x, args: -jnp.abs(
            jnp.sum(x[0]) - target_state
        ),  # Reward function
        "get_input_spikes": lambda t, x, args: jr.bernoulli(
            jr.PRNGKey((t / dt0).astype(int)), p=p, shape=(N_inputs,)
        ),
        "get_desired_balance": lambda t, x, args: jnp.array(
            [1.0]
        ),  # Desired E/I balance
    }

    print("Running simulation...")
    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=100, args=args
    )

    print("Plotting results...")
    plot_learning_results(sol, model, t0, t1, dt0, args)


if __name__ == "__main__":
    main()
