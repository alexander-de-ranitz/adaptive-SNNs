import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.environment import EnvironmentModel
from adaptive_SNN.models.models import (
    OUP,
    AgentSystem,
    LIFNetwork,
    NoisyNetwork,
)
from adaptive_SNN.models.reward import RewardModel
from adaptive_SNN.utils.plotting import (
    plot_learning_results,
    plot_simulate_noisy_SNN_results,
)
from adaptive_SNN.utils.solver import simulate_noisy_SNN

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def simulate_OUP():
    key = jr.PRNGKey(0)
    t0 = 0
    t1 = 1
    dt0 = 0.0001

    # From Destexhe, 2001
    tau = 2.6  # 2.6 ms
    mean = 18e-9  # 18 nS
    sigma = 3.5e-9  # 3.5 nS
    D = 2 * sigma / tau

    print(f"tau: {tau}, mean: {mean}, D: {D}")

    noise_model = OUP(theta=tau, noise_scale=D, mean=mean, dim=2)
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = noise_model.initial
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=init_state,
        saveat=saveat,
        adjoint=dfx.ForwardMode(),
        max_steps=None,
    )
    t = sol.ts
    x = sol.ys
    labels = ["Excitatory", "Inhibitory"]
    colors = ["g", "r"]
    for i in range(x.shape[1]):
        plt.plot(t, x[:, i], label=labels[i], color=colors[i])
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Conductance")
    plt.title("OU Process")
    plt.tight_layout()
    plt.show()


def simulate_noisy_neurons():
    t0 = 0
    t1 = 0.5
    dt0 = 0.001
    key = jr.PRNGKey(1)

    N_neurons = 1
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=0, fully_connected_input=True, key=key
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=50.0, noise_scale=50e-9, mean=25 * 1e-9, dim=N_neurons)
    noise_I_model = OUP(theta=50.0, noise_scale=50e-9, mean=50 * 1e-9, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=None
    )

    plot_simulate_noisy_SNN_results(sol, model, t0, t1, dt0)


def simulate_neuron_with_random_input():
    t0 = 0
    t1 = 0.5
    dt0 = 0.001
    key = jr.PRNGKey(1)

    N_neurons = 1
    N_inputs = 50
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=N_inputs, fully_connected_input=True, key=key
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    # Input spikes: Poisson with rate 20 Hz
    rate = 20  # firing rate in Hz
    p = 1.0 - jnp.exp(-rate * dt0)  # per-step spike probability, Poisson process
    args = {
        "p": p,
        "learning": False,
    }  # p = probability of spike in each input neuron at each time step

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_simulate_noisy_SNN_results(sol, model, t0, t1, dt0)


def train_SNN():
    t0 = 0
    t1 = 0.5
    dt0 = 0.001
    key = jr.PRNGKey(1)

    N_neurons = 1
    N_inputs = 50
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=N_inputs, fully_connected_input=True, key=key
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
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
    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    # Input spikes: Poisson with rate 20 Hz
    rate = 20  # firing rate in Hz
    p = 1.0 - jnp.exp(-rate * dt0)  # per-step spike probability, Poisson process

    args = {
        "learning": lambda t, x, args: True,
        "network_output": lambda t, x, args: (0.01 / dt0) * jnp.sum(x[0].S[0]),
        "compute_reward": lambda t, x, args: -jnp.abs(
            jnp.sum(x[0]) - 5.0
        ),  # Reward function
        "input_spikes": lambda t, x, args: jr.bernoulli(
            jr.PRNGKey(int(t / dt0)), p=p, shape=(N_inputs,)
        ),
        "desired_balance": lambda t, x, args: jnp.array([12.0]),  # Desired E/I balance
    }

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )
    plot_learning_results(sol, model, t0, t1, dt0, args)


def simulate_balancing():
    t0 = 0
    t1 = 0.5
    dt0 = 0.0001
    key = jr.PRNGKey(1)

    N_neurons = 1
    N_inputs = 50
    input_types = jnp.concatenate(
        [jnp.ones((int(N_inputs * 0.8),)), jnp.zeros((int(N_inputs * 0.2),))], axis=0
    )  # First half excitatory, second half inhibitory

    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons,
        N_inputs=N_inputs,
        input_neuron_types=input_types,
        fully_connected_input=True,
        key=key,
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=1.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    model = NoisyNetwork(
        neuron_model=neuron_model,
        noise_E_model=noise_E_model,
        noise_I_model=noise_I_model,
    )

    # Run simulation
    solver = dfx.EulerHeun()
    init_state = model.initial

    # Input spikes: Poisson with rate 20 Hz
    rate = 150  # firing rate in Hz
    p = 1.0 - jnp.exp(-rate * dt0)  # per-step spike probability, Poisson process
    args = {
        "input_spikes": lambda t, x, args: jr.bernoulli(
            jr.PRNGKey(int(t / dt0)), p=p, shape=(N_inputs,)
        ),
        "learning": lambda t, x, args: False,
        "desired_balance": lambda t, x, args: 0.3,  # Desired E/I balance
    }

    sol = simulate_noisy_SNN(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_simulate_noisy_SNN_results(sol, model, t0, t1, dt0)


if __name__ == "__main__":
    # simulate_noisy_neurons()
    # simulate_OUP()
    # simulate_neuron_with_random_input()
    # train_SNN()
    simulate_balancing()
    pass
