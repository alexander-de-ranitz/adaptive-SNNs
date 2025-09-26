import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.models import OUP, LIFNetwork, NoisyLIFModel
from adaptive_SNN.utils.plotting import plot_results
from adaptive_SNN.utils.solver import run_SNN_simulation

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def simulate_OUP():
    key = jr.PRNGKey(0)
    t0 = 0
    t1 = 1000
    dt0 = 0.5

    noise_model = OUP(theta=1, noise_scale=1, dim=3)
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

    print(x.shape)
    print(type(x))

    plt.plot(t, x)
    print(jnp.mean(x, axis=0))
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.title("OU Process")
    plt.show()


def simulate_noisy_neurons():
    t0 = 0
    t1 = 0.5
    dt0 = 0.001
    key = jr.PRNGKey(1)

    N_neurons = 2
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=0, fully_connected_input=True, key=key
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=50.0, noise_scale=50e-9, mean=25 * 1e-9, dim=N_neurons)
    noise_I_model = OUP(theta=50.0, noise_scale=50e-9, mean=50 * 1e-9, dim=N_neurons)
    model = NoisyLIFModel(
        N_neurons=N_neurons,
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
    args = {"p": p}  # p = probability of spike in each input neuron at each time step

    sol, spikes = run_SNN_simulation(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_results(sol, spikes, model, t0, t1, dt0)


def simulate_input_neurons():
    t0 = 0
    t1 = 0.5
    dt0 = 0.001
    key = jr.PRNGKey(1)

    N_neurons = 1
    N_inputs = 3
    # Set up models
    neuron_model = LIFNetwork(
        N_neurons=N_neurons, N_inputs=N_inputs, fully_connected_input=True, key=key
    )
    key, _ = jr.split(key)
    noise_E_model = OUP(theta=0.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    noise_I_model = OUP(theta=0.0, noise_scale=0.0, mean=0.0, dim=N_neurons)
    model = NoisyLIFModel(
        N_neurons=N_neurons,
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
    args = {"p": p}  # p = probability of spike in each input neuron at each time step

    sol, spikes = run_SNN_simulation(
        model, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=args
    )

    plot_results(sol, spikes, model, t0, t1, dt0)


if __name__ == "__main__":
    # simulate_noisy_neurons()
    # simulate_OUP()
    simulate_input_neurons()
