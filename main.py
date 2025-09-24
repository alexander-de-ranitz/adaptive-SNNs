import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from adaptive_SNN.models.models import NeuronModel, OUP, NoisyNeuronModel
from adaptive_SNN.utils.solver import run_SNN_simulation
from matplotlib import pyplot as plt

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def simulate_neurons():
    t0=0
    t1=1
    dt0=0.01
    model = NeuronModel(N_neurons=5, num_inputs=0, key=jr.PRNGKey(0))
    solver = dfx.EulerHeun()
    key = jr.PRNGKey(42)
    terms = model.terms(key)
    init_state = model.initial
    saveat = dfx.SaveAt(steps=True)

    sol = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=init_state, saveat=saveat, adjoint=dfx.ForwardMode())
    t = sol.ts
    V, cond, spikes = sol.ys
    print(V.shape)
    for i in range(5):
        plt.plot(t, V[:,i], label=f"Neuron {i+1}")
    plt.legend()
    plt.show()

def simulate_OUP():
    key = jr.PRNGKey(0)
    t0=0
    t1=1000
    dt0=0.5

    noise_model = OUP(theta=1, noise_scale=1, dim = 3)
    solver = dfx.EulerHeun()
    terms = noise_model.terms(key)
    init_state = noise_model.initial
    saveat = dfx.SaveAt(ts=jnp.linspace(t0, t1, 1000))

    sol = dfx.diffeqsolve(terms, solver, t0=t0, t1=t1, dt0=dt0, y0=init_state, saveat=saveat, adjoint=dfx.ForwardMode(), max_steps=None)
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
    t0=0
    t1=10
    dt0=0.1
    model = NoisyNeuronModel(N_neurons=1)
    solver = dfx.EulerHeun()
    key = jr.PRNGKey(42)
    terms = model.terms(key)
    init_state = model.initial
    sol = run_SNN_simulation(terms, solver, t0, t1, dt0, init_state, save_every_n_steps=1, args=None)
    t = sol.ts

    (V, cond, spikes), noise_E, noise_I = sol.ys

    # Plot membrane potentials and noise as two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(model.N_neurons):
        ax1.plot(t, V[:,i], label=f"Neuron {i+1} V")
        ax2.plot(t, noise_E[:,i], label=f"Neuron {i+1} Noise E", c='r', linestyle='--')
        ax2.plot(t, noise_I[:,i], label=f"Neuron {i+1} Noise I", c='b', linestyle=':')
        ax2.plot(t, noise_E[:,i] + noise_I[:,i], c='k', label=f"Neuron {i+1} Total Noise")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.title("Noisy Neuron Model Simulation")
    plt.show()

if __name__ == "__main__":
    simulate_noisy_neurons()
    #simulate_OUP()
