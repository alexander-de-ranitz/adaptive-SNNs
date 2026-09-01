import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from diffrax import SaveAt
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from adaptive_SNN.simulation_configs.single_synapse_learning_AC import (
    create_single_synapse_learning_config,
)
from adaptive_SNN.utils.runner import run_simulation
from adaptive_SNN.utils.save_helper import save_part_of_state
from adaptive_SNN.visualization import plot_noise_STA


def plot_noise_level_STA():
    sols = []
    key = jr.PRNGKey(15105)
    min_noises = [1e-10, 1e-9, 10e-9]
    for i, noise_level in enumerate(min_noises):
        key = jr.fold_in(key, i)
        config = create_single_synapse_learning_config(
            key=key, initial_synapse_weight=1.0
        )
        config.balance = 0.01
        config.min_noise_std = noise_level
        config.t1 = 510

        save_at = SaveAt(
            ts=jnp.linspace(10, config.t1, int((config.t1 - 10) / config.dt)),
            fn=lambda t, x, args: save_part_of_state(
                x,
                S=True,
                V=True,
                perturbations=True,
            ),
        )
        config.save_at = save_at

        config.save_file = f"results/STA/STA_noise_level_{noise_level:.0e}.npz"

        sol, model = run_simulation(config, save_results=True)
        sols.append(sol)

    plot_noise_STA(
        sols,
        model.agent.network if model is not None else None,
        noise_levels=min_noises,
        neurons_to_plot=jnp.array([0]),
    )


def plot_STA_voltage_trace():
    save_at = SaveAt(
        steps=True,
        fn=lambda t, x, args: save_part_of_state(
            x,
            S=True,
            V=True,
        ),
    )

    key = jr.PRNGKey(26358)

    config = create_single_synapse_learning_config(key=key, initial_synapse_weight=1.0)
    config.min_noise_std = 0.0
    config.save_at = save_at
    config.t1 = 100
    config.save_file = "results/STA_voltage_trace.npz"
    sol, model = run_simulation(config, save_results=True)

    # Get info from first neuron only
    S = sol.ys.agent_state.network_state.S[:, 0]
    V = sol.ys.agent_state.network_state.V[:, 0]

    V_STA = []
    spike_idx = jnp.where(S == 1)[0]
    offset_left = 1000
    offset_right = 100
    print(f"Number of spikes: {len(spike_idx)}")
    for idx in spike_idx:
        start_idx = idx - offset_left
        end_idx = idx + offset_right
        if start_idx >= 0 and end_idx < len(V):
            V_STA.append(V[start_idx:end_idx])

    V_STA = jnp.vstack(V_STA)
    V_mean = jnp.mean(V_STA, axis=0)
    t = jnp.arange(-offset_left, offset_right) * 1e-4  # Convert to seconds

    # Fit exponential
    def exponential(t, A, tau, C):
        return A * jnp.exp(t / tau) + C

    popt, _ = curve_fit(
        exponential,
        t[t < 0],
        V_mean[t < 0],
        p0=(1, 0.01, 0),
        bounds=([-jnp.inf, 1e-6, -jnp.inf], [jnp.inf, jnp.inf, jnp.inf]),
    )
    print(f"Fitted parameters: A={popt[0]}, tau={popt[1]}, C={popt[2]}")
    print(f"Mean voltage 1.8ms before spike: {exponential(-0.0018, *popt)}")

    delta_Vs = [0.5**k for k in range(6, 14)]
    gating = [jnp.exp((exponential(-0.0018, *popt) + 50e-3) / dv) for dv in delta_Vs]
    [print(f"Gating for delta_V={dv}: {g}") for dv, g in zip(delta_Vs, gating)]

    plt.plot(t * 1e3, V_mean, c="k", linewidth=2)
    plt.plot(
        t[:offset_left] * 1e3,
        exponential(t[:offset_left], *popt),
        "r--",
        label="Fitted Exponential",
    )
    plt.title("Spike-Triggered Average of Voltage")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.show()


if __name__ == "__main__":
    plot_noise_level_STA()
    # plot_STA_voltage_trace()
