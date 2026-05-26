import jax

jax.config.update("jax_enable_x64", True)

from diffrax import EulerHeun, SaveAt
from equinox import tree_at
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.networks.gated_LIF import GatedLIFNetwork
from adaptive_SNN.models.RPE import BiphasicRPEModel
from adaptive_SNN.simulation_configs.biofeedback_experiment import create_config
from adaptive_SNN.solver import solve_ODE
from adaptive_SNN.utils.runner import run_simulation


def main():
    cfg = create_config(
        model_cls=GatedLIFNetwork, N_neurons=1000, key=jr.PRNGKey(192837465)
    )
    cfg.t1 = 2.0
    # Only recurrent synapses are plastic in this experiment, so we set the learning rate for input synapses to zero
    cfg.lr = 25.0 * jnp.hstack(
        [
            jnp.ones((cfg.N_neurons, cfg.N_neurons)),
            jnp.zeros((cfg.N_neurons, cfg.N_inputs)),
        ]
    )
    cfg.noise_level = 5e-9
    cfg.save_at = SaveAt(
        ts=jnp.arange(0.0, cfg.t1, cfg.dt),
        fn=lambda t, x, args: (
            x.agent_state.network_state.S[0].astype(jnp.bool),
            x.agent_state.RPE.RPE.astype(jnp.float32),
            x.agent_state.network_state.W[0].astype(jnp.float32),
            x.agent_state.network_state.features.eligibility[0].astype(jnp.float32),
        ),
    )

    sol, model = run_simulation(
        cfg, overwrite=True, save_results=True, load_if_exists=True
    )

    S = sol.ys[0]
    RPE = sol.ys[1]
    W = sol.ys[2]
    eligibility = sol.ys[3]
    ts = sol.ts

    fig, axs = plt.subplots(4, 1, figsize=(6, 6), sharex=True)

    spike_times_per_neuron = (
        [ts[jnp.nonzero(S[:, i])[0]] * 1e3 for i in range(S.shape[1])][::-1]
        if S.ndim > 1
        else [ts[jnp.nonzero(S)[0]] * 1e3]
    )  # Reverse order for better visualization
    axs[0].eventplot(spike_times_per_neuron)
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Neuron Index")
    axs[0].set_title("Spike Raster Plot")

    axs[1].plot(ts * 1e3, RPE)
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("RPE")
    axs[1].set_title("Reward Prediction Error")

    W = W[:, ~jnp.isnan(W[0])]
    print("Initial weight:", W[1])
    print("Final weight:", W[-1])
    diff = W[-1] - W[1]
    print("Weight change:", diff)
    axs[2].plot(ts * 1e3, W)
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylabel("Weight")
    axs[2].set_title("Synaptic Weights")

    e_to_plot = eligibility[:, : cfg.N_neurons][
        :, model.agent.network.base_network.excitatory_mask[: cfg.N_neurons]
    ]
    axs[3].plot(ts * 1e3, e_to_plot)
    axs[3].set_xlabel("Time (ms)")
    axs[3].set_ylabel("Eligibility Trace")
    axs[3].set_title("Eligibility Trace")
    plt.show()


def plot_RPE_kernel():
    model = BiphasicRPEModel(time_constants=jnp.array([0.1, 1.0]))
    dt = 1e-4
    initial_state = model.initial

    # Set initial state as if there was a positive RPE update at time 0, to see the kernel response
    initial_state = tree_at(lambda s: s.internal_state, initial_state, model.amplitudes)
    initial_state = tree_at(
        lambda s: s.RPE,
        initial_state,
        jnp.array([initial_state.internal_state[0] - initial_state.internal_state[1]]),
    )

    print(f"Initial state: {initial_state.RPE}")

    sol = solve_ODE(
        model=model,
        y0=initial_state,
        solver=EulerHeun(),
        save_at=SaveAt(t0=True, steps=True),
        t0=0.0,
        t1=5.0,
        dt0=dt,
        args={
            "RPE_update": 0.0
        },  # Provide a unit RPE update to see the kernel response
    )
    t = sol.ts
    kernel = sol.ys.RPE
    print(kernel.mean())
    plt.plot(t, kernel)
    plt.axhline(kernel.mean(), color="gray", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("RPE Kernel")
    plt.title("Biphasic RPE Kernel")
    plt.show()


if __name__ == "__main__":
    # main()
    plot_RPE_kernel()
