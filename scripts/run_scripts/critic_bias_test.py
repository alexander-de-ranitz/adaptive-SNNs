import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dfx
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array
from matplotlib import pyplot as plt

from adaptive_snn.models.reward_prediction import LinearReadoutCritic
from adaptive_snn.solver import solve_ODE
from adaptive_snn.utils.encoding import poisson_grid_encoding
from adaptive_snn.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)


class State(eqx.Module):
    critic_state: object
    RPE: Array
    env_state: Array


class CriticEnvSystem(eqx.Module):
    critic: LinearReadoutCritic

    @property
    def initial(self):
        critic_initial = self.critic.initial
        # critic_initial = eqx.tree_at(
        #     lambda x: x.weights,
        #     critic_initial,
        #     jr.normal(jr.PRNGKey(0), critic_initial.weights.shape),
        # )
        return State(
            critic_state=critic_initial,
            RPE=jnp.zeros(1),
            env_state=jnp.array([0.0, 1.0]),
        )

    @property
    def noise_shape(self):
        return State(critic_state=self.critic.noise_shape, RPE=None, env_state=None)

    def pre_step_update(self, t, state, args):
        new_critic_state = self.critic.pre_step_update(
            t, state.critic_state, args, None, None, None, state.env_state
        )
        return State(
            critic_state=new_critic_state, RPE=state.RPE, env_state=state.env_state
        )

    def drift(self, t, state, args):
        critic_drift = self.critic.drift(t, state.critic_state, args, None, state.RPE)
        env_drift = (
            2 * jnp.pi * jnp.array([jnp.cos(2 * jnp.pi * t), -jnp.sin(2 * jnp.pi * t)])
        )
        return State(
            critic_state=critic_drift,
            RPE=jnp.zeros_like(state.RPE),
            env_state=env_drift,
        )

    def diffusion(self, t, state, args):
        diffusion_tree = jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr), else_do=ElementWiseMul(jnp.zeros_like(arr))
            ),
            state,
        )
        return MixedPyTreeOperator(diffusion_tree)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, state, args):
        new_critic_state = self.critic.update(t, state.critic_state, args)
        new_state = eqx.tree_at(lambda x: x.critic_state, state, new_critic_state)
        reward = args["reward_fn"](t, new_state, args)
        new_RPE = args["RPE_fn"](t, new_state, args, reward)
        return State(
            critic_state=new_critic_state, RPE=new_RPE, env_state=state.env_state
        )


def simulate_learned_value(
    dt,
    tau_filtered_input,
    tau_discount,
    rate,
    critic_lr,
    t1,
    n_input_neurons,
    warmup_t,
    key,
):
    critic = LinearReadoutCritic(
        use_bias=False,
        tau_filtered_input=tau_filtered_input,
        input_dim=n_input_neurons**2,
    )
    model = CriticEnvSystem(critic=critic)
    gamma = 1 - dt / tau_discount

    def input_fn(t, x, args, input_spikes, env_state):
        step_idx = jnp.asarray(jnp.rint(t / dt), dtype=jnp.int64)
        current_key = jr.fold_in(key, step_idx)

        rates = poisson_grid_encoding(
            state=env_state,
            grid_shape=(n_input_neurons, n_input_neurons),
            grid_range=[(-1.0, 1.0), (-1.0, 1.0)],
            total_rate=rate * n_input_neurons**2,
        ).flatten()

        return jr.poisson(current_key, rates * dt)

    args = {
        "critic_input_fn": input_fn,
        "get_critic_lr": lambda t, state, args: critic_lr,
        "reward_fn": lambda t, state, args: dt * (-state.env_state[1] - 1),
        "RPE_fn": lambda t, x, args, r: r
        - x.critic_state.previous_value
        + gamma * x.critic_state.value,
    }

    def save_fn(t, state, args):
        return state.critic_state.weights, state.critic_state.value

    initial_state = model.initial
    sol = solve_ODE(
        model,
        dfx.Euler(),
        t0=0.0,
        t1=t1,
        dt0=dt,
        args=args,
        save_at=dfx.SaveAt(ts=jnp.arange(warmup_t, t1, dt * 10), fn=save_fn),
        y0=initial_state,
        return_final_state=False,
        key=key,
    )

    return sol.ts, sol.ys[0], sol.ys[1]


def compute_true_value():
    r = -1 - jnp.cos(2 * jnp.pi * jnp.linspace(0, 10, 1000))
    kernel = jnp.exp(-jnp.linspace(0, 9, 900) / 1.0)
    return jnp.convolve(r, kernel[::-1], mode="valid")[:100] / jnp.sum(kernel)


def alternative_true_value(N, tau_filtered_input, tau_discount, rate, dt):
    gamma = 1 - dt / tau_discount
    C = 0.5 * (1.0 - gamma * jnp.exp(-dt / tau_filtered_input))
    epsilon = 1.0 / (1.0 + C / ((1.0 - gamma) * N**2 * rate * tau_filtered_input))
    r = -1 - jnp.cos(2 * jnp.pi * jnp.linspace(0, 10, 1000))
    tau_discount = tau_discount * epsilon
    kernel = jnp.exp(-jnp.linspace(0, 9, 900) / tau_discount)
    return jnp.convolve(r, kernel[::-1], mode="valid")[:100] / jnp.sum(kernel)


def main():
    tau_filtered_input = 0.1
    tau_discount = 1.0
    rate = 10.0
    dt = 1e-4
    mean_reward = -1.0 * dt
    t1 = 100.0
    true_value = compute_true_value()
    gamma = 1 - dt / tau_discount
    mean_V_true = mean_reward / (1 - gamma)
    C = 0.5 * (1.0 - gamma * jnp.exp(-dt / tau_filtered_input))
    epsilon = lambda N: 1.0 / (
        1.0 + C / ((1.0 - gamma) * N**2 * rate * tau_filtered_input)
    )

    warmup = t1 / 2  # let the critic converge before averaging
    base_critic_lr = 100
    n_runs = 3  # independent runs (different seeds) averaged to reduce noise

    n_neurons = jnp.array([1, 2, 4, 8, 16])

    learned_values = []
    for N in n_neurons:
        critic_lr = (
            base_critic_lr / N
        )  # scale the learning rate with N to keep the effective learning rate constant
        run_values = []
        for run_idx in range(n_runs):
            ts, learned_weights, learned_value = simulate_learned_value(
                dt,
                tau_filtered_input,
                tau_discount,
                rate,
                critic_lr,
                t1,
                N,
                warmup,
                jr.fold_in(jr.PRNGKey(99), run_idx),
            )
            run_values.append(learned_value[:, 0])

        learned_value = jnp.mean(jnp.stack(run_values), axis=0)
        phase = jnp.mod(ts, 1.0)
        edges = jnp.linspace(0, 1, len(true_value) + 1)
        pred_val = jnp.array(
            [
                jnp.mean(learned_value[(phase >= lo) & (phase < hi)])
                for lo, hi in zip(edges[:-1], edges[1:])
            ]
        )
        plt.plot(jnp.linspace(0, 1, len(pred_val)), pred_val, label="Learned value")
        plt.plot(jnp.linspace(0, 1, len(true_value)), true_value, label="True value")
        plt.plot(
            jnp.linspace(0, 1, len(true_value)),
            true_value * epsilon(N),
            "k--",
            label="Expected Fixed Point",
        )
        plt.plot(
            jnp.linspace(0, 1, len(true_value)),
            epsilon(N)
            * alternative_true_value(N, tau_filtered_input, tau_discount, rate, dt),
            "r--",
            label="Alternative Fixed Point",
        )
        plt.legend()
        plt.show()

        value = jnp.mean(learned_value)
        learned_values.append(value)
        analytic_val = epsilon(N) * mean_V_true
        print(
            f"N={N:3d}  V_learned={value:.2f}  V_true={float(mean_V_true):.2f}, V_analytic={float(analytic_val):.2f}"
        )

    n_neurons_smooth = jnp.geomspace(1 * 0.8, 1000 * 1.2, 1000)
    analytic_values = [
        float(analytic_val)
        for analytic_val in [epsilon(n) * mean_V_true for n in n_neurons_smooth]
    ]

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.axhline(float(mean_V_true), linestyle="--", color="black", label="True value")
    ax.plot(
        n_neurons_smooth,
        analytic_values,
        "-",
        color="gray",
        label="Expected fixed point",
    )
    ax.plot(
        n_neurons,
        learned_values,
        "o",
        label="Learned value",
        linestyle="none",
        color="darkgreen",
        markersize=6,
    )
    ax.set_xlim(1 * 0.8, 1000 * 1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_\text{in}$")
    ax.set_ylabel(r"Error (\%)")
    ax.set_yticks(
        ax.get_yticks(),
        labels=[
            f"{100 * (y - float(mean_V_true)) / float(mean_V_true):.0f}"
            for y in ax.get_yticks()
        ],
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
