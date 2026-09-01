import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dfx
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.reward_prediction import LinearReadoutCritic
from adaptive_SNN.solver import solve_ODE
from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)


class LearningState(eqx.Module):
    critic_state: object
    RPE: jnp.ndarray


class LearningCritic(eqx.Module):
    critic: LinearReadoutCritic

    @property
    def initial(self):
        # Use normally distributed weights for the critic instead of zero's to test
        # the theoretical result that all weights converge to the same value
        critic_initial = self.critic.initial
        critic_initial = eqx.tree_at(
            lambda x: x.weights,
            critic_initial,
            jr.normal(jr.PRNGKey(0), critic_initial.weights.shape),
        )
        return LearningState(critic_state=critic_initial, RPE=jnp.zeros(1))

    @property
    def noise_shape(self):
        return LearningState(critic_state=self.critic.noise_shape, RPE=None)

    def pre_step_update(self, t, state, args):
        new_critic_state = self.critic.pre_step_update(
            t, state.critic_state, args, None, None, None, None
        )
        return LearningState(critic_state=new_critic_state, RPE=state.RPE)

    def drift(self, t, state, args):
        critic_drift = self.critic.drift(t, state.critic_state, args, None, state.RPE)
        return LearningState(critic_state=critic_drift, RPE=jnp.zeros_like(state.RPE))

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
        reward = args["reward_fn"](t, new_critic_state, args)
        new_RPE = args["RPE_fn"](t, new_critic_state, args, reward)
        return LearningState(critic_state=new_critic_state, RPE=new_RPE)


def simulate_learned_value(
    dt,
    tau_filtered_input,
    tau_discount,
    rate,
    critic_lr,
    reward,
    t1,
    n_channels,
    warmup_t,
    key,
):
    """Train a critic with N independent Poisson input channels; return mean of V(t), initial weights, and final weights."""
    critic = LinearReadoutCritic(
        use_bias=False, tau_filtered_input=tau_filtered_input, input_dim=n_channels
    )
    model = LearningCritic(critic=critic)
    gamma = 1 - dt / tau_discount

    def input_fn(t, x, args, input_spikes, env_state):
        step_idx = jnp.asarray(jnp.rint(t / dt), dtype=jnp.int64)
        current_key = jr.fold_in(key, step_idx)
        return jr.poisson(current_key, rate * jnp.ones((n_channels,)) * dt).astype(
            jnp.float64
        )

    args = {
        "critic_input_fn": input_fn,
        "get_critic_lr": lambda t, state, args: critic_lr,
        "reward_fn": lambda t, state, args: jnp.array([reward]),
        "RPE_fn": lambda t, x, args, r: r - x.previous_value + gamma * x.value,
    }

    def save_fn(t, state, args):
        return state.critic_state.value

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
        return_final_state=True,
        key=key,
    )

    initial_weights = initial_state.critic_state.weights
    final_state = sol.ys[1]
    final_weights = final_state.critic_state.weights
    return jnp.mean(sol.ys[0]), initial_weights, final_weights


def analytic_value(
    n_channels, reward, gamma, dt, tau_filtered_input, nu_bar_per_channel
):
    """V* = r/(1-gamma) * epsilon, epsilon = (1 + C/((1-gamma) * sum_j nu_bar_j))^-1,
    C = 0.5 * (1 - gamma * exp(-dt/tau_nu))."""
    total_nu_bar = n_channels * nu_bar_per_channel
    C = 0.5 * (1.0 - gamma * jnp.exp(-dt / tau_filtered_input))
    epsilon = 1.0 / (1.0 + C / ((1.0 - gamma) * total_nu_bar))
    return reward / (1.0 - gamma) * epsilon


def main():
    tau_filtered_input = 0.1
    tau_discount = 1.0
    rate = 10.0
    reward = 1.0
    dt = 1e-4
    t1 = 100.0
    nu_bar = tau_filtered_input * rate

    gamma = 1 - dt / tau_discount
    V_true = reward / (1.0 - gamma)

    warmup = t1 / 2  # let the critic converge before averaging
    base_critic_lr = 1000

    n_neurons = jnp.round(jnp.geomspace(1, 1000, 5)).astype(int)

    learned_values = []
    for N in n_neurons:
        critic_lr = (
            base_critic_lr / N
        )  # scale the learning rate with N to keep the effective learning rate constant
        value, W_initial, W_final = simulate_learned_value(
            dt,
            tau_filtered_input,
            tau_discount,
            rate,
            critic_lr,
            reward,
            t1,
            N,
            warmup,
            jr.PRNGKey(0),
        )

        fig, axs = plt.subplots(1, 2, figsize=(6, 2))
        axs[0].hist(W_initial, bins=20, color="gray", alpha=1, label="Initial weights")
        axs[1].hist(W_final, bins=20, color="darkgreen", alpha=1, label="Final weights")
        axs[0].set_xlabel("Weights")
        axs[0].set_ylabel("Frequency")
        axs[0].legend()
        axs[1].set_xlabel("Weights")
        axs[1].set_ylabel("Frequency")
        axs[1].legend()
        plt.show()
        learned_values.append(value)
        print(f"N={N:3d}  V_learned={value:.2f}  V_true={float(V_true):.2f}")

    n_channels_smooth = jnp.geomspace(1 * 0.8, 1000 * 1.2, 1000)
    analytic_values = [
        float(analytic_value(n, reward, gamma, dt, tau_filtered_input, nu_bar))
        for n in n_channels_smooth
    ]

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.axhline(float(V_true), linestyle="--", color="black", label="True value")
    ax.plot(
        n_channels_smooth,
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
            f"{100 * (y - float(V_true)) / float(V_true):.0f}" for y in ax.get_yticks()
        ],
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
