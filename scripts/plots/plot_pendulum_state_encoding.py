import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.environments.pendulum import PendulumEnvironment
from adaptive_SNN.simulation_configs.pendulum_config import (
    compute_rates,
)


def plot_encoding_scheme():
    """Visualize the tuning curves of the encoding population for the pendulum environment."""
    env_state = jnp.array([0.1, 0.5])  # Some example state
    rates = compute_rates(env_state=env_state, width_factor=0.1, pad_factor=0.2)
    print(
        f"Min rate: {jnp.min(rates):.2f}, Max rate: {jnp.max(rates):.2f}, Sum rate: {jnp.sum(rates):.2f}"
    )
    print(rates.shape)
    plt.imshow(
        rates.reshape(16, 16),
        aspect="auto",
        interpolation="none",
    )
    plt.colorbar(label="Firing Rate (Hz)")
    plt.xlabel("")
    plt.ylabel("")
    plt.title("Tuning Curves of Encoding Population")
    plt.show()


def compute_error_floor():
    """Computes the error when trying to fit a linear readout of the state encoding on the value function"""
    fig, axs = plt.subplots(5, 3, figsize=(10, 8), sharex="row", sharey="row")
    colors = ["C0", "C1", "C2"]
    for i, width_factor in enumerate([0.1, 0.2, 0.3]):
        env = PendulumEnvironment()
        n_states = 1000
        states = jr.uniform(
            jr.PRNGKey(0),
            (n_states, 2),
            minval=jnp.array(
                [-env.max_allowed_angle, -env.max_allowed_angular_velocity]
            ),
            maxval=jnp.array([env.max_allowed_angle, env.max_allowed_angular_velocity]),
        )
        rates = (
            jnp.stack(
                [
                    compute_rates(state, pad_factor=0.2, width_factor=width_factor)
                    for state in states
                ]
            )
            + 100
        )
        values = -jnp.array(
            [state.T @ env.cost_to_go_matrix @ state for state in states]
        )

        # The predictor does not get the rates directly. The rates generate spike trains, and the spike trains are filtered to produce a continuous signal
        # We use the number of spikes in a 50ms window as a proxy for the filtered signal
        time_window = 0.05
        spike_trains = jax.random.poisson(jr.PRNGKey(42), rates * time_window)

        # Fit a linear model to the data
        X, residuals, rank, s = jnp.linalg.lstsq(
            jnp.concatenate([spike_trains, jnp.ones((n_states, 1))], axis=1),
            values,
            rcond=None,
        )

        test_states = jr.uniform(
            jr.PRNGKey(1001),
            (n_states, 2),
            minval=jnp.array(
                [-env.max_allowed_angle, -env.max_allowed_angular_velocity]
            ),
            maxval=jnp.array([env.max_allowed_angle, env.max_allowed_angular_velocity]),
        )
        test_rates = (
            jnp.stack(
                [
                    compute_rates(state, pad_factor=0.2, width_factor=width_factor)
                    for state in test_states
                ]
            )
            + 100
        )
        test_spike_trains = jax.random.poisson(
            jr.PRNGKey(1002), test_rates * time_window
        )
        test_values = -jnp.array(
            [state.T @ env.cost_to_go_matrix @ state for state in test_states]
        )

        # Plot the predicted values vs the true values
        predicted_values = (
            jnp.concat([test_spike_trains, jnp.ones((n_states, 1))], axis=1) @ X
        )
        print(
            f"Mean value: {jnp.mean(test_values):.4f}, Mean predicted value: {jnp.mean(predicted_values):.4f}"
        )
        print(
            f"R^2 score: {1 - jnp.sum((test_values - predicted_values) ** 2) / jnp.sum((test_values - jnp.mean(test_values)) ** 2):.4f}"
        )
        axs[0, i].scatter(test_values, predicted_values, color=colors[i], alpha=0.7)
        axs[0, i].plot(
            [jnp.min(test_values), jnp.max(test_values)],
            [jnp.min(test_values), jnp.max(test_values)],
            "k--",
        )
        axs[0, i].set_xlabel("True Value")
        axs[0, i].set_ylabel("Predicted Value")

        # Plot the residuals against true values
        residuals = test_values - predicted_values
        axs[1, i].scatter(test_values, residuals, color=colors[i], alpha=0.7)
        axs[1, i].axhline(0, color="k", linestyle="--")
        axs[1, i].set_xlabel("True Value")
        axs[1, i].set_ylabel("Residuals")

        # Plot the mean residual binned by true value
        bins = jnp.linspace(jnp.min(test_values), jnp.max(test_values), 25)
        mean_residuals = jnp.array(
            [
                jnp.mean(residuals[(test_values >= bin) & (test_values < bin_next)])
                for bin, bin_next in zip(bins[:-1], bins[1:])
            ]
        )
        axs[2, i].bar(
            bins[:-1],
            mean_residuals,
            width=bins[1] - bins[0],
            align="edge",
            color=colors[i],
            alpha=0.7,
        )
        axs[2, i].axhline(0, color="k", linestyle="--")
        axs[2, i].set_xlabel("True Value")
        axs[2, i].set_ylabel("Mean Residuals")

        # plot the residuals
        residuals = test_values - predicted_values
        print(
            f"Mean residual: {jnp.mean(residuals):.4f}, Std residual: {jnp.std(residuals):.4f}"
        )
        axs[3, i].hist(residuals, bins=30, color=colors[i], alpha=0.7)
        axs[3, i].axvline(jnp.mean(residuals), color="k", linestyle="--", label="Mean")
        axs[3, i].set_xlabel("Residuals")
        axs[3, i].set_ylabel("Count")

        # Plot the distribution of input rates for a hypothetical neuron with 10% connectivity
        W = jr.bernoulli(jr.PRNGKey(123), p=0.1, shape=(256,))
        print(f"Number of input connections: {jnp.sum(W)}")
        input_spikes = test_spike_trains @ W
        axs[4, i].hist(input_spikes, bins=30, color=colors[i], alpha=0.7)
        axs[4, i].set_xlabel("Number of Input Spikes")
        axs[4, i].set_ylabel("Count")

        if i == 1:
            axs[0, i].set_title(r"\textbf{True vs Predicted Value}")
            axs[1, i].set_title(r"\textbf{Residuals vs True Value}")
            axs[2, i].set_title(r"\textbf{Mean Residuals binned by True Value}")
            axs[3, i].set_title(r"\textbf{Distribution of Residuals}")
            axs[4, i].set_title(r"\textbf{Distribution of Input Spike Count}")

        # Place text above the first row of plots indicating the width factor
        axs[0, i].text(
            0.5,
            1.25,
            f"Tuning Curve Width = {width_factor}",
            transform=axs[0, i].transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.show()


if __name__ == "__main__":
    plot_encoding_scheme()
    compute_error_floor()
