import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import SaveAt
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from adaptive_snn.models.agent_env_system import SystemState
from adaptive_snn.simulation_configs.pendulum_config import create_pendulum_config
from adaptive_snn.utils.runner import run_simulation


def find_best_decoder(X_train, y_train, X_test, y_test):
    """Fit linear decoders with different regularization strengths and select the best one based on test R²."""
    # print shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    alphas = np.logspace(-3, 6, 10, base=10.0)
    best_r2 = -jnp.inf
    best_model = None
    r2_values_test = []
    r2_values_train = []
    for alpha in alphas:
        print(f"Testing alpha={alpha}...")
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        r2_values_test.append(r2_test)
        y_pred_train = model.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        r2_values_train.append(r2_train)
        if r2_test > best_r2:
            best_r2 = r2_test
            best_model = model
    plt.plot(alphas, r2_values_test, label="Test R²", marker="o")
    plt.plot(alphas, r2_values_train, label="Train R²", marker="o")
    plt.xscale("log")
    plt.xlabel("Alpha (Regularization Strength)")
    plt.ylabel("R² Score")
    plt.title("R² Score vs Regularization Strength")
    plt.legend()
    plt.show()
    return best_model


def test_state_readout():
    key = jr.PRNGKey(0)
    N_neurons = 100

    cfg = create_pendulum_config(N_neurons=N_neurons, key=key)
    env = cfg.environment_model(**cfg.environment_kwargs)
    optimal_control = env.control_gain
    cfg.t1 = 50.0

    def save(t, x: SystemState, args):
        return (
            x.agent_state.network_state.filtered_spike_trains,  # (N_neurons,)
            x.environment_state,  # (2,) — [angle, angular_velocity]
            x.agent_output,  # (1,) — torque
            x.reward_signal,
            x.agent_state.reward_predictor_state.value,  # (1,) — predicted reward
        )

    cfg.save_at = SaveAt(ts=jnp.linspace(cfg.t0, cfg.t1, int(20 * cfg.t1)), fn=save)

    print(f"Running simulation (t1={cfg.t1}s, N_neurons={N_neurons})...")
    cfg.save_file = f"results/pendulum_encoding_test_{N_neurons}_neurons.npz"
    sol, model = run_simulation(cfg, save_results=True, overwrite=True)

    filtered_spike_train, env_states, agent_outputs, reward, predicted_reward = (
        sol.ys
    )  # (T, N_neurons), (T, 2), (T, 1), (T, N_inputs), (T, 1)

    fraction_train = 0.8
    split_idx = int(fraction_train * filtered_spike_train.shape[0])
    train_spike_trains = filtered_spike_train[:split_idx]
    train_reward = reward[:split_idx]

    test_ts = sol.ts[split_idx:]
    test_spike_trains = filtered_spike_train[split_idx:]
    test_env_states = env_states[split_idx:]
    test_reward = reward[split_idx:]

    clf = find_best_decoder(
        X_train=train_spike_trains,
        y_train=train_reward,
        X_test=test_spike_trains,
        y_test=test_reward,
    )

    optimal_control_signal = -optimal_control @ test_env_states.T

    # Evaluate the best decoder on the test set
    y_pred_test = clf.predict(test_spike_trains)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(test_ts, test_reward, label="True reward")
    plt.plot(test_ts, y_pred_test, label="Predicted reward", linestyle="--")
    plt.plot(
        test_ts,
        predicted_reward.squeeze()[split_idx:],
        label="RLS Predicted Reward",
        linestyle="--",
    )
    plt.xlabel("Time (s)")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(test_ts, test_env_states[:, 0], label="True Angle")
    plt.plot(test_ts, test_env_states[:, 1], label="True Angular Velocity")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(
        test_ts,
        optimal_control_signal.squeeze(),
        label="Optimal Control Signal",
        linestyle="--",
    )
    plt.plot(
        test_ts,
        agent_outputs.squeeze()[split_idx:],
        label="Agent Output",
        linestyle="--",
    )
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_state_readout()
