from diffrax import EulerHeun, SaveAt
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from adaptive_SNN.models.environments import ExternalController, PendulumEnvironment
from adaptive_SNN.solver import solve_ODE


def plot_optimal_control():
    key = jr.PRNGKey(10)
    env = PendulumEnvironment(rate=1.0, key=key)
    optimal_control = lambda t, x, args: -env.control_gain @ x[:2].reshape((2, 1))

    for i in range(1):
        model = ExternalController(
            PendulumEnvironment(
                rate=1.0,
                key=jr.fold_in(key, i),
                initial_angle_range=(-0.5, 0.5),
                initial_angular_velocity_range=(-0.0, 0.0),
                max_allowed_angle=0.6,
                max_allowed_angular_velocity=100.0,
            )
        )
        save_at = SaveAt(ts=jnp.arange(0.0, 10.0, 0.01), t0=True, t1=True)
        args = {
            "get_env_input": optimal_control,
            "episode_end_fn": lambda t, x, args: jnp.any(
                jnp.abs(x)
                > jnp.array(
                    [
                        model.environment.max_allowed_angle,
                        model.environment.max_allowed_angular_velocity,
                        model.environment.max_episode_time,
                    ]
                )
            ),
        }
        sol = solve_ODE(
            model,
            solver=EulerHeun(),
            t0=0.0,
            t1=3.0,
            dt0=1e-4,
            y0=model.initial,
            save_at=save_at,
            args=args,
        )
        control = jnp.array(
            [optimal_control(t, y, args) for t, y in zip(sol.ts, sol.ys)]
        )
        angle, ang_vel = sol.ys[:, 0], sol.ys[:, 1]
        ts = sol.ts
        plt.subplot(3, 1, 1)
        plt.plot(ts, angle, c="darkgreen")

        plt.subplot(3, 1, 2)
        plt.plot(ts, ang_vel, c="darkgreen")

        plt.subplot(3, 1, 3)
        plt.plot(ts, control.squeeze(), c="darkgreen")
        plt.plot(ts, sol.ys[:, 2], c="orange")
    plt.xlabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    plot_optimal_control()
