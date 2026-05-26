from diffrax import EulerHeun, SaveAt
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from scipy.linalg import solve_continuous_are

from adaptive_SNN.models.environments.pendulum import Pendulum
from adaptive_SNN.solver import solve_ODE


def compute_optimal_controller(env: Pendulum):
    A = jnp.array([[0, 1], [env.g, 0]])
    B = jnp.array([[0], [env.rate]])
    Q = jnp.eye(2)  # State cost matrix
    R = jnp.eye(1)  # Control cost matrix
    S = solve_continuous_are(A, B, Q, R)
    K = jnp.linalg.inv(R) @ B.T @ S

    def optimal_control(t, x, args):
        u = -K @ x
        return u

    return optimal_control


def main():
    key = jr.PRNGKey(0)
    env = Pendulum(rate=1.0, key=key)
    optimal_control = compute_optimal_controller(env)
    for i in range(10):
        env = Pendulum(
            rate=0.5, key=jr.fold_in(key, i), min_max_angle_initial=(-jnp.pi, jnp.pi)
        )
        print("Initial state:", env.initial)
        save_at = SaveAt(ts=jnp.arange(0.0, 10.0, 0.01), t0=True, t1=True)
        args = {"get_env_input": optimal_control}
        sol = solve_ODE(
            env,
            solver=EulerHeun(),
            t0=0.0,
            t1=10.0,
            dt0=1e-4,
            y0=env.initial,
            save_at=save_at,
            args=args,
        )
        angle, ang_vel = sol.ys[:, 0], sol.ys[:, 1]
        print("Final state:", sol.ys[-1])
        ts = sol.ts
        plt.subplot(2, 1, 1)
        plt.plot(ts, angle, c="darkgreen")

        plt.subplot(2, 1, 2)
        plt.plot(ts, ang_vel, c="darkgreen")
    plt.xlabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    main()
