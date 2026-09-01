import jax

jax.config.update(
    "jax_enable_x64", True
)  # Enable 64-bit precision for better numerical stability
import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jax import random as jr
from jaxtyping import Array
from scipy.linalg import solve_continuous_are

from adaptive_SNN.models.environments import AbstractEnvironment
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class PendulumEnvironment(AbstractEnvironment):
    """Environment model representing a pendulum balancing system.

    The state of the pendulum is represented as a 3-dimensional vector:
        - angle: The angle of the pendulum from the vertical (0 radians means upright).
        - angular velocity: The rate of change of the pendulum's angle.
        - time: Time since start of the episode (needed to reset the environment after certain time)

    The angle is measured from the vertical, where an angle of 0 indicates the pendulum is pointing straight upwards.
    """

    # fmt: off
    dim: int = 3  # State dimension: [angle, angular velocity, time]
    rate: float = 1.0  # Rate at which the environment responds to input
    g: float = 10.0  # Gravitational constant
    initial_angle_range: tuple = (-0.1, 0.1)  # Range of initial angles (in radians)
    initial_angular_velocity_range: tuple = (-0.0, 0.0)  # Range of initial angular velocities (in radians/s)
    max_allowed_angle: float = 0.5 # Maximum allowed angle before episode termination (in radians)
    max_allowed_angular_velocity: float = 1.5 # Maximum allowed angular velocity before episode termination (in radians/s)
    max_episode_time: float = 5.0 # Maximum allowed time for an episode before termination (in seconds)
    key: Array = eqx.field(default_factory=lambda: jr.PRNGKey(6758493)) # Random key for initialization
    Q: Array = eqx.field(default_factory=lambda: jnp.diag(jnp.array([1.0, 0.1]))) # State cost matrix for LQR
    R: Array = eqx.field(default_factory=lambda: 0.001 * jnp.eye(1)) # Control cost matrix for LQR
    control_gain: Array = None  # Optimal control gain matrix, to be computed based on system dynamics
    cost_to_go_matrix: Array = None  # Cost-to-go matrix, to be computed based on system dynamics
    # fmt: on

    def __post_init__(self):
        A = jnp.array([[0, 1], [self.g, 0]])
        B = jnp.array([[0], [self.rate]])
        Q = self.Q  # State cost matrix
        R = self.R  # Control cost matrix
        S = solve_continuous_are(A, B, Q, R)
        K = jnp.linalg.inv(R) @ B.T @ S

        # Store the optimal control gain and cost-to-go matrix for use in reward shaping and analysis
        self.control_gain = K
        self.cost_to_go_matrix = S

    @property
    def initial(self):
        angle = jax.random.uniform(
            self.key,
            shape=(),
            minval=self.initial_angle_range[0],
            maxval=self.initial_angle_range[1],
        )
        return jnp.array([angle, 0.0, 0.0], dtype=default_float)

    @property
    def noise_shape(self):
        return None

    def drift(self, t, x, args, env_input):
        # Compute the torque based on the input from the agent
        torque = self.rate * jnp.squeeze(env_input)

        # Compute the change in state due to dynamics
        gravity_effect = (
            self.g * jnp.sin(x[0])
        )  # Gravitational effect on the pendulum, assuming length = 1 and mass = 1 for simplicity
        total_torque = torque + gravity_effect

        dxdt = jnp.array(
            [
                x[1],  # d(angle)/dt = angular velocity
                total_torque,  # d(angular velocity)/dt = total torque
                1.0,  # d(time)/dt = 1 (time increases at a constant rate)
            ],
            dtype=default_float,
        )

        # During warmup time, the environment state does not evolve (dxdt = 0) (except for time, which continues to increase)
        in_warmup = args.get("env_warmup_fn", lambda t, x, args: False)(t, x, args)
        dxdt = dxdt.at[:2].set(dxdt[:2] * jnp.logical_not(in_warmup))
        return dxdt

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args, env_input=None):
        return x

    def reset(self, t, x, args):
        step_idx = jnp.asarray(
            jnp.rint((t) / args.get("dt", jnp.float64(1e-4))), dtype=jnp.int64
        )
        current_key = jr.fold_in(self.key, step_idx)
        ang_key, ang_vel_key = jr.split(current_key)
        angle = jax.random.uniform(
            ang_key,
            shape=(),
            minval=self.initial_angle_range[0],
            maxval=self.initial_angle_range[1],
        )
        angular_velocity = jax.random.uniform(
            ang_vel_key,
            shape=(),
            minval=self.initial_angular_velocity_range[0],
            maxval=self.initial_angular_velocity_range[1],
        )
        return jnp.array([angle, angular_velocity, 0.0], dtype=default_float)

    def reward_fn(self, t, x, args, agent_output):
        instantaneous_cost = x[:2].T @ self.Q @ x[:2] + jnp.atleast_1d(
            agent_output
        ).T @ self.R @ jnp.atleast_1d(agent_output)
        return jnp.atleast_1d(-instantaneous_cost)
