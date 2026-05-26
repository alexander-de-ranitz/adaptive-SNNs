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

from adaptive_SNN.models.environments.base import AbstractEnvironment
from adaptive_SNN.models.environments.spike_rate import SpikeRateEnvironment
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class Pendulum(AbstractEnvironment):
    """Environment model representing a pendulum balancing system.

    The state of the pendulum is represented as a 2-dimensional vector:
        - angle: The angle of the pendulum from the vertical (0 radians means upright).
        - angular velocity: The rate of change of the pendulum's angle.

    The angle is measured from the vertical, where an angle of 0 indicates the pendulum is pointing straight upwards.
    """

    dim: int = 2  # State dimension: [angle, angular velocity]
    rate: float = 1.0  # Rate at which the environment responds to input
    g: float = 10.0  # Gravitational constant
    min_max_angle_initial: tuple = (-0.1, 0.1)  # Range of initial angles (in radians)
    key: Array = eqx.field(
        default_factory=lambda: jr.PRNGKey(0)
    )  # Random key for initialization
    Q: Array = eqx.field(
        default_factory=lambda: jnp.eye(2)
    )  # State cost matrix for LQR
    R: Array = eqx.field(
        default_factory=lambda: jnp.eye(1)
    )  # Control cost matrix for LQR

    control_gain: Array = (
        None  # Optimal control gain matrix, to be computed based on system dynamics
    )
    cost_to_go_matrix: Array = (
        None  # Cost-to-go matrix, to be computed based on system dynamics
    )

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
            minval=self.min_max_angle_initial[0],
            maxval=self.min_max_angle_initial[1],
        )
        return jnp.array([angle, 0.0], dtype=default_float)

    @property
    def noise_shape(self):
        return None

    def drift(self, t, x, args):
        if args is None or "get_env_input" not in args:
            raise ValueError(
                "Pendulum requires 'get_env_input' in args for drift computation."
            )

        # Compute the torque based on the input from the agent
        torque = self.rate * jnp.squeeze(args["get_env_input"](t, x, args))

        # Compute the change in state due to dynamics
        gravity_effect = (
            self.g * jnp.sin(x[0])
        )  # Gravitational effect on the pendulum, assuming length = 1 and mass = 1 for simplicity
        total_torque = torque + gravity_effect

        dxdt = jnp.array(
            [
                x[1],  # d(angle)/dt = angular velocity
                total_torque,  # d(angular velocity)/dt = total torque
            ],
            dtype=default_float,
        )
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

    def update(self, t, x, args):
        return x


class PendulumEnvState(eqx.Module):
    pendulum_state: Array
    spike_rate_state: Array


class PendulumControlEnv(AbstractEnvironment):
    """Pendulum environment with spike rate representation for output and reward prediction"""

    spike_rate_env: SpikeRateEnvironment
    pendulum: Pendulum

    @property
    def initial(self):
        return PendulumEnvState(
            pendulum_state=self.pendulum.initial,
            spike_rate_state=self.spike_rate_env.initial,
        )

    @property
    def noise_shape(self):
        return PendulumEnvState(
            pendulum_state=self.pendulum.noise_shape,
            spike_rate_state=self.spike_rate_env.noise_shape,
        )

    def drift(self, t, state: PendulumEnvState, args):
        pendulum_drift = self.pendulum.drift(t, state.pendulum_state, args)
        spike_rate_drift = self.spike_rate_env.drift(t, state.spike_rate_state, args)
        return PendulumEnvState(
            pendulum_state=pendulum_drift, spike_rate_state=spike_rate_drift
        )

    def diffusion(self, t, state: PendulumEnvState, args):
        pendulum_diffusion = self.pendulum.diffusion(t, state.pendulum_state, args)
        spike_rate_diffusion = self.spike_rate_env.diffusion(
            t, state.spike_rate_state, args
        )
        return PendulumEnvState(
            pendulum_state=pendulum_diffusion, spike_rate_state=spike_rate_diffusion
        )

    def update(self, t, state: PendulumEnvState, args):
        pendulum_update = self.pendulum.update(t, state.pendulum_state, args)
        spike_rate_update = self.spike_rate_env.update(t, state.spike_rate_state, args)
        return PendulumEnvState(
            pendulum_state=pendulum_update, spike_rate_state=spike_rate_update
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
