import diffrax as dfx
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.environments.base import EnvironmentABC
from adaptive_SNN.models.networks.agent import Agent, AgentState
from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


class SystemState(eqx.Module):
    agent_state: AgentState
    environment_state: Array
    reward_signal: Array


class AgentEnvSystem(eqx.Module):
    agent: Agent
    environment: EnvironmentABC

    def __init__(
        self,
        agent: Agent,
        environment: EnvironmentABC,
    ):
        self.agent = agent
        self.environment = environment

    @property
    def initial(self):
        return SystemState(
            self.agent.initial,
            self.environment.initial,
            jnp.array(0.0),  # Initial reward signal
        )

    def drift(self, t, x: SystemState, args: dict):
        """Compute deterministic time derivatives for the combined Agent-Environment system.

        The state consists of (agent_state, environment_state). The args dict
        must contain the necessary inputs for both the agent and environment models.

        Args:
            t: time
            x: (agent_state, env_state)
            args: dict containing keys required by both agent and environment models, including:
                - network_output_fn(t, agent_state, args) -> agent output
                - reward_fn(t, env_state, args) -> reward scalar
        Returns:
            (d_agent_state, d_env_state, reward_signal)
        """

        (agent_state, env_state) = x.agent_state, x.environment_state

        agent_output = args["network_output_fn"](t, agent_state, args)
        reward = args["reward_fn"](t, x, args)
        args.update(
            {
                "env_state": env_state,
                "get_env_input": lambda t, x, args: agent_output,
                "reward": reward,
            }
        )

        env_drift = self.environment.drift(t, env_state, args)
        agent_drift = self.agent.drift(t, agent_state, args)

        return SystemState(agent_drift, env_drift, jnp.array(0.0))

    def diffusion(self, t, x: SystemState, args):
        (agent_state, env_state) = x.agent_state, x.environment_state
        agent_diffusion = self.agent.diffusion(t, agent_state, args)
        env_diffusion = self.environment.diffusion(t, env_state, args)

        # Reward is not subject to noise
        reward_diffusion = DefaultIfNone(
            default=jnp.zeros_like(x.reward_signal),
            else_do=ElementWiseMul(
                jnp.zeros_like(x.reward_signal, dtype=default_float)
            ),
        )
        return MixedPyTreeOperator(
            SystemState(agent_diffusion, env_diffusion, reward_diffusion)
        )

    @property
    def noise_shape(self):
        return SystemState(
            agent_state=self.agent.noise_shape,
            environment_state=self.environment.noise_shape,
            reward_signal=None,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: SystemState, args: dict):
        (agent_state, env_state) = x.agent_state, x.environment_state
        new_agent_state = self.agent.update(t, agent_state, args)
        new_env_state = self.environment.update(t, env_state, args)
        reward_signal = args["reward"]
        return SystemState(new_agent_state, new_env_state, reward_signal)
