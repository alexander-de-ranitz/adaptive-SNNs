import diffrax as dfx
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.environments import (
    AbstractEnvironment,
    AbstractEnvironmentState,
)
from adaptive_SNN.models.networks import Agent, AgentState
from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


class SystemState(eqx.Module):
    agent_state: AgentState
    environment_state: AbstractEnvironmentState
    agent_output: Array
    reward_signal: Array


class AgentEnvSystem(eqx.Module):
    agent: Agent
    environment: AbstractEnvironment
    agent_output_shape: tuple[int, ...]

    @property
    def initial(self):
        return SystemState(
            agent_state=self.agent.initial,
            environment_state=self.environment.initial,
            agent_output=jnp.zeros(
                self.agent_output_shape, dtype=default_float
            ),  # Initial agent output
            reward_signal=jnp.zeros(1, dtype=default_float),  # Initial reward signal
        )

    def pre_step_update(self, t, x: SystemState, args):
        """Perform any necessary updates to the state before computing the drift/diffusion."""

        def _reset(t, x: SystemState, args):
            new_env_state = self.environment.reset(t, x.environment_state, args)
            new_agent_state = self.agent.reset(t, x.agent_state, args)
            return SystemState(
                agent_state=new_agent_state,
                environment_state=new_env_state,
                agent_output=jnp.zeros_like(x.agent_output),
                reward_signal=jnp.zeros_like(x.reward_signal),
            )

        # If we have an episode end function and it returns True, reset the environment and agent states
        x = jax.lax.cond(
            args.get("episode_end_fn", lambda t, x, args: False)(t, x, args),
            lambda: _reset(t, x, args),
            lambda: x,
        )

        # Compute agent output based on current agent state
        agent_output = args["network_output_fn"](
            t, x.agent_state, args, env_state=x.environment_state
        )

        input_spikes = args["input_spike_fn"](t, x, args)

        # Check if we are still in the warmup period based on the environment's internal time state
        env_in_warmup = args.get("env_warmup_fn", lambda t, x, args: False)(
            t, x.environment_state, args
        )

        # Update agent and environment states
        agent_state = self.agent.pre_step_update(
            t,
            x.agent_state,
            args,
            reward=x.reward_signal,
            env_state=x.environment_state,
            input_spikes=input_spikes,
            disable_RPE=env_in_warmup,
        )
        environment_state = self.environment.pre_step_update(
            t, x.environment_state, args
        )

        return SystemState(
            agent_state=agent_state,
            environment_state=environment_state,
            agent_output=agent_output,
            reward_signal=x.reward_signal,
        )

    def drift(self, t, x: SystemState, args: dict):
        """Compute deterministic time derivatives for the combined Agent-Environment system.

        The state consists of (agent_state, environment_state). The args dict
        must contain the necessary inputs for both the agent and environment models.

        Args:
            t: time
            x: (agent_state, env_state)
            args: dict containing keys required by both agent and environment models, including:
                - network_output_fn(t, agent_state, args, env_state) -> agent output
                - reward_fn(t, env_state, args) -> reward scalar
        Returns:
            (d_agent_state, d_env_state, reward_signal)
        """
        env_drift = self.environment.drift(
            t, x.environment_state, args, env_input=x.agent_output
        )
        agent_drift = self.agent.drift(t, x.agent_state, args, reward=x.reward_signal)
        agent_output_drift = jnp.zeros_like(x.agent_output)
        reward_signal_drift = jnp.zeros_like(x.reward_signal)

        return SystemState(
            agent_drift, env_drift, agent_output_drift, reward_signal_drift
        )

    def diffusion(self, t, x: SystemState, args):
        agent_diffusion = self.agent.diffusion(t, x.agent_state, args)
        env_diffusion = self.environment.diffusion(t, x.environment_state, args)

        # Reward is not subject to noise
        reward_diffusion = DefaultIfNone(
            default=jnp.zeros_like(x.reward_signal),
            else_do=ElementWiseMul(
                jnp.zeros_like(x.reward_signal, dtype=default_float)
            ),
        )

        # Agent output is not subject to noise
        agent_output_diffusion = DefaultIfNone(
            default=jnp.zeros_like(x.agent_output),
            else_do=ElementWiseMul(jnp.zeros_like(x.agent_output, dtype=default_float)),
        )
        return MixedPyTreeOperator(
            SystemState(
                agent_diffusion, env_diffusion, agent_output_diffusion, reward_diffusion
            )
        )

    @property
    def noise_shape(self):
        return SystemState(
            agent_state=self.agent.noise_shape,
            environment_state=self.environment.noise_shape,
            agent_output=None,
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
        # Update env state
        new_env_state = self.environment.update(
            t, x.environment_state, args, env_input=x.agent_output
        )
        new_state = eqx.tree_at(lambda x: x.environment_state, x, new_env_state)

        # Get reward signal
        reward = args["reward_fn"](t, new_state, args)
        new_state = eqx.tree_at(lambda x: x.reward_signal, new_state, reward)

        # Update agent state based on new reward signal
        new_agent_state = self.agent.update(t, x.agent_state, args, reward=reward)
        new_state = eqx.tree_at(lambda x: x.agent_state, new_state, new_agent_state)

        return new_state
