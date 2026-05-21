"""LearningAgent wrapper — couples a NoisyNetwork / PerSynapseNoisyNetwork with
a scalar filtered RPE state and a Poisson-noise RPE source.

This is a minimal alternative to the full Agent / AgentEnvSystem stack — built
so the overnight learning-dynamics experiments can run η > 0 closed-loop with
δ_r(t) = exp_filter(spike_diff + reward_noise, τ_RPE) without modifying any
upstream classes.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from adaptive_SNN.models.networks.base import NeuronModelABC
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul, MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class LearningAgentState(eqx.Module):
    network_state: object       # NoisyNetworkState or PerSynapseNoisyNetworkState
    rpe: Array                  # scalar filtered RPE


class LearningAgent(NeuronModelABC):
    """Wraps a noisy network and evolves a filtered scalar RPE state.

    The RPE drift is
        d(RPE)/dt = (-RPE + spike_diff + reward_noise_drive) / tau_RPE
    where spike_diff = S[0] - S[1] (noisy minus noiseless), and
    reward_noise_drive is sampled per step from a Poisson process at
    `rpe_noise_rate` Hz with Gaussian jump magnitude std=`rpe_noise_std`.

    The filtered RPE is exposed to the wrapped network as args["RPE"], so
    eligibility and OUA cells consume it via their standard
    compute_weight_updates path.
    """

    noisy_network: NeuronModelABC
    tau_RPE: float = 0.1
    rpe_noise_rate: float = 1.0
    rpe_noise_std: float = 1.0
    noisy_idx: int = 0       # row of S corresponding to the noisy neuron
    noiseless_idx: int = 1   # row of S corresponding to the noiseless reference

    def __init__(
        self,
        noisy_network: NeuronModelABC,
        tau_RPE: float = 0.1,
        rpe_noise_rate: float = 1.0,
        rpe_noise_std: float = 1.0,
        noisy_idx: int = 0,
        noiseless_idx: int = 1,
    ):
        self.noisy_network = noisy_network
        self.tau_RPE = tau_RPE
        self.rpe_noise_rate = rpe_noise_rate
        self.rpe_noise_std = rpe_noise_std
        self.noisy_idx = noisy_idx
        self.noiseless_idx = noiseless_idx

    @property
    def initial(self):
        return LearningAgentState(
            network_state=self.noisy_network.initial,
            rpe=jnp.zeros((), dtype=default_float),
        )

    def _poisson_jump(self, t, rpe_noise_key):
        """Approximate Poisson-jump reward noise per timestep.

        Treats the inter-step interval as one Bernoulli trial with success
        probability `rate * dt`, where `dt` is inferred from the time
        increment. We rely on the simulator stepping at a fixed dt; we use
        a per-step key derived from time index.
        """
        rng = jr.fold_in(rpe_noise_key, jnp.asarray(jnp.rint(t * 1e4), dtype=jnp.int64))
        u = jr.uniform(rng)
        # Probability of a Poisson event per step:
        p_event = self.rpe_noise_rate * jnp.array(1e-4, dtype=default_float)
        jump = jnp.where(u < p_event, jr.normal(jr.fold_in(rng, 1)) * self.rpe_noise_std, 0.0)
        return jump

    def drift(self, t, state: LearningAgentState, args: dict):
        rpe_key = args.get("rpe_noise_key", jr.PRNGKey(0))
        # Inject the current filtered RPE so the wrapped network's
        # consolidation rule consumes it.
        args = dict(args)
        args["RPE"] = state.rpe

        # Wrapped network drift (handles its own per-synapse noise routing).
        net_drift = self.noisy_network.drift(t, state.network_state, args)

        # Task-RPE drive: user-callable via args["task_rpe_fn"] takes priority over
        # the built-in `S[noisy_idx] - S[noiseless_idx]` fallback (used for SST).
        inner = state.network_state.network_state
        task_rpe_fn = args.get("task_rpe_fn", None)
        if task_rpe_fn is None:
            spike_diff = inner.S[self.noisy_idx] - inner.S[self.noiseless_idx]
            task_drive = spike_diff / jnp.array(1e-4, dtype=default_float)
        else:
            # Convention: task_rpe_fn returns a *spike-rate-scale* drive
            # (e.g., +/-1 with a 1/dt-equivalent magnitude already applied).
            task_drive = task_rpe_fn(t, state, args)

        # Poisson reward noise.
        rpe_noise_jump = self._poisson_jump(t, rpe_key)
        rpe_noise_drive = rpe_noise_jump / jnp.array(1e-4, dtype=default_float)

        d_rpe = (-state.rpe + task_drive + rpe_noise_drive) / self.tau_RPE
        return LearningAgentState(network_state=net_drift, rpe=d_rpe)

    def diffusion(self, t, state: LearningAgentState, args: dict):
        net_diff = self.noisy_network.diffusion(t, state.network_state, args)
        return MixedPyTreeOperator(
            LearningAgentState(
                network_state=net_diff,
                rpe=DefaultIfNone(
                    default=jnp.zeros_like(state.rpe),
                    else_do=ElementWiseMul(jnp.zeros_like(state.rpe, dtype=default_float)),
                ),
            )
        )

    @property
    def noise_shape(self):
        return LearningAgentState(
            network_state=self.noisy_network.noise_shape,
            rpe=None,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, state: LearningAgentState, args):
        # Inject current RPE into args so the wrapped network's `update` sees it.
        args = dict(args)
        args["RPE"] = state.rpe
        new_net = self.noisy_network.update(t, state.network_state, args)
        return LearningAgentState(network_state=new_net, rpe=state.rpe)
