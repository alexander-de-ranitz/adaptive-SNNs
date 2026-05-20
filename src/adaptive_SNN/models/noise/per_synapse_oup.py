"""Per-synapse Ornstein-Uhlenbeck noise process.

The state is a 2D matrix of shape (N_neurons, N_neurons + N_inputs) carrying
one independent OU process per (postsynaptic, presynaptic) pair. Diffusion is
restricted to excitatory synapses via the `excitatory_mask`. Drift is
element-wise -(x - mean)/tau.

This is the per-synapse endpoint (alpha = 1) of the exploration-geometry
family in [[unified-model-and-merger-plan-20260518]] Eq. 2.1b. The companion
per-neuron OU process is `OUP` / `NeuralNoiseOUP` in the same package.
"""

from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.noise.base import NoiseModelABC
from adaptive_SNN.utils.operators import ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class PerSynapseOUP(NoiseModelABC):
    """Per-synapse Ornstein-Uhlenbeck noise.

    State shape: (N_neurons, N_neurons + N_inputs).
    Drift: -(x - mean) / tau
    Diffusion: element-wise diagonal, scaled by sqrt(2/tau) * noise_std *
               excitatory_mask[None, :]. Inhibitory synapses receive zero
               diffusion (they stay at the mean throughout).

    The Wiener path has shape (N_neurons, N_neurons + N_inputs) with
    independent increments per (i, j).

    Stationary marginal variance per excitatory synapse: noise_std^2 * tau / 2.
    """

    tau: float | Array = 6e-3
    noise_std: float | Array = 0.0
    mean: float | Array = 0.0
    N_neurons: int = 1
    N_inputs: int = 0
    excitatory_mask: Array = None  # type: ignore[assignment]

    def __init__(
        self,
        N_neurons: int,
        N_inputs: int,
        excitatory_mask: Array,
        tau: float | Array = 6e-3,
        noise_std: float | Array = 0.0,
        mean: float | Array = 0.0,
    ):
        if excitatory_mask.shape != (N_neurons + N_inputs,):
            raise ValueError(
                f"excitatory_mask must have shape ({N_neurons + N_inputs},), "
                f"got {excitatory_mask.shape}"
            )
        self.N_neurons = N_neurons
        self.N_inputs = N_inputs
        self.excitatory_mask = excitatory_mask.astype(default_float)
        self.tau = tau
        self.noise_std = noise_std
        self.mean = mean

    @property
    def initial(self):
        return jnp.full(
            (self.N_neurons, self.N_neurons + self.N_inputs),
            self.mean,
            dtype=default_float,
        )

    def drift(self, t, x, args):
        return -1.0 / self.tau * (x - self.mean)

    def diffusion(self, t, x, args):
        # The args value takes precedence (state-dependent calibration via
        # PerSynapseNoisyNetwork.compute_desired_noise_std). Falls back to
        # the constructor-level noise_std.
        noise_std = (
            self.noise_std
            if args is None
            else args.get("per_synapse_noise_std", self.noise_std)
        )

        # Excitatory mask broadcast across postsynaptic neurons (rows).
        # Result shape: (N_neurons, N_neurons + N_inputs).
        scale = (
            jnp.sqrt(2.0 / self.tau)
            * noise_std
            * self.excitatory_mask[None, :].astype(default_float)
        )
        # Broadcast to full state shape if scalar/1D input.
        scale = jnp.broadcast_to(
            scale, (self.N_neurons, self.N_neurons + self.N_inputs)
        ).astype(default_float)
        return ElementWiseMul(scale)

    def update(self, t, x, args):
        # Ensure inhibitory synapses remain pinned at the mean (defensive;
        # diffusion is zero there but drift retains them at mean already).
        return x

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(
            shape=(self.N_neurons, self.N_neurons + self.N_inputs),
            dtype=default_float,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
