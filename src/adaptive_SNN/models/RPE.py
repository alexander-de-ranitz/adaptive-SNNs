from abc import ABC, abstractmethod

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class RPEState(eqx.Module):
    RPE: Array


class AbstractRPEModel(ABC, eqx.Module):
    @property
    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def drift(self, t, x, args):
        pass

    @abstractmethod
    def diffusion(self, t, x, args):
        pass

    @property
    @abstractmethod
    def noise_shape(self):
        pass

    @abstractmethod
    def terms(self, key):
        pass

    @abstractmethod
    def update(self, t, x, args):
        """Apply non-differential updates"""
        pass


class UpdateAndDecayRPEModel(AbstractRPEModel):
    """RPE model that integrates RPE updates and decays them over time"""

    tau_RPE: float = 0.1

    @property
    def initial(self):
        return RPEState(RPE=jnp.zeros((1,)))

    @property
    def noise_shape(self):
        return RPEState(RPE=None)

    def diffusion(self, t, x, args):
        return RPEState(
            RPE=DefaultIfNone(
                default=jnp.zeros_like(x.RPE),
                else_do=ElementWiseMul(jnp.zeros_like(x.RPE)),
            )
        )

    def drift(self, t, x, args):
        return RPEState(RPE=-x.RPE / self.tau_RPE)  # Exponential decay of RPE over time

    def update(self, t, x, args):
        RPE_update = args.get(
            "RPE_update", None
        )  # RPE update computed based on current state and environment
        if RPE_update is None:
            raise ValueError(
                "UpdateAndDecayRPEModel requires 'RPE_update' in args for update computation."
            )
        return RPEState(
            RPE=x.RPE + RPE_update
        )  # Update the reward by adding the RPE update

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class InstantRPEModel(AbstractRPEModel):
    """RPE model that instantaneously sets the RPE  based on the current state and environment, without any integration over time"""

    @property
    def initial(self):
        return RPEState(RPE=jnp.zeros((1,)))

    @property
    def noise_shape(self):
        return RPEState(RPE=None)

    def diffusion(self, t, x, args):
        return RPEState(
            RPE=DefaultIfNone(
                default=jnp.zeros_like(x.RPE),
                else_do=ElementWiseMul(jnp.zeros_like(x.RPE)),
            )
        )

    def drift(self, t, x, args):
        return RPEState(
            RPE=jnp.zeros_like(x.RPE)
        )  # No drift, RPE is determined solely by updates

    def update(self, t, x, args):
        RPE_update = args.get(
            "RPE_update", None
        )  # RPE update computed based on current state and environment
        if RPE_update is None:
            raise ValueError(
                "InstantRPEModel requires 'RPE_update' in args for update computation."
            )
        return RPEState(
            RPE=RPE_update
        )  # Instantaneously set RPE to the computed update

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class BiphasicRPEState(RPEState):
    RPE: Array
    internal_state: Array  # Additional state variable to track the internal dynamics of the biphasic response


class BiphasicRPEModel(AbstractRPEModel):
    """RPE model that produces a biphasic response to reward prediction errors, with an initial positive phase followed by a negative tail"""

    time_constants: Array = eqx.field(
        default_factory=lambda: jnp.array([0.2, 0.3])
    )  # Time constants for the positive and negative phases of the response
    amplitudes: Array  # Amplitudes for the positive and negative phases, computed based on time constants to ensure zero integral of the response

    def __init__(self, time_constants=None, scale=1.0):
        if time_constants is not None:
            self.time_constants = time_constants
        assert self.time_constants.shape == (2,), (
            "time_constants must be an array of shape (2,)"
        )
        assert self.time_constants[0] < self.time_constants[1], (
            "time_constants[0] must be smaller than time_constants[1] for a biphasic response with an initial positive phase followed by a negative tail"
        )
        # Amplitudes for the positive and negative phases of the response. Must fulfill: amplitude[0] = amplitude[1] * time_constants[1] / time_constants[0] to ensure integral is 0
        self.amplitudes = jnp.array(
            [self.time_constants[1] / self.time_constants[0], 1.0]
        )
        self.amplitudes = (
            scale * self.amplitudes / (self.amplitudes[0] - self.amplitudes[1])
        )  # Normalize such that the maximum is equal to the specified scale

    @property
    def initial(self):
        return BiphasicRPEState(RPE=jnp.zeros((1,)), internal_state=jnp.zeros((2,)))

    @property
    def noise_shape(self):
        return jax.tree.map(
            lambda arr: None, self.initial
        )  # No noise in either RPE or internal state

    def diffusion(self, t, x, args):
        return MixedPyTreeOperator(
            jax.tree.map(
                lambda arr: DefaultIfNone(
                    default=jnp.zeros_like(arr),
                    else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=default_float)),
                ),
                x,
            )
        )

    def drift(self, t, x: BiphasicRPEState, args):
        internal_state = x.internal_state
        internal_state_drift = -internal_state * 1 / self.time_constants
        return BiphasicRPEState(
            RPE=jnp.zeros_like(x.RPE), internal_state=internal_state_drift
        )

    def update(self, t, x: BiphasicRPEState, args):
        RPE_update = args.get(
            "RPE_update", None
        )  # RPE update computed based on current state and environment
        if RPE_update is None:
            raise ValueError(
                "BiphasicRPEModel requires 'RPE_update' in args for update computation."
            )
        internal_state = x.internal_state + self.amplitudes * RPE_update
        return BiphasicRPEState(
            RPE=jnp.array(
                [internal_state[0] - internal_state[1]]
            ),  # Biphasic response is the difference between the two internal states
            internal_state=internal_state,
        )  # Update the reward by adding the RPE update

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
