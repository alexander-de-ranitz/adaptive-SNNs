from abc import ABC, abstractmethod

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


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

    @property
    def initial(self):
        return jnp.zeros((1,))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def drift(self, t, x, args):
        tau_RPE = args.get("tau_RPE", None)  # Time constant for RPE integration
        if tau_RPE is None:
            raise ValueError(
                "UpdateAndDecayRPEModel requires 'tau_RPE' in args for drift computation."
            )
        return -x / tau_RPE  # Exponential decay of RPE over time

    def update(self, t, x, args):
        RPE_update_fn = args.get(
            "RPE_fn", None
        )  # Function to compute RPE update based on current state and environment
        if RPE_update_fn is None:
            raise ValueError(
                "UpdateAndDecayRPEModel requires 'RPE_fn' in args for update computation."
            )
        RPE_update = RPE_update_fn(t, x, args)  # Compute the RPE update
        return x + RPE_update  # Update the reward by adding the RPE update

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
        return jnp.zeros((1,))

    @property
    def noise_shape(self):
        return None

    def diffusion(self, t, x, args):
        return DefaultIfNone(
            default=jnp.zeros_like(x), else_do=ElementWiseMul(jnp.zeros_like(x))
        )

    def drift(self, t, x, args):
        return jnp.zeros_like(x)  # No drift, RPE is determined solely by updates

    def update(self, t, x, args):
        RPE_fn = args.get(
            "RPE_fn", None
        )  # Function to compute RPE update based on current state and environment
        if RPE_fn is None:
            raise ValueError(
                "InstantRPEModel requires 'RPE_fn' in args for update computation."
            )
        RPE = RPE_fn(t, x, args)  # Compute the RPE
        return RPE

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
