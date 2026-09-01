import diffrax as dfx
import equinox as eqx
from jax import numpy as jnp

from adaptive_SNN.models.networks.base import AbstractLIFNetwork


class NetworkHandler(eqx.Module):
    """Wrapper to run the network standalone, with external RPE and input spikes."""

    network: AbstractLIFNetwork

    @property
    def initial(self):
        return self.network.initial

    def drift(self, t, x, args):
        RPE = args.get("RPE_fn", lambda t, x, args: 0.0)(t, x, args)
        return self.network.drift(t, x, args, RPE)

    def diffusion(self, t, x, args):
        return self.network.diffusion(t, x, args)

    def pre_step_update(self, t, x, args):
        input_spikes = args.get(
            "input_spike_fn", lambda t, x, args: jnp.zeros((self.network.N_neurons,))
        )(t, x, args)
        return self.network.pre_step_update(t, x, args, input_spikes)

    def update(self, t, x, args):
        return self.network.update(t, x, args)

    @property
    def noise_shape(self):
        return self.network.noise_shape

    def terms(self, key):
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift),
            dfx.ControlTerm(
                self.diffusion,
                dfx.UnsafeBrownianPath(
                    shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
                ),
            ),
        )
