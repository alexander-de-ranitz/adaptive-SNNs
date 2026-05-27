from abc import ABC, abstractmethod

import equinox as eqx


class AbstractEnvironmentState(eqx.Module):
    """State of the environment process."""

    pass


class AbstractEnvironment(ABC, eqx.Module):
    @property
    @abstractmethod
    def initial(self):
        pass

    @property
    @abstractmethod
    def noise_shape(self):
        pass

    @abstractmethod
    def drift(self, t, x, args, env_input):
        pass

    @abstractmethod
    def diffusion(self, t, x, args):
        pass

    @abstractmethod
    def terms(self, key):
        pass

    def update(self, t, x, args, env_input):
        """Optional update function for applying non-differentiable updates to the environment state."""
        pass

    def pre_step_update(self, t, x, args):
        """Optional function to apply updates to the environment state before computing the drift/diffusion."""
        return x
