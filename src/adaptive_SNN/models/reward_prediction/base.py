from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class RewardPrediction(eqx.Module):
    value: Array  # Scalar array reward prediction


class AbstractRewardPredictor(ABC, eqx.Module):
    @property
    @abstractmethod
    def initial(self):
        pass

    @abstractmethod
    def drift(self, t, x, args, reward, network_state):
        pass

    @abstractmethod
    def diffusion(self, t, x, args):
        pass

    @abstractmethod
    def update(self, t, x, args):
        """Apply non-differential updates"""
        pass

    def pre_step_update(self, t, x, args, reward, network_state):
        """Apply any necessary updates to the state before computing the drift/diffusion."""
        return x

    @property
    @abstractmethod
    def noise_shape(self):
        pass

    @abstractmethod
    def terms(self, key):
        pass
