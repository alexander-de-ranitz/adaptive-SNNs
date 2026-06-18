import diffrax as dfx
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_SNN.models.reward_prediction import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_SNN.utils.operators import DefaultIfNone, ElementWiseMul

default_float = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


class CriticPrediction(RewardPrediction):
    input_features: Array
    input_features_prev: Array
    weights: Array
    previous_value: Array  # Store the previous reward prediction for RPE computation


class LinearReadoutCritic(AbstractRewardPredictor):
    """A simple spiking critic that uses a single feed-forward layer to predict the reward signal from the state-encoding population."""

    scale: float = 1.0
    tau_filtered_input: float = 0.05  # Time constant for filtering the input features
    input_dim: int = 1  # Dimension of the input features for reward prediction

    @property
    def initial(self):
        return CriticPrediction(
            value=jnp.zeros((1,)),
            previous_value=jnp.zeros((1,)),
            input_features=jnp.zeros((self.input_dim,)),
            input_features_prev=jnp.zeros((self.input_dim,)),
            weights=jnp.zeros((self.input_dim + 1,)),  # + 1 for bias term
        )

    @property
    def noise_shape(self):
        return CriticPrediction(
            value=None,
            input_features=None,
            input_features_prev=None,
            weights=None,
            previous_value=None,
        )

    def pre_step_update(
        self,
        t,
        state: CriticPrediction,
        args,
        reward,
        network_state,
        input_spikes,
        env_state,
    ):
        new_value = jnp.atleast_1d(
            state.weights @ jnp.concatenate([state.input_features, jnp.array([1.0])])
        )
        new_features = state.input_features + input_spikes
        return CriticPrediction(
            value=new_value,
            previous_value=state.value,
            input_features=new_features,
            input_features_prev=state.input_features,
            weights=state.weights,
        )

    def drift(self, t, state: CriticPrediction, args, reward, RPE) -> CriticPrediction:
        # TD(0) update rule for the critic weights
        d_W = (
            args["get_critic_lr"](t, state, args)
            * RPE
            * jnp.concatenate([state.input_features_prev, jnp.array([1.0])])
        )

        # features exponentially decay towards zero with time constant tau_filtered_input
        d_features = -state.input_features / self.tau_filtered_input

        drift = jax.tree.map(lambda arr: jnp.zeros_like(arr, dtype=arr.dtype), state)
        drift = eqx.tree_at(lambda s: s.weights, drift, d_W)
        drift = eqx.tree_at(lambda s: s.input_features, drift, d_features)
        return drift

    def diffusion(self, t, state: CriticPrediction, args) -> CriticPrediction:
        return jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr),
                else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=default_float)),
            ),
            state,
        )

    def update(self, t, state: CriticPrediction, args) -> CriticPrediction:
        return state

    def reset(self, t, state: CriticPrediction, args) -> CriticPrediction:
        """Reset everything but the weights at the end of an episode."""
        return CriticPrediction(
            value=jnp.zeros_like(state.value),
            previous_value=jnp.zeros_like(state.previous_value),
            input_features=jnp.zeros_like(state.input_features),
            input_features_prev=jnp.zeros_like(state.input_features_prev),
            weights=state.weights,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
