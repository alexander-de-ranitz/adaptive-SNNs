import diffrax as dfx
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array

from adaptive_snn.models.reward_prediction import (
    AbstractRewardPredictor,
    RewardPrediction,
)
from adaptive_snn.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


class CriticPrediction(RewardPrediction):
    features: Array
    features_prev: Array
    features_prev_prev: Array
    weights: Array
    previous_value: Array  # Store the previous reward prediction for RPE computation


class LinearReadoutCritic(AbstractRewardPredictor):
    """A simple spiking critic that uses a single feed-forward layer to predict the reward signal from the state-encoding population."""

    scale: float = 1.0
    tau_filtered_input: float = 0.05  # Time constant for filtering the input features
    input_dim: int = 1  # Dimension of the input features for reward prediction
    pre_trained_weights: str = None
    use_bias: bool = True  # Whether to include a bias term in the linear readout

    @property
    def initial(self):
        if self.pre_trained_weights is not None:
            weights = jnp.load(self.pre_trained_weights)
        else:
            weights = jnp.zeros(
                (self.input_dim + int(self.use_bias),)
            )  # + 1 for bias term if using

        return CriticPrediction(
            value=jnp.zeros((1,)),
            previous_value=jnp.zeros((1,)),
            features=jnp.zeros((self.input_dim,)),
            features_prev=jnp.zeros((self.input_dim,)),
            features_prev_prev=jnp.zeros((self.input_dim,)),
            weights=weights,
        )

    @property
    def noise_shape(self):
        return CriticPrediction(
            value=None,
            features=None,
            features_prev=None,
            features_prev_prev=None,
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
        new_features = state.features + args["critic_input_fn"](
            t, state, args, input_spikes, env_state
        )
        return CriticPrediction(
            value=jnp.zeros_like(state.value),  # Will be updated in the update step
            previous_value=state.value,  # Store current value as previous value for the next step
            features=new_features,
            features_prev=state.features,  # Store unupdated features as previous for the next step
            features_prev_prev=state.features_prev,
            weights=state.weights,
        )

    def drift(self, t, state: CriticPrediction, args, reward, RPE) -> CriticPrediction:
        # TD(0) update rule for the critic weights
        if self.use_bias:
            feat = jnp.concatenate(
                [state.features_prev_prev, jnp.array([1.0])]
            )  # Add bias term
        else:
            feat = state.features_prev_prev
        d_W = args["get_critic_lr"](t, state, args) * RPE * feat

        # features exponentially decay towards zero with time constant tau_filtered_input
        d_features = -state.features / self.tau_filtered_input

        drift = jax.tree.map(lambda arr: jnp.zeros_like(arr, dtype=arr.dtype), state)
        drift = eqx.tree_at(lambda s: s.weights, drift, d_W)
        drift = eqx.tree_at(lambda s: s.features, drift, d_features)
        return drift

    def diffusion(self, t, state: CriticPrediction, args) -> MixedPyTreeOperator:
        diffusion = jax.tree.map(
            lambda arr: DefaultIfNone(
                default=jnp.zeros_like(arr),
                else_do=ElementWiseMul(jnp.zeros_like(arr, dtype=default_float)),
            ),
            state,
        )
        return MixedPyTreeOperator(diffusion)

    def update(self, t, state: CriticPrediction, args) -> CriticPrediction:
        # Make sure both value estimates use the same weights
        prev_feat = (
            jnp.concatenate([state.features_prev, jnp.array([1.0])])
            if self.use_bias
            else state.features_prev
        )
        new_feat = (
            jnp.concatenate([state.features, jnp.array([1.0])])
            if self.use_bias
            else state.features
        )
        prev_value = state.weights @ prev_feat
        new_value = state.weights @ new_feat

        state = eqx.tree_at(lambda s: s.value, state, jnp.atleast_1d(new_value))
        state = eqx.tree_at(
            lambda s: s.previous_value, state, jnp.atleast_1d(prev_value)
        )
        return state

    def reset(self, t, state: CriticPrediction, args) -> CriticPrediction:
        """Reset everything but the weights at the end of an episode."""
        return CriticPrediction(
            value=jnp.zeros_like(state.value),
            previous_value=jnp.zeros_like(state.previous_value),
            features=jnp.zeros_like(state.features),
            features_prev=jnp.zeros_like(state.features_prev),
            features_prev_prev=jnp.zeros_like(state.features_prev_prev),
            weights=state.weights,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )
