import jax.numpy as jnp

from adaptive_SNN.models.reward_prediction import RLSRewardPredictor


def make_predictor(input_dim=1, lambda_=0.9999, P_init=100.0):
    return RLSRewardPredictor(input_dim=input_dim, lambda_=lambda_, P_init=P_init)


def make_args(feature_fn):
    return {"feature_fn": feature_fn}


def constant_feature_fn(value):
    return lambda *_: jnp.array([value])


def test_initial_state_shapes():
    dim = 3
    predictor = make_predictor(input_dim=dim)
    state = predictor.initial

    assert state.value.shape == (1,)
    assert state.weights.shape == (dim,)
    assert state.P.shape == (dim, dim)


def test_initial_state_values():
    dim = 2
    P_init = 50.0
    predictor = make_predictor(input_dim=dim, P_init=P_init)
    state = predictor.initial

    assert jnp.all(state.value == 0.0)
    assert jnp.all(state.weights == 0.0)
    assert jnp.allclose(state.P, jnp.eye(dim) * P_init)


def test_drift_returns_zeros():
    predictor = make_predictor(input_dim=2)
    state = predictor.initial
    drift = predictor.drift(0.0, state, {}, reward=jnp.array(1.0), network_state=None)

    assert jnp.all(drift.value == 0.0)
    assert jnp.all(drift.weights == 0.0)
    assert jnp.all(drift.P == 0.0)


def test_update_is_identity():
    predictor = make_predictor()
    state = predictor.initial
    updated = predictor.update(0.0, state, {})

    assert jnp.allclose(updated.value, state.value)
    assert jnp.allclose(updated.weights, state.weights)
    assert jnp.allclose(updated.P, state.P)


def test_pre_step_update_prediction():
    """With zero weights the predicted reward is zero regardless of features."""
    predictor = make_predictor(input_dim=1)
    state = predictor.initial
    args = make_args(constant_feature_fn(5.0))

    new_state = predictor.pre_step_update(
        0.0, state, args, reward=jnp.array(0.0), network_state=None
    )

    assert jnp.allclose(new_state.value, 0.0)


def test_pre_step_update_weights_change():
    """Weights should update when there is a non-zero reward error."""
    predictor = make_predictor(input_dim=1)
    state = predictor.initial
    args = make_args(constant_feature_fn(1.0))

    new_state = predictor.pre_step_update(
        0.0, state, args, reward=jnp.array(1.0), network_state=None
    )

    assert not jnp.allclose(new_state.weights, state.weights)


def test_pre_step_update_P_changes():
    """P matrix should be updated after each step."""
    predictor = make_predictor(input_dim=1)
    state = predictor.initial
    args = make_args(constant_feature_fn(1.0))

    new_state = predictor.pre_step_update(
        0.0, state, args, reward=jnp.array(1.0), network_state=None
    )

    assert not jnp.allclose(new_state.P, state.P)


def test_rls_converges_to_constant_reward():
    """RLS should learn to predict a constant reward accurately after many steps."""
    input_dim = 1
    predictor = make_predictor(input_dim=input_dim, lambda_=1.0)
    state = predictor.initial
    args = make_args(constant_feature_fn(1.0))
    true_reward = 3.7

    for _ in range(50):
        state = predictor.pre_step_update(
            0.0, state, args, reward=jnp.array(true_reward), network_state=None
        )

    assert jnp.isclose(state.value, true_reward, atol=1e-3)


def test_rls_multidim_converges():
    """RLS with multi-dimensional features should converge to predict a linear reward."""
    input_dim = 3
    predictor = make_predictor(input_dim=input_dim, lambda_=1.0)
    state = predictor.initial

    true_weights = jnp.array([1.0, -2.0, 0.5])
    features = jnp.array([1.0, 0.5, 2.0])
    true_reward = true_weights @ features

    feature_fn = lambda *_: features
    args = make_args(feature_fn)

    for _ in range(100):
        state = predictor.pre_step_update(
            0.0, state, args, reward=true_reward, network_state=None
        )

    assert jnp.isclose(state.value, true_reward, atol=1e-3)


def test_P_remains_symmetric():
    """P should stay symmetric after updates."""
    input_dim = 3
    predictor = make_predictor(input_dim=input_dim)
    state = predictor.initial
    features = jnp.array([1.0, 2.0, 0.5])
    args = make_args(lambda *_: features)

    for _ in range(10):
        state = predictor.pre_step_update(
            0.0, state, args, reward=jnp.array(1.0), network_state=None
        )

    assert jnp.allclose(state.P, state.P.T, atol=1e-10)


def test_zero_reward_no_weight_change():
    """If reward is zero and initial prediction is zero, weights should not change."""
    predictor = make_predictor(input_dim=1)
    state = predictor.initial
    args = make_args(constant_feature_fn(1.0))

    new_state = predictor.pre_step_update(
        0.0, state, args, reward=jnp.array(0.0), network_state=None
    )

    assert jnp.allclose(new_state.weights, state.weights)
