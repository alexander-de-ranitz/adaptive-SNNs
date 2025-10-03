from abc import ABC, abstractmethod

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from adaptive_SNN.models.environment import EnvironmentModel
from adaptive_SNN.models.reward import RewardModel

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class LIFState(eqx.Module):
    """State container for LIF network.

    Attributes:
        V: Membrane potentials (N_neurons,)
        S: Spike vector (N_neurons + N_inputs,)
        W: Synaptic weight matrix (N_neurons, N_neurons + N_inputs)
        G: Synaptic conductances (N_neurons, N_neurons + N_inputs)
    """

    V: Array
    S: Array
    W: Array
    G: Array


class NeuronModel(ABC, eqx.Module):
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
        """Apply non-differential updates to the state, e.g. spikes, resets, balancing, etc."""
        pass


class OUP(eqx.Module):
    theta: float | Array = 1.0
    noise_scale: float | Array = 1.0
    mean: float | Array = 0.0
    dim: int = 1

    @property
    def initial(self):
        return jnp.ones((self.dim,)) * self.mean

    def drift(self, t, x, args):
        return -self.theta * (x - self.mean)

    def diffusion(self, t, x, args):
        return jnp.eye(x.shape[0]) * self.noise_scale

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim,), dtype=default_float)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )


class LIFNetwork(NeuronModel):
    """Leaky Integrate-and-Fire (LIF) neuron network model with conductance-based synapses.

    The state consists of the membrane potentials, spikes, weights, and synaptic conductances of all neurons.
    """

    # TODO: Add refractory period and synaptic delays

    leak_conductance: float = 16.7 * 1e-9  # nS
    membrane_capacitance: float = 250 * 1e-12  # pF
    resting_potential: float = -70.0 * 1e-3  # mV
    connection_prob: float = 0.1
    reversal_potential_E: float = 0.0  # mV
    reversal_potential_I: float = -75.0 * 1e-3  # mV
    tau_E: float = 2.0 * 1e-3  # ms
    tau_I: float = 6.0 * 1e-3  # ms
    synaptic_increment: float = 1.0 * 1e-9  # nS
    firing_threshold: float = -50.0 * 1e-3  # mV
    V_reset: float = -60.0 * 1e-3  # mV
    input_weight: float = 10.0  # Weight of input spikes
    learning_rate: float = 1e-3  # Learning rate for plasticity

    N_neurons: int = eqx.field(static=True)
    N_inputs: int = eqx.field(static=True)
    excitatory_mask: Array  # Binary vector of size N_neurons + N_inputs with: 1 (excitatory) and 0 (inhibitory)
    synaptic_time_constants: (
        Array  # Vector of size N_neurons with synaptic time constants (tau_E or tau_I)
    )
    weights: Array  # Shape (N_neurons, N_neurons + N_inputs)

    def __init__(
        self,
        N_neurons: int,
        N_inputs: int = 0,
        input_neuron_types: Array = None,  # Binary vector of size N_inputs with: 1 (excitatory) and 0 (inhibitory)
        fully_connected_input: bool = True,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        self.N_neurons = N_neurons
        self.N_inputs = N_inputs
        key, key_1, key_2, key_3 = jr.split(key, 4)

        # Set weights
        # TODO: move this to initial state, the weights should be part of the state and not a class attribute
        self.weights = jr.normal(
            key_1, (N_neurons, N_neurons + N_inputs)
        ) * jr.bernoulli(key_2, self.connection_prob, (N_neurons, N_neurons + N_inputs))
        if (
            fully_connected_input and N_inputs > 0
        ):  # Make all input connections fully connected
            self.weights = self.weights.at[:, N_neurons:].set(
                jnp.ones(shape=(N_neurons, N_inputs)) * self.input_weight
            )

        # Set neuron types
        neuron_types = jnp.where(
            jr.bernoulli(key_3, 0.8, (N_neurons,)), True, False
        )  # 80% excitatory, 20% inhibitory
        # If no input neuron types provided, assume all input neurons are excitatory
        if input_neuron_types is None:
            input_neuron_types = jnp.ones((N_inputs,), dtype=bool)
        self.excitatory_mask = jnp.concatenate(
            [neuron_types, input_neuron_types], dtype=bool
        )
        self.synaptic_time_constants = jnp.where(
            self.excitatory_mask, self.tau_E, self.tau_I
        )

    @property
    def initial(self):
        """Return initial network state as LIFState."""
        V_init = (
            jnp.zeros((self.N_neurons,), dtype=default_float) + self.resting_potential
        )
        conductance_init = jnp.zeros(
            (self.N_neurons, self.N_neurons + self.N_inputs), dtype=default_float
        )
        spikes_init = jnp.zeros((self.N_neurons + self.N_inputs,), dtype=default_float)
        return LIFState(V_init, spikes_init, self.weights, conductance_init)

    def drift(self, t, state: LIFState, args) -> LIFState:
        """Compute deterministic time derivatives for LIF state.

        Args:
            t: time (unused for autonomous dynamics)
            state: LIFState current state
            args: dict containing keys:
                - inhibitory_noise(t, state, args) -> (N_neurons,)
                - excitatory_noise(t, state, args) -> (N_neurons,)
                - learning_rate(t, state, args) -> scalar
                - RPE(t, state, args) -> scalar
        Returns:
            LIFState of derivatives (dV, dS, dW, dG)
        """
        V, S, W, G = state.V, state.S, state.W, state.G

        # Compute leak current
        leak_current = -self.leak_conductance * (V - self.resting_potential)

        # Compute E/I currents from recurrent connections
        weighted_conductances = W * G
        inhibitory_conductances = jnp.sum(
            weighted_conductances * jnp.invert(self.excitatory_mask[None, :]), axis=1
        )
        excitatory_conductances = jnp.sum(
            weighted_conductances * self.excitatory_mask[None, :], axis=1
        )

        # Get noise from args and add to total conductances
        inhibitory_conductances += args["inhibitory_noise"](t, state, args)
        excitatory_conductances += args["excitatory_noise"](t, state, args)

        # Compute total recurrent current
        recurrent_current = inhibitory_conductances * (
            self.reversal_potential_I - V
        ) + excitatory_conductances * (self.reversal_potential_E - V)

        dVdt = (leak_current + recurrent_current) / self.membrane_capacitance

        # Compute synaptic conductance changes
        dGdt = -1 / self.synaptic_time_constants[None, :] * G

        # Compute weight changes
        learning_rate = args["learning_rate"](t, state, args)
        RPE = args["RPE"](t, state, args)
        E_noise = jnp.outer(
            args["excitatory_noise"](t, state, args), self.excitatory_mask
        )

        # No learning of inhibitory weights for now
        # TODO: If desired, implement inhibitory weight learning
        # I_noise = jnp.outer(
        #                 args["inhibitory_noise"](t, state, args),
        #                 jnp.invert(self.excitatory_mask),
        #             )

        dW = learning_rate * RPE * (E_noise) * G

        dS = jnp.zeros_like(S)  # Spikes are handled separately, so no change here

        return LIFState(dVdt, dS, dW, dGdt)

    def diffusion(self, t, state: LIFState, args) -> LIFState:
        # TODO: Fix this. See below
        # The diffusion (vf) and noise term (control) are used as:
        # jnp.tensordot(jnp.conj(vf), control, axes=jnp.ndim(control)),
        # applied to each leaf of the pytree.
        # See https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html#numpy.linalg.tensordot
        # With the values below, this tensordot returns a pytree with shapes: ((N_neurons,), (), ())
        # this is then (presumably) broadcasted to the shapes of the state ((N_neurons,), (N_neurons, N_neurons + N_inputs), (N_neurons, N_neurons + N_inputs))
        # As everything is zero, this does not cause issues, but it is not correct.
        #
        # I would probably like to do element-wise multiplication.
        # Perhaps a custom Lineax operator could be used here? Investigate this further.

        return LIFState(
            jnp.zeros_like(state.V, dtype=default_float),  # dV noise
            jnp.zeros_like(state.S, dtype=default_float),  # dS noise
            jnp.zeros_like(state.W, dtype=default_float),  # dW noise
            jnp.zeros_like(state.G, dtype=default_float),  # dG noise
        )

    @property
    def noise_shape(self):
        # TODO: same as above: should this be None instead of shapes of zeros?
        return LIFState(
            jax.ShapeDtypeStruct(shape=(self.N_neurons,), dtype=default_float),
            jax.ShapeDtypeStruct(
                shape=(self.N_neurons + self.N_inputs,), dtype=default_float
            ),
            jax.ShapeDtypeStruct(
                shape=(self.N_neurons, self.N_neurons + self.N_inputs),
                dtype=default_float,
            ),
            jax.ShapeDtypeStruct(
                shape=(self.N_neurons, self.N_neurons + self.N_inputs),
                dtype=default_float,
            ),
        )

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

    def update(self, t, x, args):
        """Apply non-differential updates to the state, e.g. spikes, resets, balancing, etc."""
        state = self.spike_and_reset(t, x, args)
        state = self.force_balanced_weights(t, state, args)
        return state

    def spike_and_reset(self, t, state: LIFState, args):
        V, _, W, G = state.V, state.S, state.W, state.G
        recurrent_spikes = (V > self.firing_threshold).astype(V.dtype)
        V_new = (1.0 - recurrent_spikes) * V + recurrent_spikes * self.V_reset

        input_sp = args["input_spikes"](t, state, args)
        spikes = jnp.concatenate((recurrent_spikes, input_sp))
        G_new = G + spikes[None, :] * self.synaptic_increment
        return LIFState(V_new, spikes, W, G_new)

    def compute_balance(self, t, state, args):
        """Compute the ratio of total inhibitory to excitatory input weights for each neuron."""
        weights = state.W
        excitatory_weights = jnp.sum(weights * self.excitatory_mask[None, :], axis=1)
        inhibitory_weights = jnp.sum(
            weights * jnp.invert(self.excitatory_mask[None, :]), axis=1
        )
        balance = inhibitory_weights / (
            excitatory_weights + 1e-12
        )  # Avoid division by zero
        return balance

    def force_balanced_weights(self, t, state, args):
        """Adjust weights to achieve a desired E/I balance for each neuron"""
        desired_balance = args["desired_balance"](t, state, args)
        current_balance = self.compute_balance(t, state, args)
        adjust_ratio = desired_balance / current_balance

        # If desired_balance is 0, do not adjust anything
        adjust_ratio = jnp.where(
            adjust_ratio == 0.0, jnp.ones_like(state.V), adjust_ratio
        )
        adjust_ratio = jnp.clip(
            adjust_ratio, 0.01, 100.0
        )  # For safety, do not allow too large adjustments
        # Scale inhibitory weights to achieve desired balance
        balanced_weights = state.W * (
            jnp.outer(adjust_ratio, jnp.invert(self.excitatory_mask))
            + jnp.outer(jnp.ones(self.N_neurons), self.excitatory_mask)
        )
        return LIFState(state.V, state.S, balanced_weights, state.G)


class NoisyNetworkState(eqx.Module):
    network_state: LIFState
    noise_E_state: Array  # (N_neurons,)
    noise_I_state: Array  # (N_neurons,)


class NoisyNetwork(NeuronModel):
    base_network: NeuronModel
    noise_E: OUP  # TODO: might be better to use a single OUP with 2 dimensions
    noise_I: OUP
    # TODO: For generability, it would be better to allow any noise model. We would have an ABC NoiseModel that OUP inherits from.

    def __init__(
        self, neuron_model: NeuronModel, noise_I_model: OUP, noise_E_model: OUP
    ):
        self.base_network = neuron_model
        self.noise_E = noise_E_model
        self.noise_I = noise_I_model

        assert self.noise_E.dim == self.base_network.N_neurons, (
            "Dimension of excitatory noise must match number of neurons"
        )
        assert self.noise_I.dim == self.base_network.N_neurons, (
            "Dimension of inhibitory noise must match number of neurons"
        )

    @property
    def initial(self):
        return NoisyNetworkState(
            self.base_network.initial, self.noise_E.initial, self.noise_I.initial
        )

    def drift(self, t, state: NoisyNetworkState, args):
        network_state, noise_E_state, noise_I_state = (
            state.network_state,
            state.noise_E_state,
            state.noise_I_state,
        )

        # TODO: As above, this could be made more general by allowing any noise model and just passing that
        # TODO: Move this so args are not changed in an inner function for JIT compatibility
        network_args = {
            **args,
            "excitatory_noise": lambda t, x, args: noise_E_state,
            "inhibitory_noise": lambda t, x, args: noise_I_state,
        }

        network_drift = self.base_network.drift(t, network_state, network_args)
        noise_E_drift = self.noise_E.drift(t, noise_E_state, args)
        noise_I_drift = self.noise_I.drift(t, noise_I_state, args)
        return NoisyNetworkState(network_drift, noise_E_drift, noise_I_drift)

    def diffusion(self, t, state: NoisyNetworkState, args):
        network_state, noise_E_state, noise_I_state = (
            state.network_state,
            state.noise_E_state,
            state.noise_I_state,
        )
        network_diffusion = self.base_network.diffusion(t, network_state, args)
        noise_E_diffusion = self.noise_E.diffusion(t, noise_E_state, args)
        noise_I_diffusion = self.noise_I.diffusion(t, noise_I_state, args)
        return NoisyNetworkState(
            network_diffusion, noise_E_diffusion, noise_I_diffusion
        )

    @property
    def noise_shape(self):
        return NoisyNetworkState(
            self.base_network.noise_shape,
            self.noise_E.noise_shape,
            self.noise_I.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x: NoisyNetworkState, args):
        network_state, noise_E_state, noise_I_state = (
            x.network_state,
            x.noise_E_state,
            x.noise_I_state,
        )
        new_network_state = self.base_network.update(t, network_state, args)
        return NoisyNetworkState(new_network_state, noise_E_state, noise_I_state)


class AgentSystem(eqx.Module):
    noisy_network: NoisyNetwork
    reward_model: RewardModel
    environment: EnvironmentModel

    def __init__(
        self,
        neuron_model: NoisyNetwork,
        reward_model: RewardModel,
        environment: EnvironmentModel,
    ):
        self.noisy_network = neuron_model
        self.reward_model = reward_model
        self.environment = environment

    @property
    def initial(self):
        return (
            self.noisy_network.initial,
            self.reward_model.initial,
            self.environment.initial,
        )

    def drift(self, t, x, args):
        """Compute deterministic time derivatives for LearningModel state.

        The state consists of (network_state, reward_state, environment_state). The args dict
        must contain functions to compute the network output and reward, which are used to compute
        the reward prediction error (RPE) for learning.

        Args:
            t: time
            x: (network_state, reward_state, environment_state)
            args: dict containing keys:
                - network_output(t, network_state, args) -> scalar
                - compute_reward(t, environment_state, args) -> scalar
        Returns:
            (d_network_state, d_reward_state, d_environment_state)
        """

        (network_state, reward_state, env_state) = x

        if args is None or "network_output" not in args or "compute_reward" not in args:
            raise ValueError(
                "Args must contain 'network_output' and 'compute_reward' functions."
            )

        # Compute network output, reward, and RPE
        network_output = args["network_output"](t, network_state, args)
        reward = args["compute_reward"](t, env_state, args)
        RPE = jnp.asarray(reward - reward_state)

        # Add to args for use in models
        args = {
            **args,
            "env_input": lambda t, x, args: network_output,
            "RPE": lambda t, x, args: RPE,
            "reward": lambda t, x, args: reward,
        }

        neuron_drift = self.noisy_network.drift(t, network_state, args)
        reward_drift = self.reward_model.drift(t, reward_state, args)
        env_drift = self.environment.drift(t, env_state, args)
        return (neuron_drift, reward_drift, env_drift)

    def diffusion(self, t, x, args):
        (neuron_state, reward_state, env_state) = x
        neuron_diffusion = self.noisy_network.diffusion(t, neuron_state, args)
        reward_diffusion = self.reward_model.diffusion(t, reward_state, args)
        env_diffusion = self.environment.diffusion(t, env_state, args)
        return (neuron_diffusion, reward_diffusion, env_diffusion)

    @property
    def noise_shape(self):
        return (
            self.noisy_network.noise_shape,
            self.reward_model.noise_shape,
            self.environment.noise_shape,
        )

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(
            shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea
        )
        return dfx.MultiTerm(
            dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise)
        )

    def update(self, t, x, args):
        (network_state, reward_state, env_state) = x
        new_network_state = self.noisy_network.update(t, network_state, args)
        return (new_network_state, reward_state, env_state)
