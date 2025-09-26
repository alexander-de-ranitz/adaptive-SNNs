import warnings
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
    def compute_spikes_and_update(self, t, x, args):
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
        return jnp.eye(self.dim) * self.noise_scale

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

    The state consists of the membrane potentials, weights, and synaptic conductances of all neurons.
    """

    leak_conductance: float = 16.7 * 1e-9  # nS
    membrane_conductance: float = 250 * 1e-12  # pF
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
        fully_connected_input: bool = True,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        self.N_neurons = N_neurons
        self.N_inputs = N_inputs
        key, key_1, key_2, key_3 = jr.split(key, 4)

        # Set weights
        self.weights = jr.normal(
            key_1, (N_neurons, N_neurons + N_inputs)
        ) * jr.bernoulli(key_2, self.connection_prob, (N_neurons, N_neurons + N_inputs))
        if (
            fully_connected_input and N_inputs > 0
        ):  # Make all input connections fully connected
            self.weights = self.weights.at[:, N_neurons:].set(
                jnp.ones(shape=(N_neurons, N_inputs)) * self.input_weight
            )

        # Set neuron types and time constants
        neuron_types = jnp.where(
            jr.bernoulli(key_3, 0.8, (N_neurons,)), True, False
        )  # 80% excitatory, 20% inhibitory
        self.excitatory_mask = jnp.concatenate(
            [neuron_types, jnp.ones((N_inputs,))], dtype=bool
        )  # inputs are all excitatory
        self.synaptic_time_constants = jnp.where(
            self.excitatory_mask, self.tau_E, self.tau_I
        )

    @property
    def initial(self):
        """Return initial network state.

        State tuple ordering:
            V: Membrane potentials (N_neurons,)
            W: Synaptic weight matrix (N_neurons, N_neurons + N_inputs)
            G: Synaptic conductances (N_neurons, N_neurons + N_inputs)
        """
        V_init = jnp.zeros((self.N_neurons,)) + self.resting_potential
        conductance_init = jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs))
        return (V_init, self.weights, conductance_init)

    def drift(self, t, x, args):
        # Unpack state (V, W, G)
        V, W, conductances = x

        # Compute leak current
        leak_current = -self.leak_conductance * (V - self.resting_potential)

        # Compute E/I currents from recurrent connections
        # Use dynamic weights W from state (instead of module attribute) to allow plasticity
        weighted_conductances = W * conductances
        inhibitory_conductances = jnp.sum(
            weighted_conductances * jnp.invert(self.excitatory_mask[None, :]), axis=1
        )
        excitatory_conductances = jnp.sum(
            weighted_conductances * self.excitatory_mask[None, :], axis=1
        )

        # Get noise from args and add to total conductances
        if args and "inhibitory_noise" in args:
            inhibitory_conductances += args["inhibitory_noise"](t, x, args)
        if args and "excitatory_noise" in args:
            excitatory_conductances += args["excitatory_noise"](t, x, args)

        # Compute total recurrent current
        recurrent_current = inhibitory_conductances * (
            self.reversal_potential_I - V
        ) + excitatory_conductances * (self.reversal_potential_E - V)

        dVdt = (leak_current + recurrent_current) / self.membrane_conductance

        # Compute synaptic conductance changes
        dGdt = -1 / self.synaptic_time_constants * conductances

        # Compute weight changes
        # TODO: implement plasticity rule here
        dW = jnp.zeros_like(W)  # Placeholder for plasticity rule

        return dVdt, dW, dGdt

    def diffusion(self, t, x, args):
        # Currently deterministic; return zeros for (V, W, G) components
        return (
            jnp.zeros((self.N_neurons,)),
            jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs)),  # dW noise
            jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs)),  # dG noise
        )

    @property
    def noise_shape(self):
        # TODO: same as above: should this be None instead of shapes of zeros?
        return (
            jax.ShapeDtypeStruct(shape=(self.N_neurons,), dtype=default_float),
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

    def compute_spikes_and_update(self, t, x, args):
        V, W, G = x
        spikes = jnp.float32(V > self.firing_threshold)
        V_new = (1.0 - spikes) * V + spikes * self.V_reset  # Reset voltage

        # Add input spikes if provided in args
        if args and "input_spikes" in args:
            if self.N_inputs == 0:
                warnings.warn(
                    "Input spikes provided to neuron model with no inputs, ignoring input spikes.",
                    stacklevel=3,
                )
            if jnp.any(args["input_spikes"](t, x, args) > 1) or jnp.any(
                args["input_spikes"](t, x, args) < 0
            ):
                warnings.warn("Input spikes must be binary (0 or 1).", stacklevel=3)

            spikes = jnp.concatenate((spikes, args["input_spikes"](t, x, args)))
        elif self.N_inputs > 0:
            spikes = jnp.concatenate((spikes, jnp.zeros((self.N_inputs,))))
            warnings.warn(
                "No input spikes provided to neuron model with inputs, assuming 0 input spikes.",
                stacklevel=3,
            )

        G_new = (
            G + spikes[None, :] * self.synaptic_increment
        )  # increase conductance on spike

        # Weights unchanged here (plasticity could be added later using spikes)
        return (V_new, W, G_new), spikes


class NoisyLIFModel(eqx.Module):
    N_neurons: int = eqx.field(
        static=True
    )  # TODO: this can be removed if we always get N_neurons from network
    network: LIFNetwork
    noise_E: OUP  # TODO: might be better to use a single OUP with 2 dimensions
    noise_I: OUP

    def __init__(self, N_neurons: int, neuron_model, noise_I_model, noise_E_model):
        self.N_neurons = N_neurons
        self.network = neuron_model
        self.noise_E = noise_E_model
        self.noise_I = noise_I_model

        assert self.network.N_neurons == self.N_neurons, (
            "Number of neurons in network must match N_neurons"
        )
        assert self.noise_E.dim == self.N_neurons, (
            "Dimension of excitatory noise must match number of neurons"
        )
        assert self.noise_I.dim == self.N_neurons, (
            "Dimension of inhibitory noise must match number of neurons"
        )

    @property
    def initial(self):
        return (self.network.initial, self.noise_E.initial, self.noise_I.initial)

    def drift(self, t, x, args):
        (V, W, conductances), noise_E_state, noise_I_state = x

        if args is None:
            args = {}
        args["excitatory_noise"] = lambda t, x, args: noise_E_state
        args["inhibitory_noise"] = lambda t, x, args: noise_I_state

        network_drift = self.network.drift(t, (V, W, conductances), args)
        noise_E_drift = self.noise_E.drift(t, noise_E_state, args)
        noise_I_drift = self.noise_I.drift(t, noise_I_state, args)
        return (network_drift, noise_E_drift, noise_I_drift)

    def diffusion(self, t, x, args):
        (V, W, conductances), noise_E_state, noise_I_state = x
        network_diffusion = self.network.diffusion(t, (V, W, conductances), args)
        noise_E_diffusion = self.noise_E.diffusion(t, noise_E_state, args)
        noise_I_diffusion = self.noise_I.diffusion(t, noise_I_state, args)
        return (network_diffusion, noise_E_diffusion, noise_I_diffusion)

    @property
    def noise_shape(self):
        return (
            self.network.noise_shape,
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


class LearningModel(eqx.Module):
    neuron_model: NoisyLIFModel
    reward_model: RewardModel
    environment: EnvironmentModel
    learning_rate: float = 1e-3

    def __init__(
        self,
        neuron_model: NoisyLIFModel,
        reward_model: RewardModel,
        environment: EnvironmentModel,
        learning_rate: float = 1e-3,
    ):
        self.neuron_model = neuron_model
        self.reward_model = reward_model
        self.environment = environment
        self.learning_rate = learning_rate

    @property
    def initial(self):
        return (
            self.neuron_model.initial,
            self.reward_model.initial,
            self.environment.initial,
        )

    def drift(self, t, x, args):
        (neuron_state, reward_state, env_state) = x
        if args is None:
            args = {}

        # Compute network output, reward, and RPE
        network_output = args["network_output"](t, neuron_state, args)
        reward = args["compute_reward"](t, env_state, args)
        RPE = reward - reward_state

        # Add to args for use in models
        args["env_input"] = lambda t, x, args: network_output
        args["RPE"] = lambda t, x, args: RPE
        args["reward"] = lambda t, x, args: reward

        neuron_drift = self.neuron_model.drift(t, neuron_state, args)
        reward_drift = self.reward_model.drift(t, reward_state, args)
        env_drift = self.environment.drift(t, env_state, args)
        return (neuron_drift, reward_drift, env_drift)

    def diffusion(self, t, x, args):
        (neuron_state, reward_state, env_state) = x
        neuron_diffusion = self.neuron_model.diffusion(t, neuron_state, args)
        reward_diffusion = self.reward_model.diffusion(t, reward_state, args)
        env_diffusion = self.environment.diffusion(t, env_state, args)
        return (neuron_diffusion, reward_diffusion, env_diffusion)

    @property
    def noise_shape(self):
        return (
            self.neuron_model.noise_shape,
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
