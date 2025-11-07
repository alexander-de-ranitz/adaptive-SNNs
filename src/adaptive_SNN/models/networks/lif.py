import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from adaptive_SNN.models.base import NeuronModelABC
from adaptive_SNN.utils import ElementWiseMul, MixedPyTreeOperator

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class LIFState(eqx.Module):
    """State container for LIF network.

    Used to make it easier to work with the state by avoiding tuples with many indices.

    Attributes:
        V: Membrane potentials (N_neurons,)
        S: Spike vector (N_neurons + N_inputs,)
        W: Synaptic weight matrix (N_neurons, N_neurons + N_inputs). -inf indicates no connection
        G: Synaptic conductances (N_neurons, N_neurons + N_inputs)
        time_since_last_spike: Time since last spike for each neuron (N_neurons,)
        spike_buffer: Circular buffer of past spikes (buffer_size, N_neurons + N_inputs)
        buffer_index: Current write position in spike buffer (scalar int)
    """

    V: Array
    S: Array
    W: Array
    G: Array
    time_since_last_spike: Array
    spike_buffer: Array
    buffer_index: Array  # Scalar array to maintain JAX compatibility


class LIFNetwork(NeuronModelABC):
    """Leaky Integrate-and-Fire (LIF) neuron network model with conductance-based synapses.

    The state consists of the membrane potentials, spikes, weights, and synaptic conductances of all neurons.
    """

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
    refractory_period: float = 2.0 * 1e-3  # ms
    mean_synaptic_delay: float = 1.5 * 1e-3  # ms

    input_weight: float  # Weight of input spikes
    fully_connected_input: bool  # If True, all input neurons connect to all neurons with weight input_weight
    N_neurons: int
    N_inputs: int
    excitatory_mask: Array  # Binary vector of size N_neurons + N_inputs with: 1 (excitatory) and 0 (inhibitory)
    synaptic_time_constants: Array  # Vector of size N_neurons + N_inputs with synaptic time constants (tau_E or tau_I)
    synaptic_delay_matrix: (
        Array  # Matrix of synaptic delays (N_neurons, N_neurons + N_inputs) in seconds
    )
    buffer_size: int  # Size of spike history buffer
    dt: float  # Timestep size for simulation in seconds, used for delay buffer

    def __init__(
        self,
        dt,
        N_neurons: int,
        N_inputs: int = 0,
        input_neuron_types: Array = None,  # Binary vector of size N_inputs with: 1 (excitatory) and 0 (inhibitory)
        fully_connected_input: bool = True,
        input_weight: float = 1.0,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        """Initialize LIF network model.

        Args:
            N_neurons: Number of neurons in the network
            N_inputs: Number of input neurons
            input_neuron_types: Binary vector of size N_inputs with: 1 (excitatory) and 0 (inhibitory). If None, all input neurons are excitatory.
            fully_connected_input: If True, all input neurons connect to all neurons with weight input_weight
            input_weight: Weight of input spikes
            dt: Timestep size for simulation (s)
            key: JAX random key for weight initialization
        """
        self.N_neurons = N_neurons
        self.N_inputs = N_inputs
        self.fully_connected_input = fully_connected_input
        self.input_weight = input_weight
        self.dt = dt

        key, subkey = jr.split(key)

        # Set neuron types, 80% excitatory, 20% inhibitory
        neuron_types = jr.permutation(
            subkey,
            jnp.concatenate(
                [
                    jnp.ones((int(0.8 * N_neurons),), dtype=bool),
                    jnp.zeros((N_neurons - int(0.8 * N_neurons),), dtype=bool),
                ]
            ),
        )

        # If no input neuron types provided, assume all input neurons are excitatory
        if input_neuron_types is None:
            input_neuron_types = jnp.ones((N_inputs,), dtype=bool)

        self.excitatory_mask = jnp.concatenate(
            [neuron_types, input_neuron_types], dtype=bool
        )
        self.synaptic_time_constants = jnp.where(
            self.excitatory_mask, self.tau_E, self.tau_I
        )

        # Set synaptic delays
        # TODO: Tune this properly
        key, subkey = jr.split(key)
        delays = self.mean_synaptic_delay * (
            1.0
            + 0.1
            * jr.normal(subkey, shape=(self.N_neurons, self.N_neurons + self.N_inputs))
        )
        delays = jnp.clip(
            delays,
            min=0.5 * self.mean_synaptic_delay,
            max=2.0 * self.mean_synaptic_delay,
        )  # Avoid too small or too large
        self.synaptic_delay_matrix = delays

        # Compute buffer size based on max delay
        self.buffer_size = (jnp.ceil(jnp.max(delays) / self.dt) + 1).astype(jnp.int32)

    @property
    def initial(self, key: jr.PRNGKey = jr.PRNGKey(0)):
        """Return initial network state as LIFState."""
        V_init = (
            jnp.zeros((self.N_neurons,), dtype=default_float) + self.resting_potential
        )
        conductance_init = jnp.zeros(
            (self.N_neurons, self.N_neurons + self.N_inputs), dtype=default_float
        )
        spikes_init = jnp.zeros((self.N_neurons + self.N_inputs,), dtype=default_float)

        key, subkey = jr.split(key)

        # Initialize weights with random sparse connectivity with no self-connections or double connections
        # The weights are drawn from a normal distribution around with mean 1 and std 0.1
        num_possible_connections = self.N_neurons * (self.N_neurons + self.N_inputs)
        num_connections = int(num_possible_connections * self.connection_prob)
        weights = jnp.abs(
            1 + 0.1 * jr.normal(key, (self.N_neurons, self.N_neurons + self.N_inputs))
        ) * jr.permutation(
            subkey,
            jnp.concatenate(
                [
                    jnp.ones(num_connections),
                    jnp.zeros(num_possible_connections - num_connections),
                ]
            ),
        ).reshape(self.N_neurons, self.N_neurons + self.N_inputs)

        weights = jnp.fill_diagonal(weights, 0.0, inplace=False)  # No self-connections
        weights = jnp.where(
            weights == 0.0, -jnp.inf, weights
        )  # Non existing connections have weight -inf

        # If fully_connected_input is True, set all input weights to input_weight
        if self.fully_connected_input and (self.N_inputs > 0):
            weights = weights.at[:, self.N_neurons :].set(
                jnp.ones(shape=(self.N_neurons, self.N_inputs)) * self.input_weight
            )

        time_since_last_spike = (
            jnp.ones((self.N_neurons,), dtype=default_float) * jnp.inf
        )

        # Initialize spike delay buffer
        spike_buffer = jnp.zeros(
            (self.buffer_size, self.N_neurons + self.N_inputs), dtype=default_float
        )
        buffer_index = jnp.array(0, dtype=jnp.int32)

        return LIFState(
            V=V_init,
            S=spikes_init,
            W=weights,
            G=conductance_init,
            time_since_last_spike=time_since_last_spike,
            spike_buffer=spike_buffer,
            buffer_index=buffer_index,
        )

    def drift(self, t, state: LIFState, args) -> LIFState:
        """Compute deterministic time derivatives for LIF state.

        Args:
            t: time (unused for autonomous dynamics)
            state: LIFState current state
            args: dict containing keys:
                - inhibitory_noise-> (N_neurons,)
                - excitatory_noise-> (N_neurons,)
                - learning_rate(t, state, args) -> scalar
                - RPE(t, state, args) -> scalar
        Returns:
            LIFState of derivatives (dV, dS, dW, dG)
        """
        V, S, W, G = state.V, state.S, state.W, state.G

        # Compute leak current
        leak_current = -self.leak_conductance * (V - self.resting_potential)

        # Compute E/I currents from recurrent connections
        weighted_conductances = (
            jnp.where(W == -jnp.inf, 0.0, W) * G
        )  # For non-existing connections, set weighted conductance to 0
        total_I_conductances = jnp.sum(
            weighted_conductances * jnp.invert(self.excitatory_mask[None, :]), axis=1
        )
        total_E_conductances = (
            jnp.sum(weighted_conductances * self.excitatory_mask[None, :], axis=1)
            + args["excitatory_noise"]
        )  # Add external excitatory noise to total excitatory conductance

        # Ensure non-negative conductances
        total_E_conductances = jnp.clip(total_E_conductances, min=0.0)
        total_I_conductances = jnp.clip(total_I_conductances, min=0.0)

        # Compute total recurrent current
        recurrent_current = total_I_conductances * (
            self.reversal_potential_I - V
        ) + total_E_conductances * (self.reversal_potential_E - V)

        dV = (leak_current + recurrent_current) / self.membrane_capacitance

        # Compute synaptic conductance changes
        dGdt = -1 / self.synaptic_time_constants[None, :] * G

        # Compute weight changes
        learning_rate = args["get_learning_rate"](t, state, args)
        RPE = args.get("RPE", jnp.array(0.0))
        E_noise = jnp.outer(args["excitatory_noise"], self.excitatory_mask)

        dW = (
            learning_rate
            * RPE
            * (E_noise / self.synaptic_increment)
            * (G / self.synaptic_increment)
        )  # Since W is in arbitrary units (not nS), scale by synaptic increment to get a sensible scale
        dW = jnp.where(
            W == -jnp.inf, 0.0, dW
        )  # No weight change for non-existing connections

        dS = jnp.zeros_like(S)  # Spikes are handled separately, so no change here

        d_time_since_last_spike = jnp.ones_like(
            state.time_since_last_spike
        )  # Time since last spike increases by 1 for all neurons

        dV = jnp.where(
            state.time_since_last_spike < self.refractory_period, 0.0, dV
        )  # Neurons in refractory period do not change their membrane potential

        # Buffer fields have zero derivative (updated discretely in spike_and_reset)
        d_spike_buffer = jnp.zeros_like(state.spike_buffer)
        d_buffer_index = jnp.zeros_like(
            state.buffer_index, dtype=state.buffer_index.dtype
        )

        return LIFState(
            dV, dS, dW, dGdt, d_time_since_last_spike, d_spike_buffer, d_buffer_index
        )

    def diffusion(self, t, state: LIFState, args) -> LIFState:
        # Our noise_shape is a pytree of 1d and 2d arrays. Diffusion must have compatible shapes.
        # since each noise value is independent, element-wise multiplication of the matrix-valued noise term is sufficient,
        # we do not need to use tensor products or similar more complex operations.
        # We wrap everything in a MixedPyTreeOperator, which can handle a pytree of both Lineax operators and arrays.
        # In this case, all elements are ElementWiseMul operators of zeros, since we do not use noise in this model.
        # However, I wanted to keep the structure for future use, e.g. if we want to add noise to conductances or weights
        return MixedPyTreeOperator(
            LIFState(
                ElementWiseMul(jnp.zeros_like(state.V, dtype=default_float)),  # V noise
                ElementWiseMul(jnp.zeros_like(state.S, dtype=default_float)),  # S noise
                ElementWiseMul(jnp.zeros_like(state.W, dtype=default_float)),  # W noise
                ElementWiseMul(jnp.zeros_like(state.G, dtype=default_float)),  # G noise
                ElementWiseMul(
                    jnp.zeros_like(state.time_since_last_spike, dtype=default_float)
                ),  # time_since_last_spike noise
                ElementWiseMul(
                    jnp.zeros_like(state.spike_buffer, dtype=default_float)
                ),  # spike_buffer noise
                ElementWiseMul(
                    jnp.zeros_like(state.buffer_index, dtype=state.buffer_index.dtype)
                ),  # buffer_index noise
            )
        )

    @property
    def noise_shape(self):
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
            jax.ShapeDtypeStruct(shape=(self.N_neurons,), dtype=default_float),
            jax.ShapeDtypeStruct(
                shape=(self.buffer_size, self.N_neurons + self.N_inputs),
                dtype=default_float,
            ),
            jax.ShapeDtypeStruct(shape=(), dtype=default_float),
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
        state = self.clip_weights(t, state, args)
        return state

    def clip_weights(self, t, state, args):
        """Clip weights to be non-negative."""
        return eqx.tree_at(
            lambda s: s.W,
            state,
            jnp.clip(state.W, min=0.0),
        )

    def get_delayed_spikes(self, state: LIFState) -> Array:
        """Retrieve spikes from buffer according to delay matrix.

        Returns:
            Array of shape (N_neurons, N_neurons + N_inputs) with delayed spike values
        """
        # Cast buffer_index to int32 for indexing operations
        buffer_idx = jnp.round(state.buffer_index).astype(jnp.int32)

        # For each synapse, compute which buffer index to read from
        # Convert delays from seconds to buffer timesteps
        delay_steps = jnp.round(self.synaptic_delay_matrix / self.dt).astype(jnp.int32)

        # buffer_index points to most recent, we need to go back by delay amount
        read_indices = ((buffer_idx - delay_steps) % self.buffer_size).astype(jnp.int32)

        # Gather spikes from buffer for each synapse
        # Use vmap to vectorize over neurons
        def get_neuron_inputs(neuron_idx):
            return state.spike_buffer[
                read_indices[neuron_idx], jnp.arange(self.N_neurons + self.N_inputs)
            ]

        delayed_spikes = jax.vmap(get_neuron_inputs)(jnp.arange(self.N_neurons))
        return delayed_spikes

    def spike_and_reset(self, t, state: LIFState, args):
        V, _, W, G = state.V, state.S, state.W, state.G
        recurrent_spikes = (V > self.firing_threshold).astype(V.dtype)
        V_new = (1.0 - recurrent_spikes) * V + recurrent_spikes * self.V_reset

        input_sp = args["get_input_spikes"](t, state, args)
        spikes = jnp.concatenate((recurrent_spikes, input_sp), dtype=V.dtype)

        # Cast buffer_index to int32 to ensure it's an integer for indexing
        buffer_idx = jnp.round(state.buffer_index).astype(jnp.int32)

        # Update spike buffer with current spikes
        state = eqx.tree_at(
            lambda s: s.spike_buffer,
            state,
            state.spike_buffer.at[buffer_idx].set(spikes),
        )
        new_buffer = state.spike_buffer
        new_buffer_index = jnp.round(
            (state.buffer_index + 1) % self.buffer_size
        ).astype(jnp.int32)

        # Get delayed spikes and update conductances based on delayed activity
        delayed_spikes = self.get_delayed_spikes(state)
        conductance_values = (
            G + delayed_spikes * self.synaptic_increment
        )  # Raw conductance update, does not consider connectivity
        G_new = jnp.where(
            W == -jnp.inf, 0.0, conductance_values
        )  # Only update conductances for existing connections, else set to 0

        time_since_last_spike = jnp.where(
            recurrent_spikes > 0, 0.0, state.time_since_last_spike
        )  # Reset time since last spike to 0 for neurons that spiked

        return LIFState(
            V_new, spikes, W, G_new, time_since_last_spike, new_buffer, new_buffer_index
        )

    def compute_balance(self, t, state, args):
        """Compute the ratio of total inhibitory to excitatory input weights for each neuron."""
        weights = jnp.where(
            state.W == -jnp.inf, 0.0, state.W
        )  # Treat non-existing connections as weight 0 for balance computation

        # Create masks for existing connections
        existing_E = jnp.isfinite(state.W) & self.excitatory_mask[None, :]
        existing_I = jnp.isfinite(state.W) & ~self.excitatory_mask[None, :]

        N_E_connections = jnp.sum(existing_E, axis=1)
        N_I_connections = jnp.sum(existing_I, axis=1)

        mean_excitatory_weights = jnp.sum(weights * existing_E, axis=1) / (
            N_E_connections + 1e-12
        )
        mean_inhibitory_weights = jnp.sum(weights * existing_I, axis=1) / (
            N_I_connections + 1e-12
        )

        balance = mean_inhibitory_weights / (mean_excitatory_weights + 1e-12)

        # In case either a neuron does not have both E and I connections, the balance is not defined
        balance = jnp.where(
            (N_E_connections == 0) | (N_I_connections == 0), jnp.nan, balance
        )
        return balance

    def force_balanced_weights(self, t, state, args):
        """Adjust weights to achieve a desired E/I balance for each neuron"""
        desired_balance = args["get_desired_balance"](t, state, args)
        current_balance = self.compute_balance(t, state, args)
        adjust_ratio = desired_balance / current_balance

        adjust_ratio = jnp.where(
            (adjust_ratio == 0.0) | (~jnp.isfinite(adjust_ratio)), 1.0, adjust_ratio
        )

        # Scale inhibitory weights to achieve desired balance
        balanced_weights = state.W * (
            jnp.outer(adjust_ratio, jnp.invert(self.excitatory_mask))
            + jnp.outer(jnp.ones(self.N_neurons), self.excitatory_mask)
        )
        return eqx.tree_at(lambda s: s.W, state, balanced_weights)
