from abc import ABC, abstractmethod
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from adaptive_SNN.models.noise.base import NoiseModelABC
from adaptive_SNN.models.noise.oup import OUP
from adaptive_SNN.utils.operators import (
    DefaultIfNone,
    ElementWiseMul,
    MixedPyTreeOperator,
)

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class NeuronModelABC(ABC, eqx.Module):
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


class LIFState(eqx.Module):
    """State container for LIF network.

    Used to make it easier to work with the state by avoiding tuples with many indices.

    Attributes:
        V: Membrane potentials (N_neurons,)
        S: Spike vector (N_neurons + N_inputs,)
        W: Synaptic weight matrix (N_neurons, N_neurons + N_inputs). NaN indicates no connection
        G: Synaptic conductances (N_neurons, N_neurons + N_inputs)
        auxiliary_info: AuxiliaryInfo object containing additional state variables
    """

    V: Array
    S: Array
    W: Array
    G: Array
    perturbations: Array
    firing_rate: Array
    mean_E_conductance: Array
    var_E_conductance: Array
    time_since_last_spike: Array
    spike_buffer: Array
    buffer_index: Array
    features: Any


class AbstractLIFNetwork(NeuronModelABC):
    """Leaky Integrate-and-Fire (LIF) neuron network model with conductance-based synapses."""

    # fmt: off

    ###########################################
    #           Biophyisical Parameters       # 
    ###########################################
    leak_conductance: float = 16.7 * 1e-9  # nS
    membrane_capacitance: float = 250 * 1e-12  # pF
    resting_potential: float = -70.0 * 1e-3  # mV
    connection_prob_E: float = 0.1
    connection_prob_I: float = 0.2
    reversal_potential_E: float = 0.0  # mV
    reversal_potential_I: float = -80.0 * 1e-3  # mV
    tau_E: float = 6.0 * 1e-3  # ms
    tau_I: float = 6.0 * 1e-3  # ms
    synaptic_increment: float = 1.0 * 1e-9  # nS
    firing_threshold: float = -50.0 * 1e-3  # mV
    V_reset: float = -60.0 * 1e-3  # mV
    refractory_period: float = 2.0 * 1e-3  # ms
    mean_synaptic_delay: float = 1.5 * 1e-3  # ms

    ###########################################
    #             Network Parameters          # 
    ###########################################
    N_neurons: int = 1
    N_inputs: int = 0
    fraction_excitatory_recurrent: float = 0.8
    fraction_excitatory_input: float = 1.0
    initial_input_weight: float = 1.0  # Mean input weight
    rec_weight_std: float  = 0.0 # Standard deviation of initial recurrent weights as fraction of mean weight
    initial_rec_weight: float  = 1.0 # Mean recurrent weight
    initial_weight_matrix: Array | None = None # Optional initial weight matrix of shape (N_neurons, N_neurons + N_inputs)
    fully_connected_input: bool  = False  # If True, all input neurons connect to all neurons with weight initial_input_weight
    input_types: Array | None  = None # Optional binary vector of size N_inputs with: 1 (excitatory) and 0 (inhibitory)
    excitatory_mask: Array  = eqx.field(init=False) # Binary vector of size N_neurons + N_inputs with: 1 (excitatory) and 0 (inhibitory)
    synaptic_time_constants: Array = eqx.field(init=False) # Vector of size N_neurons + N_inputs with synaptic time constants (tau_E or tau_I)
    synaptic_delay_matrix: Array = eqx.field(init=False) # Matrix of shape (N_neurons, N_neurons) with synaptic delays for recurrent connections

    ###########################################
    #               Miscellaneous             # 
    ###########################################
    dt: float  = 1e-4  # Timestep size for simulation in seconds, used for delay buffer
    key: Array  = eqx.field(default_factory=lambda: jr.PRNGKey(0))  # Random key initialization
    EMA_tau: float = 1  # Time constant for exponential moving average of firing rate and mean/var of conductance
    buffer_size: int  = 1 # Size of spike history buffer
    noise_model: NoiseModelABC | None = None # Noise model to add noise to the network
    min_noise_std: float = 0.0 # Minimum std of noise to prevent it from going to zero when synaptic variance is low

    # fmt: on

    def __post_init__(self):
        """Post-initialization to set up neuron types, synaptic delays, and other derived parameters."""

        # Initialize noise model
        if self.noise_model is None:
            self.noise_model = OUP(dim=self.N_neurons, tau=self.tau_E)

        # Set up neuron types for recurrent connections
        key, subkey = jr.split(self.key)
        neuron_types = jr.permutation(
            subkey,
            jnp.concatenate(
                [
                    jnp.ones(
                        (
                            int(
                                jnp.round(
                                    self.fraction_excitatory_recurrent * self.N_neurons
                                )
                            ),
                        ),
                        dtype=bool,
                    ),
                    jnp.zeros(
                        (
                            self.N_neurons
                            - int(
                                jnp.round(
                                    self.fraction_excitatory_recurrent * self.N_neurons
                                )
                            ),
                        ),
                        dtype=bool,
                    ),
                ]
            ),
        )
        # Set up neuron types for input connections
        key, subkey = jr.split(key)
        if self.input_types is not None:
            if self.input_types.shape != (self.N_inputs,):
                raise ValueError(
                    f"input_types must have shape ({self.N_inputs},), but has shape {self.input_types.shape}"
                )
            input_neuron_types = self.input_types
        else:
            N_E_inputs = int(jnp.round(self.N_inputs * self.fraction_excitatory_input))
            N_I_inputs = self.N_inputs - N_E_inputs
            input_neuron_types = jr.permutation(
                subkey,
                jnp.concatenate(
                    [
                        jnp.ones(shape=(N_E_inputs,), dtype=bool),
                        jnp.zeros(shape=(N_I_inputs,), dtype=bool),
                    ]
                ),
            )

        # Define excitatory mask and synaptic time constants for all connections
        self.excitatory_mask = jnp.concatenate(
            [neuron_types, input_neuron_types], dtype=bool
        )
        self.synaptic_time_constants = jnp.where(
            self.excitatory_mask, self.tau_E, self.tau_I
        )

        # Set synaptic delays
        # TODO: Tune this properly
        key, subkey = jr.split(key)
        self.synaptic_delay_matrix = jr.uniform(
            subkey,
            shape=(self.N_neurons, self.N_neurons),
            minval=0.0,
            maxval=2 * self.mean_synaptic_delay,
            dtype=default_float,
        )

        # Compute buffer size based on max delay
        self.buffer_size = int(
            jnp.ceil(jnp.max(self.synaptic_delay_matrix) / self.dt) + 1
        )

        # Store key to be used for weight initialization
        # weights are initialised in the initial property as weights are part of the state
        # but we need to set the key to ensure consistent key handling
        self.key = key

    @property
    def initial(self):
        """Return initial network state as LIFState."""

        # Initialize weights
        key, subkey = jr.split(self.key)
        if self.initial_weight_matrix is not None:
            if self.initial_weight_matrix.shape != (
                self.N_neurons,
                self.N_neurons + self.N_inputs,
            ):
                raise ValueError(
                    f"initial_weight_matrix must have shape ({self.N_neurons}, {self.N_neurons + self.N_inputs}), but has shape {self.initial_weight_matrix.shape}"
                )
            weights = self.initial_weight_matrix
        else:
            weights = self.initialize_weights(subkey)

        # Initialize other state variables to zeros
        V_init = (
            jnp.zeros((self.N_neurons,), dtype=default_float) + self.resting_potential
        )
        perturbations_init = self.noise_model.initial
        conductance_init = jnp.zeros(
            (self.N_neurons, self.N_neurons + self.N_inputs), dtype=default_float
        )
        spikes_init = jnp.zeros((self.N_neurons,), dtype=default_float)

        # Initialize auxiliary info
        firing_rate = jnp.zeros((self.N_neurons,), dtype=default_float)
        time_since_last_spike = (
            jnp.ones((self.N_neurons,), dtype=default_float) * jnp.inf
        )  # For refractory period, set to inf initially so neurons can spike right away
        spike_buffer = jnp.zeros(
            (self.buffer_size, self.N_neurons), dtype=default_float
        )
        buffer_index = jnp.array(0, dtype=default_float)
        mean_E_conductance = jnp.zeros((self.N_neurons,), dtype=default_float)
        std_E_conductance = jnp.zeros((self.N_neurons,), dtype=default_float)

        return LIFState(
            V=V_init,
            S=spikes_init,
            W=weights,
            G=conductance_init,
            perturbations=perturbations_init,
            firing_rate=firing_rate,
            time_since_last_spike=time_since_last_spike,
            spike_buffer=spike_buffer,
            buffer_index=buffer_index,
            mean_E_conductance=mean_E_conductance,
            var_E_conductance=std_E_conductance,
            features=self.init_features(),
        )

    @abstractmethod
    def init_features(self):
        raise NotImplementedError

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

        # Compute derivatives of the state variables
        dV = self.compute_voltage_update(t, state, args)
        dG = -1 / self.synaptic_time_constants[None, :] * state.G
        dW = self.compute_weight_updates(t, state, args)
        dS = jnp.zeros_like(state.S)  # Spikes are handled separately, so no change here

        # Perturbations drift is defined by the noise model
        d_perturbations = self.noise_model.drift(t, state.perturbations, args)

        # Time since last spike increases at rate 1
        d_time_since_last_spike = jnp.ones_like(state.time_since_last_spike)

        # Firing rate is modelled as an exponential moving average of spikes
        d_firing_rate = -state.firing_rate / self.EMA_tau

        # Compute total excitatory synaptic conductance per neuron
        W, G = state.W, state.G
        weighted_conductances = jnp.where(~jnp.isnan(W), W, 0.0) * G
        total_E_conductance_per_neuron = jnp.sum(
            weighted_conductances * self.excitatory_mask[None, :], axis=1
        )

        # Update mean and variance of excitatory conductance as exponential moving averages
        d_mean_E_conductance = (
            -state.mean_E_conductance + total_E_conductance_per_neuron
        ) / self.EMA_tau
        d_var_E_conductance = (
            -state.var_E_conductance
            + jnp.square(total_E_conductance_per_neuron - state.mean_E_conductance)
        ) / self.EMA_tau

        # Buffer fields have zero derivative (they are updated in spike_and_reset)
        d_spike_buffer = jnp.zeros_like(state.spike_buffer)
        d_buffer_index = jnp.zeros_like(
            state.buffer_index,
            dtype=state.buffer_index.dtype,
        )

        return LIFState(
            V=dV,
            S=dS,
            W=dW,
            G=dG,
            perturbations=d_perturbations,
            firing_rate=d_firing_rate,
            mean_E_conductance=d_mean_E_conductance,
            var_E_conductance=d_var_E_conductance,
            time_since_last_spike=d_time_since_last_spike,
            spike_buffer=d_spike_buffer,
            buffer_index=d_buffer_index,
            features=self.compute_feature_drift(t, state, args),
        )

    @abstractmethod
    def compute_feature_drift(self, t, state: LIFState, args):
        raise NotImplementedError

    def diffusion(self, t, state: LIFState, args) -> MixedPyTreeOperator:
        """Define how noise enters the system

        No noise is currently used, but this function is defined for compatability and future use.
        The diffusion is defined as a Lineax operator (although it is not a linear operator in this case).
        It is called as diffusion().mv(noise) where the noise has the shape defined by self.noise_shape.
        The operator used here return zeros of the same shape as the state if the noise is None, else it returns
        the element-wise multiplication of the noise with zeros. This allows for noise to be a combination of None's and arrays,
        while ensuring the output shape exactly matches the state shape.

        Args:
            t: time
            state: LIFState current state
            args: dict

        Returns:
            MixedPyTreeOperator defining how noise enters the system
        """

        # Start with zero/no-op diffusion everywhere and then fill in the stochastic parts.
        diffusion = jax.tree.map(
            lambda leaf: DefaultIfNone(
                default=jnp.zeros_like(leaf),
                else_do=ElementWiseMul(jnp.zeros_like(leaf, dtype=default_float)),
            ),
            state,
        )

        # Possible features must define their diffusion
        diffusion = eqx.tree_at(
            lambda s: s.features,
            diffusion,
            self.compute_feature_diffusion(t, state, args),
        )

        # Diffusion of the noise process is computed in the noise model,
        # but takes a state-dependent noise_std as input
        noise_std = self.compute_desired_noise_std(t, state, args)
        diffusion = eqx.tree_at(
            lambda s: s.perturbations,
            diffusion,
            self.noise_model.diffusion(t, state, args, noise_std=noise_std),
        )

        # Since diffusion can consist of a mix of matrices and Lineax operators,
        # we return it as a MixedPyTreeOperator which can handle this case.
        return MixedPyTreeOperator(diffusion)

    @abstractmethod
    def compute_feature_diffusion(self, t, state: LIFState, args):
        raise NotImplementedError

    @property
    def noise_shape(self):
        # No noise in this model, but we need to return a pytree of the same structure as the state
        return LIFState(
            V=None,
            S=None,
            W=None,
            G=None,
            perturbations=self.noise_model.noise_shape,
            firing_rate=None,
            mean_E_conductance=None,
            var_E_conductance=None,
            time_since_last_spike=None,
            spike_buffer=None,
            buffer_index=None,
            features=self.noise_shape_features(),
        )

    @abstractmethod
    def noise_shape_features(self):
        raise NotImplementedError

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

    def update(self, t, x: LIFState, args) -> LIFState:
        """Apply non-differential updates to the state, e.g. spikes, resets, balancing, etc."""
        perturbations = self.noise_model.update(t, x.perturbations, args)
        state = eqx.tree_at(lambda s: s.perturbations, x, perturbations)
        state = self.spike_and_reset(t, state, args)
        state = self.clip_weights(t, state, args)
        state = self.force_balanced_weights(t, state, args)
        return state

    def compute_voltage_update(self, t, state: LIFState, args):
        """Compute dV/dt for the LIF neurons, incorporating external noise if present.

        The change in membrane potential is computed based on leak currents and excitatory/inhibitory conductance:

        C_m * dV/dt = -g_L*(V - E_L) + g_E*(V - E_E) + g_I(V - E_I)

        Where g_L is the leak conductance, g_E and g_I are the total excitatory and inhibitory conductances,
        and E_L, E_E, E_I are the respective reversal potentials. If external excitatory noise is provided in args,
        it is added to the total excitatory conductance. The conductances are ensured to be non-negative.

        Args:
            t: Current time (unused)
            state: Current LIFState
            args: Dictionary of additional arguments, may contain:
                - excitatory_noise: Array of shape (N_neurons,) representing external excitatory conductance noise

        Returns:
            dV: Array of shape (N_neurons,) representing the time derivative of membrane potentials
        """
        V, W, G = state.V, state.W, state.G

        # Compute leak current
        leak_current = -self.leak_conductance * (V - self.resting_potential)

        # Compute E/I currents from recurrent connections
        weighted_conductances = (
            jnp.where(jnp.isnan(W), 0.0, W) * G
        )  # For non-existing connections, set weighted conductance to 0
        total_I_conductances = jnp.sum(
            weighted_conductances * jnp.invert(self.excitatory_mask[None, :]), axis=1
        )
        synaptic_E_conductances = jnp.sum(
            weighted_conductances * self.excitatory_mask[None, :], axis=1
        )
        E_noise = state.perturbations
        total_E_conductances = (
            synaptic_E_conductances + E_noise
        )  # Add external excitatory noise to total excitatory conductance

        # Ensure non-negative conductances
        total_E_conductances = jnp.clip(total_E_conductances, min=0.0)
        total_I_conductances = jnp.clip(total_I_conductances, min=0.0)

        # Compute total recurrent current
        recurrent_current = total_I_conductances * (
            self.reversal_potential_I - V
        ) + total_E_conductances * (self.reversal_potential_E - V)

        dV = (leak_current + recurrent_current) / self.membrane_capacitance

        # Neurons in refractory period do not change their membrane potential
        dV = jnp.where(state.time_since_last_spike < self.refractory_period, 0.0, dV)
        return dV

    @abstractmethod
    def compute_weight_updates(self, t, state: LIFState, args) -> Array:
        raise NotImplementedError

    def clip_weights(self, t, state, args):
        """Clip weights to be non-negative.

        Only applied to existing connections (weights != -inf).
        """
        return eqx.tree_at(
            lambda s: s.W,
            state,
            jnp.where(~jnp.isnan(state.W), jnp.clip(state.W, min=0.0), state.W),
        )

    def get_delayed_spikes(self, state: LIFState) -> Array:
        """Retrieve spikes from buffer according to delay matrix.

        The function computes which spikes to read from the spike buffer based on the synaptic delay matrix.
        For each synapse, it calculates the appropriate buffer index to read from by subtracting the delay (in timesteps)
        from the current buffer index. The spikes are then gathered from the buffer for each synapse.

        Returns:
            Array of shape (N_neurons, N_neurons) with delayed spike values
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
                read_indices[neuron_idx], jnp.arange(self.N_neurons)
            ]

        delayed_spikes = jax.vmap(get_neuron_inputs)(jnp.arange(self.N_neurons))
        return delayed_spikes

    def spike_and_reset(self, t, state: LIFState, args) -> LIFState:
        """Handle spiking and resetting of neurons.

        Neurons that cross the firing threshold emit a spike and have their membrane potential reset.
        The spike buffer is updated with the current spikes, and the buffer index is incremented.
        Synaptic conductances are updated based on delayed spikes from the buffer for recurrent connections
        and current input spikes for input connections.
        Auxiliary information such as firing rate and time since last spike are also updated.

        Args:
            t: Current time
            state: Current LIFState
            args: Dictionary of additional arguments, must contain:
                - get_input_spikes(t, state, args) -> Array of shape (N_neurons, N_inputs) representing current input spikes

        Returns:
            Updated LIFState after spiking and updates
        """
        V, W, G = state.V, state.W, state.G
        recurrent_spikes = (V > self.firing_threshold).astype(V.dtype)  # (N_neurons,)
        V_new = (1.0 - recurrent_spikes) * V + recurrent_spikes * self.V_reset

        # Cast buffer_index to int32 to ensure it's an integer for indexing
        buffer_idx = jnp.round(state.buffer_index).astype(jnp.int32)

        # Update spike buffer with current spikes
        state = eqx.tree_at(
            lambda s: s.spike_buffer,
            state,
            state.spike_buffer.at[buffer_idx].set(recurrent_spikes),
        )
        new_buffer = state.spike_buffer
        new_buffer_index = jnp.round(
            (state.buffer_index + 1) % self.buffer_size
        ).astype(state.buffer_index.dtype)

        # Get delayed spikes and update conductances based on delayed activity
        delayed_spikes = self.get_delayed_spikes(
            state
        )  # Only for the recurrent connections: shape = (N_neurons, N_neurons)

        # Update conductances from recurrent spikes
        G_new = G.at[:, : self.N_neurons].add(delayed_spikes * self.synaptic_increment)

        # Update conductances from input spikes
        input_spikes = args["get_input_spikes"](t, state, args)

        # Check input_spikes shape, to catch when using old shape assumptions
        if input_spikes.shape != (self.N_neurons, self.N_inputs):
            raise ValueError(
                f"Input spikes shape {input_spikes.shape} does not match expected shape {(self.N_neurons, self.N_inputs)}"
            )

        G_new = G_new.at[:, self.N_neurons :].add(
            input_spikes * self.synaptic_increment
        )

        G_new = jnp.where(
            jnp.isnan(W), 0.0, G_new
        )  # Only update conductances for existing connections, else set to 0

        time_since_last_spike = jnp.where(
            recurrent_spikes > 0, 0.0, state.time_since_last_spike
        )  # Reset time since last spike to 0 for neurons that spiked

        new_firing_rate = state.firing_rate + recurrent_spikes / self.EMA_tau

        return LIFState(
            V=V_new,
            S=recurrent_spikes,
            W=W,
            G=G_new,
            perturbations=state.perturbations,
            firing_rate=new_firing_rate,
            mean_E_conductance=state.mean_E_conductance,
            var_E_conductance=state.var_E_conductance,
            time_since_last_spike=time_since_last_spike,
            spike_buffer=new_buffer,
            buffer_index=new_buffer_index,
            features=self.compute_feature_update(t, state, args),
        )

    @abstractmethod
    def compute_feature_update(self, t, state: LIFState, args):
        raise NotImplementedError

    def compute_balance(self, t, state, args):
        """Compute the ratio of total inhibitory to excitatory input weights for each neuron.

        Balance is computed as:
            balance = (total_I_weights * |E_I - V_rest| * tau_I) / (total_E_weights * |E_E - V_rest| * tau_E)

        Returns NaN if a neuron has no non-zero excitatory or inhibitory connections.
        """
        weights = jnp.where(
            jnp.isnan(state.W), 0.0, state.W
        )  # Treat non-existing connections as weight 0 for balance computation

        # Create masks for existing connections
        existing_E = ~jnp.isnan(state.W) & self.excitatory_mask[None, :]
        existing_I = ~jnp.isnan(state.W) & ~self.excitatory_mask[None, :]

        N_E_connections = jnp.sum(existing_E, axis=1)
        N_I_connections = jnp.sum(existing_I, axis=1)

        # TODO: fix this. This is 1) a bit hacky and 2) bloated since we currently only use E external input
        if "N_simulated_E_inputs" in args:
            is_E_input = self.excitatory_mask[None, :] & jnp.concatenate(
                [
                    jnp.zeros((self.N_neurons,), dtype=bool),
                    jnp.ones((self.N_inputs,), dtype=bool),
                ]
            )
            existing_E = jnp.where(is_E_input, args["N_simulated_E_inputs"], existing_E)
            N_E_connections += (
                args["N_simulated_E_inputs"] - 1
            )  # -1 since the existing connections already count as 1
        if "N_simulated_I_inputs" in args:
            is_I_input = ~self.excitatory_mask[None, :] & jnp.concatenate(
                [
                    jnp.zeros((self.N_neurons,), dtype=bool),
                    jnp.ones((self.N_inputs,), dtype=bool),
                ]
            )
            existing_I = jnp.where(is_I_input, args["N_simulated_I_inputs"], existing_I)
            N_I_connections += args["N_simulated_I_inputs"]

        total_E_weights = jnp.sum(weights * existing_E, axis=1)
        total_I_weights = jnp.sum(weights * existing_I, axis=1)

        # If a neuron has no excitatory or no inhibitory connections, we cannot compute a balance, so we set it to NaN in that case
        balance = jnp.where(
            (total_E_weights == 0.0),
            jnp.nan,
            (
                total_I_weights
                * jnp.abs(self.reversal_potential_I - self.resting_potential)
                * self.tau_I
            )
            / (
                total_E_weights
                * jnp.abs(self.reversal_potential_E - self.resting_potential)
                * self.tau_E
            ),
        )

        return balance

    def force_balanced_weights(self, t, state, args):
        """Adjust weights to achieve a desired E/I balance for each neuron.

        Balance is enforced by scaling inhibitory weights for each neuron to achieve the desired balance given the current excitatory weights.
        If a neuron only has inhibitory connections of weight zero, all inhibitory weights are set to the same weight at which the desired balance is reached.
        If a neuron has zero excitatory weights, the balance cannot be achieved by scaling inhibitory weights, so no changes are made.
        To avoid changing weights, the desired balance can be set to 0.0, in which case no weights are modified.
        """

        desired_balance = args["get_desired_balance"](t, state, args)
        total_I_weights = jnp.sum(
            jnp.where(
                jnp.isnan(state.W),
                0.0,
                state.W * jnp.invert(self.excitatory_mask[None, :]),
            ),
            axis=1,
        )

        # If there are no non-zero inhibitory connections, we set all inhibitory weights to 1 to allow the balance to be set to the correct value
        state = eqx.tree_at(
            lambda s: s.W,
            state,
            jnp.where(
                (
                    desired_balance != 0.0
                )  # If desired balance is 0, we do not want to change the weights
                & (total_I_weights == 0.0)[:, None]
                & jnp.invert(self.excitatory_mask[None, :])
                & ~jnp.isnan(state.W),
                1.0,
                state.W,
            ),
        )

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

    def initialize_weights(self, key: jr.PRNGKey):
        """Initialize synaptic weight matrix with random sparse connections.

        No self-connections are allowed. Non-existing connections have weight NaN.
        """
        key, subkey, subkey2 = jr.split(key, 3)

        connection_prob = jnp.where(
            self.excitatory_mask[: self.N_neurons],
            self.connection_prob_E,
            self.connection_prob_I,
        )
        rec_mask = jr.bernoulli(
            subkey2, p=connection_prob[None, :], shape=(self.N_neurons, self.N_neurons)
        )

        rec_weights = jnp.where(
            rec_mask,
            self.initial_rec_weight
            * jnp.clip(
                1
                + self.rec_weight_std
                * jr.normal(
                    subkey, (self.N_neurons, self.N_neurons), dtype=default_float
                ),
                min=0.5,
                max=1.5,
            ),
            jnp.nan,
        )

        # Remove self-connections
        rec_weights = jnp.fill_diagonal(rec_weights, jnp.nan, inplace=False)

        key, subkey = jr.split(key)
        N_input_connections = (
            int(self.N_neurons * self.N_inputs * self.connection_prob_E)
            if not self.fully_connected_input
            else self.N_neurons * self.N_inputs
        )

        input_mask = jr.permutation(
            subkey,
            jnp.concatenate(
                [
                    jnp.ones(N_input_connections),
                    jnp.zeros(self.N_neurons * self.N_inputs - N_input_connections),
                ]
            ),
        ).reshape(self.N_neurons, self.N_inputs)

        input_weights = jnp.where(
            input_mask,
            self.initial_input_weight,
            jnp.nan,
        )

        weights = jnp.concatenate([rec_weights, input_weights], axis=1)
        return weights

    def compute_desired_noise_std(self, t, state: LIFState, args):
        """For each neuron, compute the desired scale of the noise to be added.

        If the noise_scale_hyperparam is zero, no noise is added. Otherwise, the desired noise std is given as:
            desired_noise_std = min_noise_std + noise_scale_hyperparam * sqrt(var_E_conductance)

        Returns:
            Array: Noise scale for each neuron.
        """
        synaptic_variance = state.var_E_conductance

        use_noise = args.get("use_noise", jnp.array([True] * self.N_neurons))

        # Compute desired noise std using the computed variance and a hyperparameter, then clip to min value
        noise_scale_hyperparam = args.get("noise_scale_hyperparam", 0.0)
        desired_noise_std = jnp.sqrt(synaptic_variance) * noise_scale_hyperparam
        desired_noise_std = self.min_noise_std + desired_noise_std
        desired_noise_std = jnp.where(use_noise, desired_noise_std, 0.0)
        return desired_noise_std
