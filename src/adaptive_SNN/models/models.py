import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import equinox as eqx
from dataclasses import field
from jaxtyping import Array, PyTree, Shaped
from typing import Callable
from abc import abstractmethod

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

class OUP(eqx.Module):
    theta: float = 1.0
    noise_scale: float = 1.0
    dim : int = 1

    @property
    def initial(self):
        return jnp.zeros((self.dim,))

    def drift(self, t, x, args):
        return -self.theta * x

    def diffusion(self, t, x, args):
        return jnp.eye(self.dim) * self.noise_scale

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(self.dim, ), dtype=default_float)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))
    

class NeuronModel(eqx.Module):
    leak_conductance: float = 16.7 # nS
    membrane_conductance: float = 250 # pF
    resting_potential: float = -70.0 # mV
    connection_prob: float = 0.1
    reversal_potential_E: float = 0.0 # mV
    reversal_potential_I: float = 75.0 # mV
    tau_E: float = 2.0 # ms
    tau_I: float = 6.0 # ms
    synaptic_increment: float = 1.0
    firing_threshold: float = -50.0 # mV
    V_reset: float = -60.0 # mV
    
    N_neurons: int = eqx.field(static=True)
    N_inputs: int = eqx.field(static=True)
    excitatory_mask: Array  # Vector of size N_neurons + N_inputs with: 1 (excitatory) and 0 (inhibitory)
    synaptic_time_constants: Array # Vector of size N_neurons with synaptic time constants (tau_E or tau_I)
    weights: Array # Shape (N_neurons, N_neurons + N_inputs)
    
    
    def __init__(self, N_neurons: int, N_inputs: int = 10, key: jr.PRNGKey = jr.PRNGKey(0)):
        self.N_neurons = N_neurons
        self.N_inputs = N_inputs
        key, key_1, key_2, key_3 = jr.split(key, 4)
        self.weights = jr.normal(key_1, (N_neurons, N_neurons + N_inputs)) * jr.bernoulli(key_2, self.connection_prob, (N_neurons, N_neurons + N_inputs)) *0.1 # nS
        neuron_types = jnp.where(jr.bernoulli(key_3, 0.8, (N_neurons,)), 1, 0) # 80% excitatory, 20% inhibitory
        self.excitatory_mask = jnp.concatenate([neuron_types, jnp.ones((N_inputs,))]) # inputs are all excitatory
        self.synaptic_time_constants = jnp.where(self.excitatory_mask, self.tau_E, self.tau_I)
    
    @property
    def initial(self):
        V_init = jnp.zeros((self.N_neurons,)) + self.resting_potential
        conductance_init = jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs))
        return (V_init, conductance_init)
    
    def drift(self, t, x, args):
        V, conductances = x

        # Compute leak current
        leak_current = -self.leak_conductance * (V - self.resting_potential)

        # Compute E/I currents from recurrent connections
        weighted_conductances = self.weights * conductances
        inhibitory_mask = jnp.where(self.excitatory_mask, 0, 1)
        inhibitory_conductances = jnp.sum(weighted_conductances * inhibitory_mask[None, :], axis=1)
        excitatory_conductances = jnp.sum(weighted_conductances * self.excitatory_mask[None, :], axis=1)

        # Get noise from args and add to total conductances
        if args and 'inhibitory_noise' in args:
            inhibitory_conductances += args['inhibitory_noise'](t, x, args)
        if args and 'excitatory_noise' in args:
            excitatory_conductances += args['excitatory_noise'](t, x, args)

        # Compute total recurrent current
        recurrent_current = inhibitory_conductances * (self.reversal_potential_I - V) + excitatory_conductances * (self.reversal_potential_E - V)

        dVdt = (leak_current + recurrent_current) / self.membrane_conductance

        dGdt = -1/self.synaptic_time_constants * conductances

        return dVdt, dGdt
    
    def diffusion(self, t, x, args):
        #TODO: should this be None instead of zeros? Seems wasteful to compute zeros every time
        #TODO: update: this should be removed entirely, since the neuron model itself is deterministic
        return (jnp.zeros((self.N_neurons,)), jnp.zeros((self.N_neurons, self.N_neurons + self.N_inputs)))
    
    @property
    def noise_shape(self):
        #TODO: same as above: should this be None instead of shapes of zeros?
        return (jax.ShapeDtypeStruct(shape=(self.N_neurons,), dtype=default_float),
                jax.ShapeDtypeStruct(shape=(self.N_neurons, self.N_neurons + self.N_inputs), dtype=default_float))
    
    def terms(self, key):
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)))
    
    def compute_spikes_and_update(self, t, x, args):
        V, G = x
        spikes = jnp.float32(V > self.firing_threshold)
        V_new = (1.0 - spikes) * V + spikes * self.V_reset # Reset voltage

        # Add input spikes if provided in args
        if args and 'input_spikes' in args:
            all_spikes = jnp.concatenate((spikes, args['input_spikes'](t, x, args)))
        elif self.N_inputs > 0:
            all_spikes = jnp.concatenate((spikes, jnp.zeros((self.N_inputs,))))
            raise RuntimeWarning("No input spikes provided to neuron model with inputs, assuming 0 input spikes.")
        
        G_new = G + all_spikes[None, :] * self.synaptic_increment # increase conductance on spike

        return V_new, G_new, spikes
    

class NoisyNeuronModel(eqx.Module):
    N_neurons: int = eqx.field(static=True) # TODO: this can be removed if we always get N_neurons from network
    network: NeuronModel
    noise_E: OUP
    noise_I: OUP

    def __init__(self,  N_neurons: int, neuron_model, noise_I_model, noise_E_model):
        self.N_neurons = N_neurons
        self.network = neuron_model
        self.noise_E = noise_E_model
        self.noise_I = noise_I_model

    @property
    def initial(self):
        return (self.network.initial, self.noise_E.initial, self.noise_I.initial)
    
    def drift(self, t, x, args):
        (V, conductances), noise_E_state, noise_I_state = x
        args =  {'excitatory_noise': lambda t, x, args: noise_E_state, 'inhibitory_noise': lambda t, x, args: noise_I_state}
        network_drift = self.network.drift(t, (V, conductances), args)
        noise_E_drift = self.noise_E.drift(t, noise_E_state, args)
        noise_I_drift = self.noise_I.drift(t, noise_I_state, args)
        return (network_drift, noise_E_drift, noise_I_drift)
    
    def diffusion(self, t, x, args):
        (V, conductances), noise_E_state, noise_I_state = x
        network_diffusion = self.network.diffusion(t, (V, conductances), args)
        noise_E_diffusion = self.noise_E.diffusion(t, noise_E_state, args)
        noise_I_diffusion = self.noise_I.diffusion(t, noise_I_state, args)
        return (network_diffusion, noise_E_diffusion, noise_I_diffusion)
    
    @property
    def noise_shape(self):
        return (self.network.noise_shape, self.noise_E.noise_shape, self.noise_I.noise_shape)
    
    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))