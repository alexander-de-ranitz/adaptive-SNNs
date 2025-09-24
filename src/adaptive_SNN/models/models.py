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
    num_inputs: int = eqx.field(static=True)
    excitatory_mask: Array  # Vector of size N_neurons with: 1 (excitatory) and 0 (inhibitory)
    synaptic_time_constants: Array # Vector of size N_neurons with synaptic time constants (tau_E or tau_I)
    input_weights: Array # Shape (N_neurons, num_inputs)
    recurrent_weights: Array # Shape (N_neurons, N_neurons)
    
    
    def __init__(self, N_neurons: int, num_inputs: int = 10, key: jr.PRNGKey = jr.PRNGKey(0)):
        self.N_neurons = N_neurons
        self.num_inputs = num_inputs
        key, key_1, key_2, key_3, key_4 = jr.split(key, 5)
        self.input_weights = jr.normal(key_1, (N_neurons, num_inputs)) * 0.1 # nS
        self.recurrent_weights = jr.normal(key_2, (N_neurons, N_neurons)) * jr.bernoulli(key_3, self.connection_prob, (N_neurons, N_neurons)) *0.1 # nS
        self.excitatory_mask = jnp.where(jr.bernoulli(key_4, 0.8, (N_neurons,)), 1, 0) # 80% excitatory, 20% inhibitory
        self.synaptic_time_constants = jnp.where(self.excitatory_mask, self.tau_E, self.tau_I)
    
    @property
    def initial(self):
        V_init = jnp.zeros((self.N_neurons,)) + self.resting_potential
        conductance_init = jnp.zeros((self.N_neurons, self.N_neurons))
        spikes_init = jnp.zeros((self.N_neurons,))
        return (V_init, conductance_init, spikes_init)
    
    def drift(self, t, x, args):
        V, conductances, spikes = x

        # TODO: add input current from args if provided

        # Compute leak current
        leak_current = -self.leak_conductance * (V - self.resting_potential)

        # Compute E/I currents from recurrent connections
        weighted_conductances = self.recurrent_weights * conductances
        inhibitory_mask = jnp.where(self.excitatory_mask, 0, 1)
        inhibitory_conductances = jnp.sum(weighted_conductances * inhibitory_mask, axis=1)
        excitatory_conductances = jnp.sum(weighted_conductances * self.excitatory_mask, axis=1)

        # Get noise from args and add to total conductances
        if args and 'inhibitory_noise' in args:
            inhibitory_conductances += args['inhibitory_noise'](t, x, args)
        if args and 'excitatory_noise' in args:
            excitatory_conductances += args['excitatory_noise'](t, x, args)

        # Compute total recurrent current
        recurrent_current = inhibitory_conductances * (self.reversal_potential_I - V) + excitatory_conductances * (self.reversal_potential_E - V)

        dVdt = (leak_current + recurrent_current) / self.membrane_conductance

        dGdt = -1/self.synaptic_time_constants * conductances

        return dVdt, dGdt, jnp.zeros_like(spikes)
    

    def diffusion(self, t, x, args):
        #TODO: should this be None instead of zeros? Seems wasteful to compute zeros every time
        return (jnp.zeros((self.N_neurons,)), jnp.ones((self.N_neurons, self.N_neurons)), jnp.zeros((self.N_neurons,)))
    
    @property
    def noise_shape(self):
        #TODO: same as above: should this be None instead of shapes of zeros?
        return (jax.ShapeDtypeStruct(shape=(self.N_neurons,), dtype=default_float),
                jax.ShapeDtypeStruct(shape=(self.N_neurons, self.N_neurons), dtype=default_float),
                jax.ShapeDtypeStruct(shape=(self.N_neurons,), dtype=default_float))
    
    def terms(self, key):
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)))
    

class NoisyNeuronModel(eqx.Module):
    N_neurons: int = eqx.field(static=True)
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
        (V, conductances, spikes), noise_E_state, noise_I_state = x
        args =  {'excitatory_noise': lambda t, x, args: noise_E_state, 'inhibitory_noise': lambda t, x, args: noise_I_state}
        network_drift = self.network.drift(t, (V, conductances, spikes), args)
        noise_E_drift = self.noise_E.drift(t, noise_E_state, args)
        noise_I_drift = self.noise_I.drift(t, noise_I_state, args)
        return (network_drift, noise_E_drift, noise_I_drift)
    
    def diffusion(self, t, x, args):
        (V, conductances, spikes), noise_E_state, noise_I_state = x
        network_diffusion = self.network.diffusion(t, (V, conductances, spikes), args)
        noise_E_diffusion = self.noise_E.diffusion(t, noise_E_state, args)
        noise_I_diffusion = self.noise_I.diffusion(t, noise_I_state, args)
        return (network_diffusion, noise_E_diffusion, noise_I_diffusion)
    
    @property
    def noise_shape(self):
        return (self.network.noise_shape, self.noise_E.noise_shape, self.noise_I.noise_shape)
    
    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))