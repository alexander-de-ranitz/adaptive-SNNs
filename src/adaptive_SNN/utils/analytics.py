from jax import numpy as jnp


def compute_required_input_weight(
    target_mean_g_syn: float,
    N_inputs: int,
    tau: float,
    input_rate: float,
    synaptic_increment: float,
) -> float:
    """Compute the necessary mean input weight to achieve a target mean synaptic conductance.

    Given a target mean synaptic conductance, the number of input neurons, the synaptic time constant,
    the average firing rate of the input neurons, and the synaptic increment per spike, this function
    computes the mean input weight required. The calculation assumes that the input neurons are independent
    Poisson processes with a uniform firing rate.

    Args:
        target_mean_g_syn (float): Target mean synaptic conductance (in Siemens).
        N_inputs (int): Number of input neurons.
        tau (float): Synaptic time constant (in seconds).
        input_rate (float): Average firing rate of input neurons (in Hz), assumed to be independent Poisson processes of uniform rate.
        synaptic_increment (float): Increment in conductance per spike (in Siemens).

    Returns:
        float: The computed mean input weight.
    """
    weight = target_mean_g_syn / (N_inputs * tau * input_rate * synaptic_increment)
    return weight


def compute_expected_synaptic_std(
    N_inputs: int,
    input_rate: float,
    tau: float,
    synaptic_increment: float,
    input_weight: float,
) -> float:
    """Compute the expected standard deviation of synaptic conductance fluctuations.

    This function calculates the expected standard deviation of synaptic conductance fluctuations
    based on the excitatory input firing rate, synaptic time constant, synaptic increment per spike,
    and input weight.

    Args:
        N_inputs (int): Number of input neurons.
        input_rate (float): Input firing rate (in Hz). (assmued to be independent Poisson processes of uniform rate)
        tau (float): Associated synaptic time constant (in seconds).
        synaptic_increment (float): Increment in conductance per spike (in Siemens).
        input_weight (float): Strength of input synapses, assumed uniform across all input neurons.

    Returns:
        float: The expected standard deviation of synaptic conductance fluctuations.
    """
    expected_syn_std = jnp.sqrt(
        0.5 * N_inputs * input_rate * tau * (synaptic_increment**2) * (input_weight**2)
    )
    return expected_syn_std


def compute_oup_diffusion_coefficient(target_std: float, tau: float) -> float:
    """Compute the diffusion coefficient for an Ornstein-Uhlenbeck process.

    This function calculates the diffusion coefficient required for an Ornstein-Uhlenbeck (OU)
    process to achieve a specified standard deviation of fluctuations, given the time constant.

    Args:
        target_std (float): Target standard deviation of fluctuations.
        tau (float): Time constant of the OU process (in seconds).

    Returns:
        float: The computed diffusion coefficient.
    """
    D = 2 * (target_std**2) / tau
    return D


def compute_mean_membrane_voltage(
    leak_conductance: float,
    g_E: float,
    g_I: float,
    resting_potential: float,
    reversal_potential_E: float,
    reversal_potential_I: float,
) -> float:
    """
    Compute the steady-state (expected) membrane potential V_i.

    Equation:
        V_i = (g_L * E_L + g_E * E_E + g_I * E_I) / (g_L + g_E + g_I)

    Parameters
    ----------
    leak_conductance : float
        Leak conductance (in siemens).
    g_E : float
        Total excitatory conductance (in siemens).
    g_I : float
        Total inhibitory conductance (in siemens).
    E_L : float
        Leak reversal potential (in volts).
    reversal_potential_E : float
        Excitatory reversal potential (in volts), e.g. 0 mV.
    reversal_potential_I : float
        Inhibitory reversal potential (in volts), e.g. -80 mV.

    Returns
    -------
    float
        Steady-state membrane potential (in volts).
    """
    total_g = leak_conductance + g_E + g_I
    V_i = (
        leak_conductance * resting_potential
        + g_E * reversal_potential_E
        + g_I * reversal_potential_I
    ) / total_g
    return V_i


def compute_required_I_conductance(
    target_mean_voltage: float,
    leak_conductance: float,
    g_E: float,
    resting_potential: float,
    reversal_potential_E: float,
    reversal_potential_I: float,
) -> float:
    """
    Compute the required inhibitory conductance g_I to achieve a target mean membrane voltage.

    Rearranged Equation:
        g_I = (g_L * (E_L - V_target) + g_E * (E_E - V_target)) / (V_target - E_I)

    Parameters
    ----------
    target_mean_voltage : float
        Target mean membrane voltage (in volts).
    leak_conductance : float
        Leak conductance (in siemens).
    g_E : float
        Total excitatory conductance (in siemens).
    resting_potential : float
        Leak reversal potential (in volts).
    reversal_potential_E : float
        Excitatory reversal potential (in volts), e.g. 0 mV.
    reversal_potential_I : float
        Inhibitory reversal potential (in volts), e.g. -80 mV.

    Returns
    -------
    float
        Required inhibitory conductance (in siemens).
    """
    numerator = leak_conductance * (resting_potential - target_mean_voltage) + g_E * (
        reversal_potential_E - target_mean_voltage
    )
    denominator = target_mean_voltage - reversal_potential_I
    g_I = numerator / denominator
    return g_I
