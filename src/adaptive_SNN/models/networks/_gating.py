"""Shared voltage-gating function for credit assignment.

This is the per-neuron gate gamma(V_i) of unified §3.4 Eq. 3.8 / Alexander Eq. 28.
Both GatedLIFNetwork (eligibility-trace + gate) and OUAMeanReversionLIFNetwork
(OUA mean-reversion + gate) import this module-level function so that the gate
definition lives in one place.

The gate is normalised so that its integral over [V_rest, V_th] equals
(V_th - V_rest) -- the area under a constant-1 default gate. This means
average update magnitudes are comparable across delta_V settings.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def voltage_gate(
    voltage: Array,
    delta_V: float | Array,
    reversal_potential_E: float,
    firing_threshold: float,
    resting_potential: float,
) -> Array:
    """Voltage-gating function (Alexander Eq. 28; unified Eq. 3.8).

    gamma(V) = (E_E - V) / dV * exp((V - V_th) / dV)

    then normalised by (area / default_area) where default_area = V_th - V_rest.
    """
    default_area = 1.0 * (firing_threshold - resting_potential)
    driving_force = reversal_potential_E - voltage

    def integral(V_):
        return (reversal_potential_E + delta_V - V_) * -jnp.exp(
            (V_ - firing_threshold) / delta_V
        )

    area = integral(resting_potential) - integral(firing_threshold)
    gating = (
        driving_force / delta_V * jnp.exp((voltage - firing_threshold) / delta_V)
    )
    normalization_factor = area / default_area
    return gating / normalization_factor
