from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import diffrax as dfx
import equinox as eqx
from diffrax import SaveAt
from jax import numpy as jnp

from adaptive_SNN.models.agent_env_system import AgentEnvSystem
from adaptive_SNN.models.environments import SpikeRateEnvironment
from adaptive_SNN.models.environments.base import AbstractEnvironment
from adaptive_SNN.models.networks import Agent, LIFNetwork, NoisyNetwork
from adaptive_SNN.models.networks.base import NeuronModelABC
from adaptive_SNN.models.noise import OUP
from adaptive_SNN.models.noise.base import NoiseModelABC
from adaptive_SNN.models.reward import AbstractRewardModel, MovingAverageRewardModel
from adaptive_SNN.models.RPE import AbstractRPEModel, InstantRPEModel


@dataclass
class SimulationConfig:
    # Time parameters
    t0: float = 0.0
    t1: float = 100.0
    dt: float = 1e-4
    warmup_time: float = 10.0
    save_at: SaveAt = dfx.SaveAt()

    # Model classes
    base_network_cls: type[NeuronModelABC] = LIFNetwork
    noisy_network_cls: type[NeuronModelABC] = NoisyNetwork
    agent_cls: type[NeuronModelABC] = Agent
    agent_env_system_cls: type[NeuronModelABC] = AgentEnvSystem

    # Model hyperparameters
    N_neurons: int = 1
    N_inputs: int = 2
    connection_prob: float = 0.0
    noise_level: float = 0.0
    lr: float = 0.0
    initial_weight: float = 0.0
    balance: float = 0.0
    min_noise_std: float = 0.0
    fraction_excitatory_recurrent: float = 0.8
    fraction_excitatory_input: float = 0.8
    initial_weight_matrix: jnp.ndarray | None = None
    mean_synaptic_delay: float = 0.0
    base_network_kwargs: dict[str, Any] = eqx.field(default_factory=lambda: {})
    args: dict[str, Any] = eqx.field(default_factory=lambda: {})

    # Reward model
    reward_model: type[AbstractRewardModel] = MovingAverageRewardModel
    reward_kwargs: dict[str, Any] = eqx.field(default_factory=lambda: {})
    reward_fn: Callable[..., Any] | None = None
    reward_noise_model: type[NoiseModelABC] = OUP
    reward_noise_kwargs: dict[str, Any] = eqx.field(default_factory=lambda: {})
    RPE_model: type[AbstractRPEModel] = InstantRPEModel
    RPE_model_kwargs: dict[str, Any] = eqx.field(default_factory=lambda: {})

    # Environment parameters
    environment_model: type[AbstractEnvironment] = SpikeRateEnvironment
    environment_kwargs: dict[str, Any] = eqx.field(default_factory=lambda: {})

    # Input parameters
    input_spike_fn: Callable[..., Any] | None = None
    input_types: jnp.ndarray | None = None

    weight_std: float = 0.0
    fully_connected_input: bool = True

    # Other
    key: int | jnp.ndarray = 0
    save_file: str | None = None
    network_output_fn: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        self._validate_scalars()
        self._validate_probabilities()
        self._validate_arrays()
        self._validate_callables()

    def _validate_scalars(self) -> None:
        if self.t1 <= self.t0:
            raise ValueError(f"Expected t1 > t0, got t0={self.t0}, t1={self.t1}")
        if self.dt <= 0.0:
            raise ValueError(f"Expected dt > 0, got dt={self.dt}")
        if self.warmup_time < 0.0:
            raise ValueError(f"Expected warmup_time >= 0, got {self.warmup_time}")
        if self.N_neurons <= 0:
            raise ValueError(f"Expected N_neurons > 0, got {self.N_neurons}")
        if self.N_inputs < 0:
            raise ValueError(f"Expected N_inputs >= 0, got {self.N_inputs}")

    def _validate_probabilities(self) -> None:
        for name in (
            "connection_prob",
            "fraction_excitatory_recurrent",
            "fraction_excitatory_input",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Expected {name} in [0, 1], got {value}")

    def _validate_arrays(self) -> None:
        if self.initial_weight_matrix is not None:
            expected_shape = (self.N_neurons, self.N_neurons + self.N_inputs)
            if self.initial_weight_matrix.shape != expected_shape:
                raise ValueError(
                    "Expected initial_weight_matrix shape "
                    f"{expected_shape}, got {self.initial_weight_matrix.shape}"
                )

        if self.input_types is not None:
            if self.input_types.shape != (self.N_inputs,):
                raise ValueError(
                    f"Expected input_types shape {(self.N_inputs,)}, got {self.input_types.shape}"
                )

    def _validate_callables(self) -> None:
        for name in ("reward_fn", "input_spike_fn", "network_output_fn"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"Expected {name} to be callable or None, got {type(value)}"
                )

        if self.save_file is not None and not isinstance(self.save_file, str):
            raise TypeError(
                f"Expected save_file to be a string or None, got {type(self.save_file)}"
            )

    def normalized_save_file(self) -> str:
        if self.save_file is None:
            self.save_file = "default_simulation_result.npz"
        if self.save_file.endswith(".npz"):
            return self.save_file
        return f"{self.save_file}.npz"

    def ensure_output_directory(self) -> None:
        path = Path(self.normalized_save_file())
        path.parent.mkdir(parents=True, exist_ok=True)

    def print_to_file(self) -> None:
        log_path = Path(self.normalized_save_file()).with_suffix("")
        info_file = f"{log_path}_info.txt"

        with open(info_file, "w", encoding="utf-8") as f:
            for field in fields(self):
                if field.name == "save_at":
                    continue
                if field.name in ["network_output_fn", "input_spike_fn", "reward_fn"]:
                    f.write(f"{field.name}:\n")
                    value = getattr(self, field.name)
                    if value is not None:
                        f.write(inspect.getsource(value) + "\n")
                    else:
                        f.write("None\n")
                    continue
                f.write(f"{field.name}: {getattr(self, field.name)}\n")
