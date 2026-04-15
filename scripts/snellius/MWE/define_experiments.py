from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    learning_rate: float = 0.0
    min_noise_std: float = 1e-9
    reward_noise_rate: float = 2.0
    reward_noise_std: float = 1.0
    input_rate: float = 50
    RPE_decay_tau: float = 0.05
    delta_V: float = 10e-3


def generate_experiment_configs():
    # We have 1 default setup. We only ever change 1 parameter relative to the default instead of a full grid search
    default_config = ExperimentConfig()
    configs = []
    seen = set()

    def add_unique(cfg: ExperimentConfig):
        key = (
            cfg.learning_rate,
            cfg.min_noise_std,
            cfg.reward_noise_rate,
            cfg.reward_noise_std,
            cfg.input_rate,
            cfg.RPE_decay_tau,
            cfg.delta_V,
        )
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    add_unique(default_config)

    # Vary reward noise rate
    for rnl in [0.0, 2.0, 4.0]:
        add_unique(ExperimentConfig(reward_noise_rate=rnl))
    # Vary min_noise_std
    for mns in [1e-9, 2e-9, 5e-9]:
        add_unique(ExperimentConfig(min_noise_std=mns))
    # Vary reward_noise_std
    for rns in [0.5, 1.0, 2.0]:
        add_unique(ExperimentConfig(reward_noise_std=rns))
    # Vary input_rate
    for ir in [25, 50, 100]:
        add_unique(ExperimentConfig(input_rate=ir))
    # Vary RPE_decay_tau
    for rdt in [0.01, 0.05, 0.1]:
        add_unique(ExperimentConfig(RPE_decay_tau=rdt))

    return configs


def generate_experiment_configs_new():
    configs = []
    seen = set()

    def add_unique(cfg: ExperimentConfig):
        key = (
            cfg.learning_rate,
            cfg.min_noise_std,
            cfg.reward_noise_rate,
            cfg.reward_noise_std,
            cfg.input_rate,
            cfg.RPE_decay_tau,
            cfg.delta_V,
        )
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    for dv in [0.03, 0.01, 0.005, 0.0025, 0.001, 0.0005]:
        add_unique(
            ExperimentConfig(
                learning_rate=0.0,
                reward_noise_rate=2.0,
                reward_noise_std=2.0,
                min_noise_std=1e-9,
                input_rate=50,
                RPE_decay_tau=0.05,
                delta_V=dv,
            )
        )

    return configs
