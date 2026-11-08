import sys
from pathlib import Path

from jax import numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.snellius._templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.pendulum_main.run"

lr_map = {
    None: 707,  # 1414 unstable, 1000 stable
    -9: 707,  # 1414 unstable, 1000 stable
    -11: 1000,  # 2000 unstable, 1414 stable
    -13: 1000,  # 2000 unstable,1414 stable
    jnp.inf: 125,  # 250 unstable, 176 stable
}


def param_grid():
    for delta_V_exponent in [None, -9, -11, -13, jnp.inf]:
        model = (
            "EligibilityLIFNetwork" if delta_V_exponent is None else "GatedLIFNetwork"
        )
        lr = lr_map[delta_V_exponent]
        name = f"{model}_delta_V_{delta_V_exponent}_pendulum_AC"
        yield (
            name,
            {
                "--delta_V_exponent": delta_V_exponent,
                "--lr": lr,
            },
        )


def main():
    log_dir, results_dir = parse_output_dir()
    jobs = build_jobs(
        MODULE_PATH, param_grid(), log_dir, results_dir, gpu=True, num_cores=16
    )
    submit(jobs, gpu=True)


if __name__ == "__main__":
    main()
