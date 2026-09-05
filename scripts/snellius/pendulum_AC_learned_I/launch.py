import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.snellius.templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.pendulum_AC_learned_I.run"


def param_grid():
    seed = 0
    for model in ["gated"]:
        for n_parallel in [4]:
            name = f"{model}_pendulum_AC_seed_{seed}_Nparallel_{n_parallel}"
            yield (
                name,
                {
                    "--model": model,
                    "--N_parallel": n_parallel,
                    "--key_seed": seed,
                },
            )


def main():
    log_dir, results_dir = parse_output_dir()
    jobs = build_jobs(
        MODULE_PATH, param_grid(), log_dir, results_dir, gpu=True, num_cores=18
    )
    submit(jobs, gpu=True)


if __name__ == "__main__":
    main()
