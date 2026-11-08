import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.snellius._templates.launch import build_jobs, parse_output_dir, submit

MODULE_PATH = "scripts.snellius.assumption_verification.run"


FIRST_ITER = 0
N_JOBS = 72 * 3


def param_grid():
    for i in range(FIRST_ITER, FIRST_ITER + N_JOBS):
        seed = 1234 + 4321 * (i + 1)
        name = f"perturbation_response_seed_{seed}_iter_{i}"
        yield (name, {"--key_seed": seed})


def main():
    log_dir, results_dir = parse_output_dir()
    jobs = build_jobs(MODULE_PATH, param_grid(), log_dir, results_dir)
    submit(jobs)


if __name__ == "__main__":
    main()
