import argparse
import shlex
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    module_path = "scripts.snellius.delta_v_tuning.experiment_delta_v_tuning"

    parser = argparse.ArgumentParser(description="Launch jobs on Snellius")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/default_folder/",
        help="Directory to store output files",
    )
    cmd_args = parser.parse_args()

    log_dir = (
        Path(cmd_args.output_dir) / "logs" if cmd_args.output_dir else base_dir / "logs"
    )
    results_dir = (
        Path(cmd_args.output_dir) / "results"
        if cmd_args.output_dir
        else base_dir / "results"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = Jobs()

    seed = 1241
    num_iterations = 7
    for i in range(num_iterations):
        seed += 129
        # Loop over delta_V values, including 0.0 for the no-gating condition
        for delta_v in [0.0] + [0.5**k for k in range(5, 14)]:
            id = f"dv_{delta_v:6f}_" if type(delta_v) is float else "no_gating_"
            name = id + f"seed_{seed}"

            cmd_args = [
                "--delta_V",
                str(delta_v),
                "--key_seed",
                str(seed),
                "--output_file",
                str(results_dir / name),
            ]

            # Create a bash script to set PYTHONPATH and run the processing script
            bash_script = "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                    f"{shlex.quote(sys.executable)} -m {shlex.quote(module_path)} {' '.join(shlex.quote(arg) for arg in cmd_args)}",
                ]
            )

            jobs.add(
                name=name,
                script=bash_script,
                stdout=str(log_dir / f"job.out.{name}"),
                stderr=str(log_dir / f"job.err.{name}"),
                wd=str(base_dir),
                numCores=1,
                iteration=1,
            )

    return jobs


def main():
    jobs = create_jobs()
    manager = LocalManager()

    print(f"Submitting {len(jobs.jobs())} jobs to the local manager...")
    manager.submit(jobs)
    manager.wait4all()
    manager.finish()


if __name__ == "__main__":
    main()
