import argparse
import shlex
import sys
from pathlib import Path

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    folder_name = "rate_learning"
    script_path = (
        base_dir
        / "scripts"
        / "snellius"
        / folder_name
        / "grid_search"
        / "run_simulation.py"
    )

    parser = argparse.ArgumentParser(
        description="Launch rate learning jobs on Snellius"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store output files (overrides default)",
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

    cmd_args = parser.parse_args()
    initial_weights = [1]
    noise_levels = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    learning_rates = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    balance = 1.75
    base_key = 0
    key_i = 0
    for iw in initial_weights:
        for nl in noise_levels:
            for lr in learning_rates:
                # Uncomment the next line to use different random seeds for each job
                # key_i += 1

                name = f"results_iw{iw:.6f}_nl{nl:.6f}_lr{lr:.6f}"

                cmd_args = [
                    "--initial_weight",
                    str(iw),
                    "--noise_level",
                    str(nl),
                    "--learning_rate",
                    str(lr),
                    "--balance",
                    str(balance),
                    "--key_seed",
                    str(base_key + key_i),
                    "--output_file",
                    str(results_dir / name),
                ]

                job_name = f"rate_learning_iw{iw}_nl{nl}_lr{lr}"

                # Create a bash script to set PYTHONPATH and run the processing script
                bash_script = "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} {' '.join(shlex.quote(arg) for arg in cmd_args)}",
                    ]
                )

                jobs.add(
                    name=job_name,
                    script=bash_script,
                    stdout=str(log_dir / f"job.out.{job_name}"),
                    stderr=str(log_dir / f"job.err.{job_name}"),
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
