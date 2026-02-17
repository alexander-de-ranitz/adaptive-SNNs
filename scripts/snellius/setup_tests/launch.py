import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    folder_name = "setup_tests"
    script_path = base_dir / "scripts" / "snellius" / folder_name / "run_simulation.py"

    parser = argparse.ArgumentParser(description="Launch setup tests jobs on Snellius")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store output files (overrides default)",
    )
    args = parser.parse_args()

    jobs = Jobs()

    log_dir = Path(args.output_dir) / "logs" if args.output_dir else base_dir / "logs"
    results_dir = (
        Path(args.output_dir) / "results" if args.output_dir else base_dir / "results"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    initial_weights = np.linspace(0.0, 15.0, 16)
    balances = [1.5, 1.75, 2.0, 2.25, 2.5]
    noise_levels = [0.1]
    base_key = 0
    key_i = 0
    for iw in initial_weights:
        for balance in balances:
            for nl in noise_levels:
                key_i += 1

                name = f"setup_tests_iw{iw:.2f}_balance{balance:.2f}_nl{nl:.2f}"
                args = [
                    "--initial_weight",
                    str(iw),
                    "--balance",
                    str(balance),
                    "--noise_level",
                    str(nl),
                    "--key_seed",
                    str(base_key + key_i),
                    "--output_file",
                    str(results_dir / name),
                ]

                job_name = name

                # Create a bash script to set PYTHONPATH and run the processing script
                bash_script = "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                        f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} {' '.join(shlex.quote(arg) for arg in args)}",
                    ]
                )

                jobs.add(
                    name=job_name,
                    script=bash_script,
                    stdout=str(log_dir / f"job.out.{job_name}"),
                    stderr=str(log_dir / f"job.err.{job_name}"),
                    wd=str(base_dir),
                    iteration=1,
                    numCores=1,
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
