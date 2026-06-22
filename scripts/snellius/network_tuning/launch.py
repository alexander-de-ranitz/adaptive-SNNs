import argparse
import shlex
import sys
from pathlib import Path

from qcg.pilotjob.api.job import Jobs
from qcg.pilotjob.api.manager import LocalManager


def create_jobs():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    module_path = "scripts.snellius.network_tuning.run_network"

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

    seed = 0
    num_iterations = 1
    balances = [1.0, 1.02]
    initial_E_weight = [2.0, 3.0]
    for i in range(num_iterations):
        for b in balances:
            for w in initial_E_weight:
                id = f"b_{b:6f}_w_{w:6f}"
                name = id + f"_seed_{seed}"

                cmd_args = [
                    "--balance",
                    str(b),
                    "--key_seed",
                    str(seed),
                    "--initial_E_weight",
                    str(w),
                    "--output_file",
                    str(results_dir / name),
                ]

                # Create a bash script to set PYTHONPATH and run the processing script
                bash_script = "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        f'export PYTHONPATH={shlex.quote(str(base_dir))}:"${{PYTHONPATH:-}}"',
                        f"{shlex.quote(sys.executable)} -m {shlex.quote(module_path)} {' '.join(shlex.quote(arg) for arg in cmd_args)}",
                    ]
                )

                jobs.add_std(
                    name=name,
                    execution={
                        "script": bash_script,
                        "stdout": str(log_dir / f"job.out.{name}"),
                        "stderr": str(log_dir / f"job.err.{name}"),
                        "wd": str(base_dir),
                    },
                    resources={
                        "numNodes": 1,
                        "numCores": {"exact": 18},
                        "nodeCrs": {"gpu": 1},
                    },
                )
        seed += 12345
    return jobs


def main():
    jobs = create_jobs()
    manager = LocalManager(
        server_args=[
            "--net",
            "--disable-nl",
            "--resources",
            "slurm",
            "--envschema",
            "slurm",
        ],
    )

    print(f"Submitting {len(jobs.jobs())} jobs to the local manager...")
    manager.submit(jobs)
    manager.wait4all()
    manager.finish()


if __name__ == "__main__":
    main()
