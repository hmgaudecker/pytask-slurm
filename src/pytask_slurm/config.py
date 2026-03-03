"""Configure pytask-slurm."""

from __future__ import annotations

from typing import Any

from pytask import hookimpl


@hookimpl
def pytask_parse_config(config: dict[str, Any]) -> None:
    """Parse the configuration."""
    config["markers"]["slurm"] = (
        "Override SLURM resources for a task. Options: partition, time,"
        " mem, cpus_per_task, account, qos, gpus. Pass extra=\"...\" for any other"
        " sbatch option (e.g. extra=\"--constraint=a100 --mail-type=END\")."
    )
    config.setdefault("slurm", False)
    config.setdefault("slurm_partition", None)
    config.setdefault("slurm_time", None)
    config.setdefault("slurm_mem", None)
    config.setdefault("slurm_cpus_per_task", None)
    config.setdefault("slurm_max_jobs", None)
    config.setdefault("slurm_poll_interval", None)
    config.setdefault("slurm_account", None)
    config.setdefault("slurm_qos", None)
    config.setdefault("slurm_gpus", None)
    config.setdefault("slurm_extra", None)
    config.setdefault("slurm_unknown_timeout", None)


@hookimpl(trylast=True)
def pytask_post_parse(config: dict[str, Any]) -> None:
    """Register the SLURM executor if --slurm is active."""
    if not config["slurm"]:
        return

    if config["pdb"] or config["trace"] or config["dry_run"]:
        return

    from pytask_slurm import execute, logging  # noqa: PLC0415

    config["pm"].register(execute)
    config["pm"].register(logging)
