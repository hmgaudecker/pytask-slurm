"""Configure pytask-slurm."""

from __future__ import annotations

from typing import Any

from pytask import hookimpl


@hookimpl
def pytask_parse_config(config: dict[str, Any]) -> None:
    """Parse the configuration."""
    config["markers"]["slurm"] = (
        "Override SLURM resources for a task"
        " (partition, time, mem, cpus_per_task, account)."
    )
    config.setdefault("slurm", False)
    config.setdefault("slurm_partition", None)
    config.setdefault("slurm_time", "01:00:00")
    config.setdefault("slurm_mem", "4G")
    config.setdefault("slurm_cpus_per_task", 1)
    config.setdefault("slurm_max_jobs", 100)
    config.setdefault("slurm_poll_interval", 5.0)
    config.setdefault("slurm_account", None)


@hookimpl(trylast=True)
def pytask_post_parse(config: dict[str, Any]) -> None:
    """Register the SLURM executor if --slurm is active."""
    if not config["slurm"]:
        return

    if config["pdb"] or config["trace"] or config["dry_run"]:
        return

    from pytask_slurm import execute  # noqa: PLC0415

    config["pm"].register(execute)
