"""Log SLURM configuration at session start."""

from __future__ import annotations

from typing import Any

from pytask import Session, console, hookimpl


@hookimpl(trylast=True)
def pytask_log_session_header(session: Session) -> None:
    """Print a summary of SLURM settings."""
    config: dict[str, Any] = session.config
    parts: list[str] = []
    if config["slurm_partition"] is not None:
        parts.append(f"partition={config['slurm_partition']}")
    if config["slurm_time"] is not None:
        parts.append(f"time={config['slurm_time']}")
    if config["slurm_mem"] is not None:
        parts.append(f"mem={config['slurm_mem']}")
    if config["slurm_cpus_per_task"] is not None:
        parts.append(f"cpus_per_task={config['slurm_cpus_per_task']}")
    if config["slurm_gpus"] is not None:
        parts.append(f"gpus={config['slurm_gpus']}")
    if config["slurm_account"] is not None:
        parts.append(f"account={config['slurm_account']}")
    if config["slurm_qos"] is not None:
        parts.append(f"qos={config['slurm_qos']}")
    if config["slurm_extra"] is not None:
        parts.append(f"extra={config['slurm_extra']!r}")
    console.print(f"SLURM: {', '.join(parts)}")
