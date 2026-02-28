"""Log SLURM configuration at session start."""

from __future__ import annotations

from typing import Any

from pytask import Session, console, hookimpl


@hookimpl(trylast=True)
def pytask_log_session_header(session: Session) -> None:
    """Print a summary of SLURM settings."""
    config: dict[str, Any] = session.config
    parts = [f"partition={config['slurm_partition'] or 'default'}"]
    parts.append(f"time={config['slurm_time']}")
    parts.append(f"mem={config['slurm_mem']}")
    parts.append(f"cpus_per_task={config['slurm_cpus_per_task']}")
    parts.append(f"max_jobs={config['slurm_max_jobs']}")
    if config["slurm_account"]:
        parts.append(f"account={config['slurm_account']}")
    if config["slurm_qos"]:
        parts.append(f"qos={config['slurm_qos']}")
    if config["slurm_extra"]:
        parts.append(f"extra={config['slurm_extra']!r}")
    console.print(f"SLURM: {', '.join(parts)}")
