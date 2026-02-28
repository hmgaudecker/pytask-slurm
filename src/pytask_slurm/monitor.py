"""Monitor SLURM job statuses via sacct."""

from __future__ import annotations

import subprocess
from enum import Enum


class SlurmJobStatus(Enum):
    """SLURM job states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    UNKNOWN = "UNKNOWN"


_STATE_MAP: dict[str, SlurmJobStatus] = {
    "PENDING": SlurmJobStatus.PENDING,
    "RUNNING": SlurmJobStatus.RUNNING,
    "COMPLETED": SlurmJobStatus.COMPLETED,
    "FAILED": SlurmJobStatus.FAILED,
    "CANCELLED": SlurmJobStatus.CANCELLED,
    "TIMEOUT": SlurmJobStatus.TIMEOUT,
    "OUT_OF_MEMORY": SlurmJobStatus.OUT_OF_MEMORY,
    "NODE_FAIL": SlurmJobStatus.NODE_FAIL,
    # sacct sometimes reports these variants.
    "COMPLETING": SlurmJobStatus.RUNNING,
    "CONFIGURING": SlurmJobStatus.PENDING,
    "REQUEUED": SlurmJobStatus.PENDING,
    "SUSPENDED": SlurmJobStatus.PENDING,
    "PREEMPTED": SlurmJobStatus.FAILED,
}


def _parse_state(raw: str) -> SlurmJobStatus:
    """Parse a SLURM state string to our enum.

    SLURM sometimes appends suffixes like ``CANCELLED by 12345``.

    """
    token = raw.split()[0] if raw else ""
    return _STATE_MAP.get(token, SlurmJobStatus.UNKNOWN)


def poll_job_statuses(job_ids: list[str]) -> dict[str, SlurmJobStatus]:
    """Query sacct for the status of the given SLURM job IDs.

    Returns a mapping from job ID to status. Jobs not found in the output are
    omitted from the result (the caller should treat them as still pending).

    """
    if not job_ids:
        return {}

    cmd = [
        "sacct",
        "-X",  # no sub-steps
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,State",
        "--jobs=" + ",".join(job_ids),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}

    statuses: dict[str, SlurmJobStatus] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            job_id, state_str = parts[0], parts[1]
            statuses[job_id] = _parse_state(state_str)

    return statuses
