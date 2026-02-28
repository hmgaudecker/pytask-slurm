"""Monitor SLURM job statuses via sacct with squeue fallback."""

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


def _parse_status_lines(stdout: str) -> dict[str, SlurmJobStatus]:
    """Parse ``jobid|STATE`` lines into a status mapping."""
    statuses: dict[str, SlurmJobStatus] = {}
    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 2:  # noqa: PLR2004
            job_id, state_str = parts[0], parts[1]
            statuses[job_id] = _parse_state(state_str)
    return statuses


def _sacct_cmd(job_ids: list[str]) -> list[str]:
    return [
        "sacct",
        "-X",  # no sub-steps
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,State",
        "--jobs=" + ",".join(job_ids),
    ]


def _squeue_cmd(job_ids: list[str]) -> list[str]:
    return [
        "squeue",
        "--noheader",
        "--format=%i|%T",
        "--jobs=" + ",".join(job_ids),
    ]


def poll_job_statuses(job_ids: list[str]) -> dict[str, SlurmJobStatus]:
    """Query sacct for the status of the given SLURM job IDs.

    Falls back to squeue when sacct is unavailable or fails.
    Returns a mapping from job ID to status. Jobs not found in the output are
    omitted from the result (the caller should treat them as still pending).

    """
    if not job_ids:
        return {}

    # Try sacct first.
    try:
        result = subprocess.run(  # noqa: S603
            _sacct_cmd(job_ids),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return _parse_status_lines(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fall back to squeue.
    try:
        result = subprocess.run(  # noqa: S603
            _squeue_cmd(job_ids),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return _parse_status_lines(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return {}
