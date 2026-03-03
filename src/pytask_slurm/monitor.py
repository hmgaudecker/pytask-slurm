"""Monitor SLURM job statuses via sacct with squeue fallback."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SlurmJobResult:
    """Status and exit code for a SLURM job."""

    status: SlurmJobStatus
    exit_code: int | None = None


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


def _parse_exit_code(raw: str) -> int | None:
    """Parse SLURM's ``exit:signal`` exit code format (e.g. ``"2:0"`` → 2).

    Returns ``None`` if the format is unrecognised.
    """
    if not raw:
        return None
    code_part = raw.split(":")[0]
    try:
        return int(code_part)
    except ValueError:
        return None


def _parse_status_lines(
    stdout: str, *, has_exit_code: bool = True
) -> dict[str, SlurmJobResult]:
    """Parse ``jobid|STATE[|ExitCode]`` lines into a result mapping.

    When *has_exit_code* is False (squeue fallback), exit_code is set to None.
    """
    results: dict[str, SlurmJobResult] = {}
    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 2:  # noqa: PLR2004
            job_id, state_str = parts[0], parts[1]
            exit_code: int | None = None
            if has_exit_code and len(parts) >= 3:  # noqa: PLR2004
                exit_code = _parse_exit_code(parts[2])
            results[job_id] = SlurmJobResult(
                status=_parse_state(state_str), exit_code=exit_code
            )
    return results


def _sacct_cmd(job_ids: list[str]) -> list[str]:
    return [
        "sacct",
        "-X",  # no sub-steps
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,State,ExitCode",
        "--jobs=" + ",".join(job_ids),
    ]


def _squeue_cmd(job_ids: list[str]) -> list[str]:
    return [
        "squeue",
        "--noheader",
        "--format=%i|%T",
        "--jobs=" + ",".join(job_ids),
    ]


def poll_job_statuses(job_ids: list[str]) -> dict[str, SlurmJobResult]:
    """Query sacct for the status of the given SLURM job IDs.

    Falls back to squeue when sacct is unavailable or fails.
    Returns a mapping from job ID to :class:`SlurmJobResult`. Jobs not found in
    the output are omitted from the result (the caller should treat them as
    still pending).

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
            return _parse_status_lines(result.stdout, has_exit_code=True)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fall back to squeue.  Note: squeue only reports jobs still in the
    # scheduler queue.  Once a job finishes it disappears from squeue output,
    # so this fallback cannot detect terminal states (COMPLETED, FAILED, etc.).
    # It is therefore only useful while jobs are still queued or running.  If
    # sacct is persistently unavailable, callers should implement a timeout or
    # result-file-based detection to avoid polling indefinitely.
    try:
        result = subprocess.run(  # noqa: S603
            _squeue_cmd(job_ids),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return _parse_status_lines(result.stdout, has_exit_code=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return {}
