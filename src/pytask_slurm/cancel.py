"""Cancel SLURM jobs via scancel."""

from __future__ import annotations

import subprocess


def cancel_jobs(job_ids: list[str]) -> None:
    """Cancel SLURM jobs. Best-effort: never raises."""
    if not job_ids:
        return

    try:
        subprocess.run(
            ["scancel", *job_ids],
            capture_output=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        pass
