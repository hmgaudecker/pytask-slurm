"""Cancel SLURM jobs via scancel."""

from __future__ import annotations

import contextlib
import subprocess


def cancel_jobs(job_ids: list[str]) -> None:
    """Cancel SLURM jobs. Best-effort: never raises."""
    if not job_ids:
        return

    with contextlib.suppress(Exception):
        subprocess.run(
            ["scancel", *job_ids],
            capture_output=True,
            check=False,
            timeout=30,
        )
