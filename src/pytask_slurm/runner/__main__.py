"""Allow ``python -m pytask_slurm.runner <payload> <result>``."""

from __future__ import annotations

import sys

from pytask_slurm.runner import run_task

_EXPECTED_ARGC = 3

if len(sys.argv) != _EXPECTED_ARGC:
    print(  # noqa: T201
        "Usage: python -m pytask_slurm.runner <payload.pkl> <result.pkl>",
        file=sys.stderr,
    )
    sys.exit(1)

run_task(sys.argv[1], sys.argv[2])
