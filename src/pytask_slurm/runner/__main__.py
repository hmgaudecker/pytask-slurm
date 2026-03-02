"""Allow ``python -m pytask_slurm.runner <payload> <result>``."""

from __future__ import annotations

import sys
import traceback

_EXPECTED_ARGC = 3

if len(sys.argv) != _EXPECTED_ARGC:
    print(  # noqa: T201
        "Usage: python -m pytask_slurm.runner <payload.pkl> <result.pkl>",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from pytask_slurm.runner import run_task

    run_task(sys.argv[1], sys.argv[2])
except Exception:  # noqa: BLE001
    # Ensure any crash (including import errors) is visible in the SLURM log.
    traceback.print_exc()
    sys.exit(2)
