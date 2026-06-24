# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Commands

```bash
pixi run tests                # run all tests
pixi run pytest tests/test_submit.py  # run single test file
pixi run pytest tests/test_submit.py::TestGetSlurmOptionsValidation::test_cpus_per_task_bool_rejected  # single test
pixi run tests-with-cov       # tests with coverage
pixi run ty                   # type check with ty
pixi run prek run --all-files  # run all pre-commit hooks (ruff, pyproject-fmt, yamlfix, etc.)
pixi install                  # install/update dependencies and lock file
```

## Architecture

pytask-slurm is a **pytask plugin** that submits tasks as SLURM jobs on HPC clusters. It
registers via the `pytask_slurm = "pytask_slurm.plugin"` entry-point.

### Execution flow

1. **Config** (`config.py`): Registers `@pytask.mark.slurm()` marker and conditionally
   activates the SLURM executor via `pytask_post_parse` hook
1. **Submit** (`submit.py`): Serializes tasks with cloudpickle, writes a `sys.path` JSON
   sidecar, and calls `sbatch --wrap="python -m pytask_slurm.runner <payload> <result>"`
1. **Poll** (`monitor.py`): Queries `sacct` for job statuses, parses SLURM state strings
   (handles suffixes like "CANCELLED by 12345" and variant names)
1. **Execute** (`execute.py`): Three-phase loop matching pytask-parallel's design —
   submit ready tasks, poll statuses, process completed/failed jobs. Cancels remaining
   jobs on exit.
1. **Worker** (`runner/`): Runs in isolated SLURM subprocess — restores `sys.path`,
   deserializes payload, executes task, captures stdout/stderr/warnings, writes result
   pickle

### Key design patterns

- **Cloudpickle interchange**: Task payloads and results are serialized to
  `.pytask/slurm/*.pkl` files; `cloudpickle.register_pickle_by_value()` handles
  module-local functions
- **sys.path sidecar**: JSON file alongside payload ensures the worker can import task
  modules that pytask loaded dynamically
- **Graceful degradation**: `poll_job_statuses()` and `cancel_jobs()` catch all
  exceptions so missing SLURM commands don't crash the scheduler
- **Per-task overrides**: `@pytask.mark.slurm(mem="16G", cpus_per_task=4)` overrides
  `pyproject.toml` values; validated at submission time in `_get_slurm_options()`. There
  are no built-in defaults — all options must be explicitly configured.

### Test structure

- `test_submit.py` / `test_monitor.py` / `test_cancel.py`: Unit tests mocking subprocess
  calls
- `test_slurm_mock.py`: Integration tests using mock sbatch/sacct/scancel scripts in
  `tests/mock_slurm/`; a fixture prepends mock dir to PATH and sets `MOCK_SLURM_STATE`
- `conftest.py`: Autouse fixture restoring `sys.path` and `sys.modules` after each test

### Dependencies

Core: pytask (>=0.5.2), pytask-parallel, cloudpickle, attrs
