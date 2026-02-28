# Developer's Guide

## Architecture

pytask-slurm registers as a pytask plugin via the `pytask_slurm = "pytask_slurm.plugin"`
entry point. When `--slurm` is active, `pytask_post_parse` registers the execute and
logging modules, which take over task execution.

### Execution flow

1. **Config** (`config.py`) — Registers the `@pytask.mark.slurm()` marker and
   conditionally activates the SLURM executor via `pytask_post_parse`.
2. **Submit** (`submit.py`) — Serializes tasks with cloudpickle, writes a `sys.path`
   JSON sidecar, and calls `sbatch --wrap="python -m pytask_slurm.runner <payload>
   <result>"`.
3. **Poll** (`monitor.py`) — Queries `sacct` for job statuses (with `squeue` fallback),
   parses SLURM state strings including suffixes like `CANCELLED by 12345` and variant
   names like `COMPLETING`.
4. **Execute** (`execute.py`) — Three-phase loop matching pytask-parallel's design:
   submit ready tasks, poll statuses, process completed/failed jobs. Cancels remaining
   jobs on exit.
5. **Worker** (`runner/`) — Runs inside the SLURM job. Restores `sys.path`, deserializes
   payload, executes the task, captures stdout/stderr/warnings, writes result pickle.

### Key design patterns

- **Cloudpickle interchange** — Task payloads and results are serialized to
  `.pytask/slurm/*.pkl` files; `cloudpickle.register_pickle_by_value()` handles
  module-local functions.
- **sys.path sidecar** — A JSON file alongside the payload ensures the worker can import
  task modules that pytask loaded dynamically.
- **Graceful degradation** — `poll_job_statuses()` and `cancel_jobs()` catch all
  subprocess exceptions so missing SLURM commands don't crash the scheduler.
- **Per-task overrides** — `@pytask.mark.slurm(mem="16G", cpus_per_task=4)` merges with
  global CLI defaults; validated at submission time in `_get_slurm_options()`.

## Running tests

```console
$ pixi run tests                    # all tests
$ pixi run pytest tests/test_submit.py  # single file
$ pixi run tests-with-cov           # with coverage
```

Integration tests in `test_slurm_mock.py` use mock `sbatch`/`sacct`/`scancel`/`squeue`
scripts in `tests/mock_slurm/` that run the task payload directly instead of going
through a real SLURM scheduler.

## Type checking

```console
$ pixi run ty
```

## Linting

```console
$ pixi run prek run --all-files
```
