# pytask-slurm

[![PyPI](https://img.shields.io/pypi/v/pytask-slurm?color=blue)](https://pypi.org/project/pytask-slurm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytask-slurm)](https://pypi.org/project/pytask-slurm)
[![PyPI - License](https://img.shields.io/pypi/l/pytask-slurm)](https://pypi.org/project/pytask-slurm)

pytask-slurm allows you to submit tasks defined with
[pytask](https://pytask-dev.readthedocs.io/) as SLURM jobs on HPC clusters.

## Installation

```console
pip install pytask-slurm
```

## Quick start

```console
pytask build --slurm
```

This submits every task as a separate SLURM job. You must configure resources (memory,
CPUs, time limit, etc.) via `pyproject.toml` or per-task decorators — see below.

## Configuring SLURM options

There are two ways to set SLURM job resources:

### 1. Configuration file (`pyproject.toml`)

Set defaults in your project's `pyproject.toml` so every task picks them up
automatically:

```toml
[tool.pytask.ini_options]
slurm = true
slurm_partition = "batch"
slurm_time = "02:00:00"
slurm_mem = "8G"
slurm_cpus_per_task = 2
slurm_account = "myaccount"
```

### 2. Per-task decorator

Override resources for individual tasks with `@pytask.mark.slurm(...)`:

```python
import pytask


@pytask.mark.slurm(mem="16G", cpus_per_task=4, time="04:00:00")
def task_heavy_computation(): ...


@pytask.mark.slurm(partition="gpu", gpus=1)
def task_gpu_training(): ...
```

## Available SLURM options

### Job resource options

| pyproject.toml key    | Decorator key   | Type | Description           |
| --------------------- | --------------- | ---- | --------------------- |
| `slurm_partition`     | `partition`     | str  | SLURM partition       |
| `slurm_time`          | `time`          | str  | Time limit (HH:MM:SS) |
| `slurm_mem`           | `mem`           | str  | Memory per job        |
| `slurm_cpus_per_task` | `cpus_per_task` | int  | CPUs per task         |
| `slurm_account`       | `account`       | str  | SLURM account         |
| `slurm_qos`           | `qos`           | str  | Quality-of-service    |
| `slurm_gpus`          | `gpus`          | int  | GPUs per task         |

All options are only passed to sbatch when explicitly set. There are no built-in
defaults — you must configure every option you need.

### Executor options (pytask-slurm internal, not passed to sbatch)

| pyproject.toml key      | Type  | Default | Description                                             |
| ----------------------- | ----- | ------- | ------------------------------------------------------- |
| `slurm_max_jobs`        | int   | 50      | Max concurrent SLURM jobs                               |
| `slurm_poll_interval`   | float | 10      | Seconds between sacct polls                             |
| `slurm_unknown_timeout` | int   | 300     | Seconds a job may stay UNKNOWN before treated as failed |

### Extra sbatch flags

| pyproject.toml key | Decorator key | Type |
| ------------------ | ------------- | ---- |
| `slurm_extra`      | `extra`       | str  |

See the next section for details.

## Extra sbatch flags

For sbatch options not covered by the built-in keys (e.g. email notifications,
constraints, exclusive mode), use `extra`.

### Via `pyproject.toml` (string)

Pass a string of raw sbatch flags:

```toml
[tool.pytask.ini_options]
slurm_extra = "--mail-user=user@example.com --mail-type=ALL --constraint=a100"
```

### Via per-task decorator (string)

Pass a string of raw sbatch flags, just like the config form:

```python
@pytask.mark.slurm(
    mem="32G",
    extra="--mail-user=user@example.com --mail-type=END --constraint=a100 --exclusive",
)
def task_big_job(): ...
```

### Combining both sources

`extra` follows the same merging rules as all other SLURM options: the decorator value
overrides the config value. If only the config sets `extra`, those flags are used. If
the decorator also sets `extra`, it replaces the config value entirely.

## Using with other pytask flags

All standard pytask flags work alongside `--slurm`:

```console
pytask build --slurm --dry-run          # discover tasks without submitting
pytask build --slurm -x                 # stop after first failure
pytask build --slurm -s                 # show task output
pytask build --slurm -f                 # force re-run all tasks
pytask build --slurm -m "not gpu"       # run only tasks without the "gpu" marker
```

Note that `--pdb`, `--trace`, and `--dry-run` disable the SLURM executor — tasks run
locally instead.

## Acknowledgements

The SLURM interaction design (job submission, status polling, state mapping) draws on
ideas from
[snakemake-executor-plugin-slurm](https://github.com/snakemake/snakemake-executor-plugin-slurm)
by the Snakemake Community, licensed under the MIT License. See [NOTICE](NOTICE) for
details.
