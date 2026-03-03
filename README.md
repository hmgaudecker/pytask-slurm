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

This submits every task as a separate SLURM job using default resources (4 GB memory, 1
CPU, 1-hour time limit). Customize resources via CLI flags, configuration file, or
per-task decorators — see below.

## Configuring SLURM options

There are three ways to set SLURM job resources, listed from lowest to highest priority:

### 1. CLI flags

Pass flags directly when invoking pytask:

```console
pytask build --slurm --slurm-partition=batch --slurm-mem=8G --slurm-time=02:00:00
```

### 2. Configuration file (`pyproject.toml`)

Set defaults in your project's `pyproject.toml` so you don't have to repeat CLI flags:

```toml
[tool.pytask.ini_options]
slurm = true
slurm_partition = "batch"
slurm_time = "02:00:00"
slurm_mem = "8G"
slurm_cpus_per_task = 2
slurm_account = "myaccount"
```

### 3. Per-task decorator

Override resources for individual tasks with `@pytask.mark.slurm(...)`:

```python
import pytask


@pytask.mark.slurm(mem="16G", cpus_per_task=4, time="04:00:00")
def task_heavy_computation(): ...


@pytask.mark.slurm(partition="gpu", gpus=1)
def task_gpu_training(): ...
```

### Priority / merging

Options are resolved in this order:

1. **Built-in defaults** (e.g. `time="01:00:00"`, `mem="4G"`, `cpus_per_task=1`)
2. **`pyproject.toml`** values override defaults
3. **CLI flags** override config values
4. **Per-task decorator** overrides everything else

## Available SLURM options

### Job resource options

| CLI flag | Config key | Decorator key | Type | Default | Description |
|---|---|---|---|---|---|
| `--slurm-partition` | `slurm_partition` | `partition` | str | *None* | SLURM partition |
| `--slurm-time` | `slurm_time` | `time` | str | `"01:00:00"` | Time limit (HH:MM:SS) |
| `--slurm-mem` | `slurm_mem` | `mem` | str | `"4G"` | Memory per job |
| `--slurm-cpus-per-task` | `slurm_cpus_per_task` | `cpus_per_task` | int | `1` | CPUs per task |
| `--slurm-account` | `slurm_account` | `account` | str | *None* | SLURM account |
| `--slurm-qos` | `slurm_qos` | `qos` | str | *None* | Quality-of-service |
| `--slurm-gpus` | `slurm_gpus` | `gpus` | int | *None* | GPUs per task |

All options are optional and only passed to sbatch when set. `time`, `mem`, and
`cpus_per_task` have built-in defaults (`"01:00:00"`, `"4G"`, `1`) so they are
always present unless explicitly set to `None`.

### Executor options

| CLI flag | Config key | Type | Default | Description |
|---|---|---|---|---|
| `--slurm-max-jobs` | `slurm_max_jobs` | int | `100` | Max concurrent SLURM jobs |
| `--slurm-poll-interval` | `slurm_poll_interval` | float | `5.0` | Seconds between sacct polls |
| `--slurm-unknown-timeout` | `slurm_unknown_timeout` | int | `600` | Seconds a job may stay UNKNOWN before treated as failed |

### Extra sbatch flags

| CLI flag | Config key | Decorator key | Type |
|---|---|---|---|
| `--slurm-extra` | `slurm_extra` | `extra` | str |

See the next section for details.

## Extra sbatch flags

For sbatch options not covered by the built-in keys (e.g. email notifications,
constraints, exclusive mode), use `extra`.

### Via CLI or `pyproject.toml` (string)

Pass a string of raw sbatch flags:

```console
pytask build --slurm --slurm-extra="--mail-user=user@example.com --mail-type=ALL"
```

```toml
[tool.pytask.ini_options]
slurm_extra = "--mail-user=user@example.com --mail-type=ALL --constraint=a100"
```

### Via per-task decorator (string)

Pass a string of raw sbatch flags, just like the CLI/config form:

```python
@pytask.mark.slurm(
    mem="32G",
    extra="--mail-user=user@example.com --mail-type=END --constraint=a100 --exclusive",
)
def task_big_job(): ...
```

### Combining both sources

`extra` follows the same merging rules as all other SLURM options: the decorator value
overrides the config value. If only the config sets `extra`, those flags are used. If the
decorator also sets `extra`, it replaces the config value entirely. A warning is logged
if `extra` contains flags that pytask-slurm already generates automatically (e.g.
`--time`, `--mem`, `--partition`).

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

## Documentation

Find the full documentation at
[pytask-slurm.readthedocs.io](https://pytask-slurm.readthedocs.io/).

## Acknowledgements

The SLURM interaction design (job submission, status polling, state mapping) draws on
ideas from
[snakemake-executor-plugin-slurm](https://github.com/snakemake/snakemake-executor-plugin-slurm)
by the Snakemake Community, licensed under the MIT License. See [NOTICE](NOTICE) for
details.
