# Quickstart

## Installation

pytask-slurm is available on [PyPI](https://pypi.org/project/pytask-slurm). Install it
with

```console
$ uv add pytask-slurm

# or

$ pixi add pytask-slurm
```

## Usage

To submit tasks as SLURM jobs, pass the `--slurm` flag when building your project.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm = true
```
````
`````

```{important}
It is not possible to combine SLURM execution with debugging. `--pdb`, `--trace`, and
`--dry-run` all deactivate the SLURM executor.
```

By default pytask-slurm requests **1 CPU**, **4 GB memory**, and a **1-hour time limit**
per job. Override these globally on the command line or per task with markers — see
{doc}`configuration` for the full reference.

## Per-task resource overrides

Use the `@pytask.mark.slurm` decorator to override SLURM options for individual tasks.

```python
import pytask


@pytask.mark.slurm(mem="16G", cpus_per_task=4, time="04:00:00")
def task_heavy_computation(output=Path("result.pkl")):
    ...
```

Marker values take precedence over the global CLI/config defaults for that task. See
{doc}`configuration` for the full list of supported keys.

## How it works

pytask-slurm follows a three-phase execution loop:

1. **Submit** — Ready tasks are serialized with cloudpickle and submitted via `sbatch
   --wrap`. Each job runs `python -m pytask_slurm.runner <payload> <result>`.
2. **Poll** — `sacct` is queried periodically (with a `squeue` fallback) to track job
   states (pending, running, completed, failed, etc.).
3. **Collect** — Result pickles are deserialized and fed back into pytask's reporting
   pipeline. On exit, any remaining jobs are cancelled via `scancel`.
