# Configuration

All SLURM options can be set on the command line, in `pyproject.toml`, or (where noted)
per task via the `@pytask.mark.slurm` decorator.

## CLI options and configuration values

### `--slurm`

Enable the SLURM executor.

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

### `--slurm-partition`

SLURM partition to submit to. Default: not set (uses the cluster default). Overridable
per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-partition=gpu
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_partition = "gpu"
```
````
````{tab-item} Marker
:sync: marker

```python
@pytask.mark.slurm(partition="gpu")
def task_train(): ...
```
````
`````

### `--slurm-time`

Wall-clock time limit per job in `HH:MM:SS` format. Default: `01:00:00`. Overridable per
task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-time=04:00:00
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_time = "04:00:00"
```
````
````{tab-item} Marker
:sync: marker

```python
@pytask.mark.slurm(time="04:00:00")
def task_long_running(): ...
```
````
`````

### `--slurm-mem`

Memory per job (e.g. `4G`, `500M`). Default: `4G`. Overridable per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-mem=16G
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_mem = "16G"
```
````
````{tab-item} Marker
:sync: marker

```python
@pytask.mark.slurm(mem="16G")
def task_big(): ...
```
````
`````

### `--slurm-cpus-per-task`

Number of CPUs per job. Default: `1`. Overridable per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-cpus-per-task=8
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_cpus_per_task = 8
```
````
````{tab-item} Marker
:sync: marker

```python
@pytask.mark.slurm(cpus_per_task=8)
def task_parallel_within(): ...
```
````
`````

### `--slurm-account`

SLURM account for job accounting. Default: not set. Overridable per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-account=myproject
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_account = "myproject"
```
````
````{tab-item} Marker
:sync: marker

```python
@pytask.mark.slurm(account="myproject")
def task_billed(): ...
```
````
`````

### `--slurm-qos`

SLURM quality-of-service. Default: not set. Overridable per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-qos=high
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_qos = "high"
```
````
````{tab-item} Marker
:sync: marker

```python
@pytask.mark.slurm(qos="high")
def task_priority(): ...
```
````
`````

### `--slurm-max-jobs`

Maximum number of SLURM jobs running concurrently. Default: `100`. Not overridable per
task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-max-jobs=50
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_max_jobs = 50
```
````
`````

### `--slurm-poll-interval`

Seconds between `sacct` polls. Default: `5.0`. Not overridable per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-poll-interval=10
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_poll_interval = 10
```
````
`````

### `--slurm-extra`

Pass-through string for arbitrary `sbatch` flags not covered by the options above. The
string is split with `shlex.split()` and appended to the `sbatch` command. Default: not
set. Not overridable per task.

`````{tab-set}
````{tab-item} CLI
:sync: cli

```console
$ pytask build --slurm --slurm-extra="--gres=gpu:1 --constraint=a100"
```
````
````{tab-item} Configuration
:sync: configuration

```toml
[tool.pytask.ini_options]
slurm_extra = "--gres=gpu:1 --constraint=a100"
```
````
`````

## `@pytask.mark.slurm` reference

The marker accepts the following keyword arguments (positional arguments are not
allowed):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `partition` | `str` | from config | SLURM partition |
| `time` | `str` | `"01:00:00"` | Wall-clock limit (`HH:MM:SS`) |
| `mem` | `str` | `"4G"` | Memory per job |
| `cpus_per_task` | `int` | `1` | CPUs per task |
| `account` | `str` | from config | SLURM account |
| `qos` | `str` | from config | Quality-of-service |

Only one `@pytask.mark.slurm(...)` decorator per task is allowed. If you need to set
multiple options, pass them all as keyword arguments to a single decorator:

```python
@pytask.mark.slurm(partition="gpu", mem="32G", time="08:00:00", qos="high")
def task_train_model(): ...
```
