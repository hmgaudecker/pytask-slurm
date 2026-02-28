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

## Usage

```console
pytask build --slurm
```

Override resources per task with markers:

```python
import pytask


@pytask.mark.slurm(mem="16G", cpus_per_task=4, time="04:00:00")
def task_heavy_computation(): ...
```

## Documentation

Find the full documentation at
[pytask-slurm.readthedocs.io](https://pytask-slurm.readthedocs.io/).

## Acknowledgements

The SLURM interaction design (job submission, status polling, state mapping) draws on
ideas from
[snakemake-executor-plugin-slurm](https://github.com/snakemake/snakemake-executor-plugin-slurm)
by the Snakemake Community, licensed under the MIT License. See [NOTICE](NOTICE) for
details.
