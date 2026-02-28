# Changes

This is a record of all past pytask-slurm releases and what went into them in reverse
chronological order. Releases follow [semantic versioning](https://semver.org/) and all
releases are available on [PyPI](https://pypi.org/project/pytask-slurm).

## 0.1.0 - Unreleased

- Initial release.
- Submit pytask tasks as SLURM jobs via `sbatch`.
- Poll job statuses via `sacct` with `squeue` fallback.
- Per-task resource overrides with `@pytask.mark.slurm(...)`.
- CLI options: `--slurm`, `--slurm-partition`, `--slurm-time`, `--slurm-mem`,
  `--slurm-cpus-per-task`, `--slurm-max-jobs`, `--slurm-poll-interval`,
  `--slurm-account`, `--slurm-qos`, `--slurm-extra`.
- Session header logging summarizing SLURM configuration.
- Graceful cancellation of remaining jobs on exit.
