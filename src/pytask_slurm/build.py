"""Extend the build command with SLURM options."""

from __future__ import annotations

import click
from pytask import hookimpl


@hookimpl
def pytask_extend_command_line_interface(cli: click.Group) -> None:
    """Extend the command line interface."""
    additional_parameters = [
        click.Option(
            ["--slurm"],
            is_flag=True,
            default=False,
            help="Submit tasks as SLURM jobs via sbatch.",
        ),
        click.Option(
            ["--slurm-partition"],
            default=None,
            help="SLURM partition.",
        ),
        click.Option(
            ["--slurm-time"],
            default="01:00:00",
            help="Time limit per job (HH:MM:SS).",
        ),
        click.Option(
            ["--slurm-mem"],
            default="4G",
            help="Memory per job.",
        ),
        click.Option(
            ["--slurm-cpus-per-task"],
            default=1,
            type=int,
            help="CPUs per task.",
        ),
        click.Option(
            ["--slurm-max-jobs"],
            default=100,
            type=int,
            help="Max concurrent SLURM jobs.",
        ),
        click.Option(
            ["--slurm-poll-interval"],
            default=5.0,
            type=float,
            help="Seconds between sacct polls.",
        ),
        click.Option(
            ["--slurm-account"],
            default=None,
            help="SLURM account.",
        ),
        click.Option(
            ["--slurm-qos"],
            default=None,
            help="SLURM quality-of-service.",
        ),
        click.Option(
            ["--slurm-extra"],
            default=None,
            help="Extra sbatch flags (e.g. '--gres=gpu:1 --constraint=a100').",
        ),
        click.Option(
            ["--slurm-unknown-timeout"],
            default=600,
            type=int,
            help="Seconds a job may stay UNKNOWN before being treated as failed.",
        ),
    ]
    cli.commands["build"].params.extend(additional_parameters)
