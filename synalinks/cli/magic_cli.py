#!/usr/bin/python3
# -*- coding: utf-8 -*-

import click

from synalinks.cli.banners import banner
from synalinks.cli.utils.setup_utils import base_setup_config
from synalinks.cli.utils.setup_utils import models_setup_config
from synalinks.cli.utils.setup_utils import secrets_setup_config
from synalinks.cli.utils.setup_utils import setup_project
from synalinks.src.version import version as get_version
from synalinks.src.backend.config import maybe_initialize_telemetry


@click.group()
def magic_cli():
    """Synalinks AI powered command-line interface."""
    maybe_initialize_telemetry()


@magic_cli.command()
def version():
    """Print synalinks version"""
    click.echo(banner())
    click.echo(f"Version: {get_version()}")


@magic_cli.command()
def memory():
    """Synalinks memory"""
    click.echo(banner())


@magic_cli.command()
def init():
    """Initializes a new project"""
    click.echo(banner())
    config = base_setup_config()
    config = models_setup_config(config)
    secrets = secrets_setup_config(config)
    setup_project(config, secrets)


if __name__ == "__main__":
    magic_cli()
