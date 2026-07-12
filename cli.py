"""Minimal operator command registration for the Sibyl provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import config_path, load_api_key, load_provider_config, resolve_hermes_home

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


def _status() -> None:
    home = resolve_hermes_home()
    config = load_provider_config(home)
    print(f"Sibyl config: {config_path(home)}")
    print(f"Configured: {'yes' if config.is_complete else 'no'}")
    print(f"API key: {'set' if load_api_key(home) else 'missing'}")
    if config.project_id:
        print(f"Project: {config.project_id}")
    if config.memory_space_id:
        print(f"Memory space: {config.memory_space_id}")


def sibyl_command(args: Namespace) -> None:
    command = getattr(args, "sibyl_command", None) or "status"
    if command == "status":
        _status()
        return
    if command == "config":
        print(config_path())
        return
    print(f"hermes sibyl {command} is not available in this provider version.")


def register_cli(subparser: ArgumentParser) -> None:
    commands = subparser.add_subparsers(dest="sibyl_command")
    commands.add_parser("status", help="Show Sibyl provider configuration")
    commands.add_parser("doctor", help="Check Sibyl provider health and scope")
    commands.add_parser("flush", help="Replay durable pending mutations")
    commands.add_parser("config", help="Show the provider configuration path")
    subparser.set_defaults(func=sibyl_command)
