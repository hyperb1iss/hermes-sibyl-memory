"""Hermes registration entry point for the Sibyl memory provider."""


def register(ctx) -> None:
    from .provider import SibylMemoryProvider

    ctx.register_memory_provider(SibylMemoryProvider())


__all__ = ["register"]
