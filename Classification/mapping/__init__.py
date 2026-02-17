"""Alpha mapping package."""

__all__ = ["run_mapping"]


def run_mapping(*args, **kwargs):
    from Classification.mapping.mapping import run_mapping as _run

    return _run(*args, **kwargs)
