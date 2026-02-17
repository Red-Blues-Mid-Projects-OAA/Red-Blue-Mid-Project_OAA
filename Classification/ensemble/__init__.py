"""Equal-weight ensemble package."""

__all__ = ["run_equal_weight_ensemble"]


def run_equal_weight_ensemble(*args, **kwargs):
    from Classification.ensemble.ensemble import run_equal_weight_ensemble as _run

    return _run(*args, **kwargs)
