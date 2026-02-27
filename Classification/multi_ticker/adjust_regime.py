"""Compatibility wrapper for DB.adjust_regime."""

from DB.adjust_regime import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("DB.adjust_regime", run_name="__main__")
